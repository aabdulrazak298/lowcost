"""Telegram bot integration for LowCostLLM — polling mode with rich formatting.

Uses telegramify-markdown + sendRichMessage for native tables, code blocks,
and formatting. Falls back to entity-based sendMessage when needed.
"""

import asyncio
import io
import logging
import httpx
from telegram import Update, MessageEntity
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegramify_markdown import convert, richify

from processor import process_query, get_last_cache_id
from llm import set_delivery_context, clear_delivery_context
from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS, AVAILABLE_MODELS,
    get_cheap_model, set_cheap_model, get_expensive_model, set_expensive_model,
    get_cheap_fallback_model, set_cheap_fallback_model,
    get_voice_enabled, set_voice_enabled, get_voice_engine, set_voice_engine,
)

logger = logging.getLogger("lowcostllm.tg")

_tg_bot_app: Application | None = None
TG_CHAR_LIMIT = 4096

# ── Rich formatting helpers (ThinkLLM-style) ───────────────────


def _chunk_text(text: str, limit: int = TG_CHAR_LIMIT) -> list[str]:
    """Split long text at paragraph boundaries (ported from ThinkLLM).

    Never truncates mid-content: prefers ``\\n\\n`` then ``\\n``, falls back to a
    hard cut only when a single line exceeds the limit.
    """
    if len(text) <= limit:
        return [text]
    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 4:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    return chunks


async def _send_rich(chat_id: int, markdown_text: str) -> bool:
    """Send via sendRichMessage with native tables, headings, etc. Falls back gracefully."""
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            rich_msg = richify(markdown_text)
            payload = {
                "chat_id": chat_id,
                "rich_message": rich_msg.to_dict(),
                "link_preview_options": {"is_disabled": True},
            }
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendRichMessage",
                    json=payload,
                )
                if r.status_code != 200:
                    logger.warning("sendRichMessage failed: %s %s", r.status_code, r.text[:200])
                    return False
                return True
        except Exception as e:
            last_exc = e
            logger.warning("sendRichMessage exception (attempt %s): %r", attempt, e)
            if attempt == 1:
                await asyncio.sleep(1.0)
    logger.warning("sendRichMessage gave up after 2 attempts: %r", last_exc)
    return False


async def _send_entity_fallback(chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fallback: convert markdown → entities and send via sendMessage.

    NOTE: telegramify's ``convert()`` returns its OWN ``MessageEntity`` class
    (telegramify_markdown.entity.MessageEntity), which python-telegram-bot cannot
    serialize — passing them raw raises ``TypeError: not JSON serializable``.
    Map to PTB's ``telegram.MessageEntity`` first.
    """
    try:
        tg_text, entities = convert(text)
        ptb_entities = [
            MessageEntity(
                type=e.type,
                offset=e.offset,
                length=e.length,
                url=getattr(e, "url", None),
                language=getattr(e, "language", None),
                custom_emoji_id=getattr(e, "custom_emoji_id", None),
            )
            for e in entities
        ]
        chunks = _chunk_text(tg_text)
        if len(chunks) == 1:
            # Single chunk ≤ limit — entity offsets stay valid, keep formatting.
            await context.bot.send_message(
                chat_id=chat_id, text=chunks[0], entities=ptb_entities,
            )
        else:
            # Multi-chunk: entity offsets would break across splits — send plain.
            for chunk in chunks:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
        return
    except Exception as e:
        logger.warning("Entity send failed: %r", e)
    # Last resort: plain text — MUST never raise (an escaping exception skips
    # stop_typing.set() downstream and leaks the typing task forever).
    try:
        for chunk in _chunk_text(text):
            await context.bot.send_message(chat_id=chat_id, text=chunk)
    except Exception as e:
        logger.error(
            "All send paths failed for chat %s — response lost. Response was:\n%s",
            chat_id, text[:2000],
        )


async def _send_response(chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a response with rich formatting, falling back gracefully."""
    # Try rich first
    if await _send_rich(chat_id, text):
        return

    # Fall back to entity-based
    await _send_entity_fallback(chat_id, text, context)


async def _send_voice_response(
    update: Update, chat_id: int, answer_text: str, full_text: str,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Voice mode: send audio bubble + hidden (spoiler) transcript.

    `answer_text` = clean answer (no footer) → TTS audio.
    `full_text` = answer + calling card → spoiler transcript, blurred until tapped.
    Falls back to normal rich text if TTS fails for any reason.
    """
    try:
        from tts import text_to_voice_ogg, clean_for_display

        ogg = await text_to_voice_ogg(answer_text)
        buf = io.BytesIO(ogg)
        buf.name = "voice.ogg"
        await update.message.reply_voice(voice=buf)

        transcript = clean_for_display(full_text)
        if transcript:
            # Spoiler entity over the whole message → blurred until tapped
            limit = 3800
            for i in range(0, len(transcript), limit):
                chunk = transcript[i:i + limit]
                await update.message.reply_text(
                    chunk,
                    entities=[MessageEntity(MessageEntity.SPOILER, 0, len(chunk))],
                )
    except Exception as e:
        logger.exception("Voice send failed — falling back to text")
        await _send_response(chat_id, full_text, context)


# ── Command handlers ────────────────────────────────────────────


async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Hello! I'm LowCostLLM — ask me anything!\n\n"
        "Commands:\n"
        "/model — show/switch models\n"
        "/voice — voice replies (on/off, edge/kokoro)\n"
        "/new — clear session\n"
        "/help — show this message"
    )


async def _help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cheap = get_cheap_model()
    expensive = get_expensive_model()
    cheap_fallback = get_cheap_fallback_model()
    await update.message.reply_text(
        f"🤖 **LowCostLLM**\n\n"
        f"Cheap: `{cheap}`\n"
        f"Expensive: `{expensive}`\n"
        f"Cheap fallback: `{cheap_fallback or 'OFF'}`\n\n"
        f"/model -c <name> — swap cheap model\n"
        f"/model -e <name> — swap expensive\n"
        f"/model -cb <name> — cheap fallback model (used if cheap fails, e.g. rate limit)\n"
        f"/model -cb off — disable cheap fallback\n"
        f"/model list — show all available\n"
        f"/voice — voice replies (on/off, edge/kokoro)\n"
        f"/new — clear chat context",
        parse_mode="Markdown",
    )


async def _model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    if not args:
        cheap = get_cheap_model()
        expensive = get_expensive_model()
        cheap_fallback = get_cheap_fallback_model()
        await update.message.reply_text(
            f"Current models:\n• Cheap: `{cheap}`\n• Expensive: `{expensive}`\n"
            f"• Cheap fallback: `{cheap_fallback or 'OFF'}`\n\n"
            f"Use /model list to see options.\n"
            f"/model -c <name> — swap cheap\n"
            f"/model -e <name> — swap expensive\n"
            f"/model -cb <name> — cheap fallback (or off to disable)",
            parse_mode="Markdown",
        )
        return

    sub = args[0].lower()

    if sub == "list":
        lines = ["**Available models:**"]
        for key, (full_id, provider) in AVAILABLE_MODELS.items():
            lines.append(f"• `{key}` → {full_id} ({provider})")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    if sub in ("-c", "-e", "cheap", "expensive"):
        if len(args) < 2:
            current = get_cheap_model() if sub in ("-c", "cheap") else get_expensive_model()
            label = "cheap" if sub in ("-c", "cheap") else "expensive"
            await update.message.reply_text(
                f"Current {label}: `{current}`\n"
                f"Usage: /model {sub} <name>\n"
                f"Use /model list to see options.",
                parse_mode="Markdown",
            )
            return

        key = args[1].lower()
        if key not in AVAILABLE_MODELS:
            await update.message.reply_text(
                f"❌ Unknown model: `{key}`\nUse /model list to see options.",
                parse_mode="Markdown",
            )
            return

        full_id, provider = AVAILABLE_MODELS[key]
        if sub in ("-c", "cheap"):
            set_cheap_model(full_id)
            label = "Cheap"
        else:
            set_expensive_model(full_id)
            label = "Expensive"
        await update.message.reply_text(
            f"✅ {label} → `{full_id}` ({provider})",
            parse_mode="Markdown",
        )
        return

    if sub in ("-cb", "cheapfallback"):
        # Cheap fallback model — used when the primary cheap model keeps
        # failing (rate limit, 5xx, timeout) after its retries.
        if len(args) < 2:
            current = get_cheap_fallback_model()
            await update.message.reply_text(
                f"Current cheap fallback: `{current or 'OFF'}`\n"
                f"Usage: /model -cb <name>  (or /model -cb off to disable)\n"
                f"Use /model list to see options.",
                parse_mode="Markdown",
            )
            return

        key = args[1].lower()
        if key in ("off", "none", "disable", "-"):
            set_cheap_fallback_model(None)
            await update.message.reply_text(
                "✅ Cheap fallback → OFF (no fallback when cheap model fails)",
                parse_mode="Markdown",
            )
            return

        if key not in AVAILABLE_MODELS:
            await update.message.reply_text(
                f"❌ Unknown model: `{key}`\nUse /model list to see options.",
                parse_mode="Markdown",
            )
            return

        full_id, provider = AVAILABLE_MODELS[key]
        set_cheap_fallback_model(full_id)
        await update.message.reply_text(
            f"✅ Cheap fallback → `{full_id}` ({provider})",
            parse_mode="Markdown",
        )
        return

    # Strict: a bare model key without a flag is ambiguous — refuse and show
    # usage instead of silently mutating the cheap model (was a silent trap).
    if sub in AVAILABLE_MODELS:
        await update.message.reply_text(
            f"❌ Missing flag: `{sub}` is a model name, not a command.\n"
            f"Use `/model -c {sub}` for cheap or `/model -e {sub}` for expensive.\n"
            f"Use /model list to see options.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            f"❌ Unknown: `{sub}`\nUse /model list to see options.",
            parse_mode="Markdown",
        )


async def _new_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    cheap = get_cheap_model()
    expensive = get_expensive_model()
    cheap_fallback = get_cheap_fallback_model()
    await update.message.reply_text(
        f"🆕 Session cleared!\n"
        f"Cheap: `{cheap}`\n"
        f"Expensive: `{expensive}`\n"
        f"Cheap fallback: `{cheap_fallback or 'OFF'}`",
        parse_mode="Markdown",
    )


async def _voice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    if not args:
        enabled = get_voice_enabled()
        engine = get_voice_engine()
        await update.message.reply_text(
            f"🔊 Voice replies: {'ON' if enabled else 'OFF'}\n"
            f"Engine: `{engine}`\n\n"
            f"/voice on — enable voice replies\n"
            f"/voice off — disable\n"
            f"/voice edge — free Microsoft edge-tts\n"
            f"/voice kokoro — OpenRouter kokoro-82m",
            parse_mode="Markdown",
        )
        return

    sub = args[0].lower()

    if sub == "on":
        set_voice_enabled(True)
        await update.message.reply_text(
            f"🔊 Voice replies ON (`{get_voice_engine()}`) — I'll read my answers aloud."
        )
    elif sub == "off":
        set_voice_enabled(False)
        await update.message.reply_text("🔇 Voice replies OFF.")
    elif sub in ("edge", "kokoro"):
        set_voice_engine(sub)
        state = "ON" if get_voice_enabled() else "OFF"
        await update.message.reply_text(
            f"🔊 Engine → `{sub}` (voice {state})\n"
            f"Tip: /voice on to enable."
        )
    else:
        await update.message.reply_text(
            "Usage: /voice on | off | edge | kokoro", parse_mode="Markdown"
        )


# ── Message handler ─────────────────────────────────────────────


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message.text
    chat_id = update.effective_chat.id
    user_id = str(update.effective_user.id)

    if not user_msg or not user_msg.strip():
        return

    # Start typing indicator IMMEDIATELY (before any checks)
    stop_typing = asyncio.Event()

    async def _keep_typing():
        while not stop_typing.is_set():
            try:
                async with httpx.AsyncClient(timeout=5) as c:
                    await c.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendChatAction",
                        json={"chat_id": chat_id, "action": "typing"},
                    )
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_typing.wait(), 4)
            except asyncio.TimeoutError:
                pass

    asyncio.create_task(_keep_typing())
    await asyncio.sleep(0)  # Yield so typing task fires immediately

    try:
        # Check allowed users
        if TELEGRAM_ALLOWED_USERS:
            allowed = [u.strip() for u in TELEGRAM_ALLOWED_USERS.split(",") if u.strip()]
            if allowed and user_id not in allowed and str(chat_id) not in allowed:
                await update.message.reply_text("⛔ Access denied.")
                return

        # Build chat history from user_data (list of {role, content} turns)
        history = context.user_data.get("history", [])

        # Process through the shared pipeline
        try:
            # Bind delivery target so image tools can self-deliver via sendPhoto
            set_delivery_context("telegram", chat_id, TELEGRAM_BOT_TOKEN)
            try:
                response, model_used, _images, usage = await process_query(
                    user_query=user_msg,
                    chat_history=history[-10:],
                )
            finally:
                clear_delivery_context()
        except Exception as e:
            logger.exception(f"Query processing failed: {e}")
            try:
                await update.message.reply_text(
                    f"❌ Sorry, something went wrong: {type(e).__name__}\n\n"
                    f"Details: {e}\n\nPlease try again in a moment."
                )
            except Exception as send_err:
                logger.error(f"Error reply also failed: {send_err}")
            return

        # Update history
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": response})
        if len(history) > 20:
            history = history[-20:]
        context.user_data["history"] = history

        # Build calling card with short model name and REAL cost (shared helper)
        from config import build_calling_card

        footer = build_calling_card(model_used, usage, cache_id=get_last_cache_id())
        full_response = response + footer

        # Send — voice bubble + hidden transcript if voice mode is ON
        if get_voice_enabled():
            await _send_voice_response(update, chat_id, response, full_response, context)
        else:
            await _send_response(chat_id, full_response, context)
    finally:
        # ALWAYS stop the typing task — an exception escaping any send path
        # previously skipped this and leaked the task forever (observed:
        # "Telegram error: Timed out" → typing indicator ran indefinitely).
        stop_typing.set()


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Telegram error: {context.error}")


# ── Bot lifecycle ────────────────────────────────────────────────


async def start_bot() -> Application:
    """Create and start the Telegram bot in polling mode."""
    global _tg_bot_app

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not configured")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", _start))
    app.add_handler(CommandHandler("help", _help_cmd))
    app.add_handler(CommandHandler("model", _model_cmd))
    app.add_handler(CommandHandler("voice", _voice_cmd))
    app.add_handler(CommandHandler("new", _new_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    app.add_error_handler(_error_handler)

    await app.initialize()
    await app.updater.start_polling(drop_pending_updates=True)
    await app.start()

    _tg_bot_app = app
    logger.info("Telegram bot started (polling mode)")
    return app


async def stop_bot(app: Application | None = None) -> None:
    global _tg_bot_app
    bot = app or _tg_bot_app
    if bot:
        try:
            await bot.updater.stop()
            await bot.stop()
            await bot.shutdown()
        except Exception:
            logger.exception("Error stopping bot")
    _tg_bot_app = None
    logger.info("Telegram bot stopped")


async def process_telegram_update(body: dict) -> None:
    global _tg_bot_app
    if _tg_bot_app is None:
        raise RuntimeError("Bot not started")
    update = Update.de_json(body, _tg_bot_app.bot)
    await _tg_bot_app.process_update(update)
