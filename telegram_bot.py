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

from processor import process_query
from llm import set_delivery_context, clear_delivery_context
from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS, AVAILABLE_MODELS,
    get_cheap_model, set_cheap_model, get_expensive_model, set_expensive_model,
    get_voice_enabled, set_voice_enabled, get_voice_engine, set_voice_engine,
)

logger = logging.getLogger("lowcostllm.tg")

_tg_bot_app: Application | None = None
TG_CHAR_LIMIT = 4096

# ── Rich formatting helpers (ThinkLLM-style) ───────────────────


async def _send_rich(chat_id: int, markdown_text: str) -> bool:
    """Send via sendRichMessage with native tables, headings, etc. Falls back gracefully."""
    try:
        rich_msg = richify(markdown_text)
        payload = {
            "chat_id": chat_id,
            "rich_message": rich_msg.to_dict(),
            "link_preview_options": {"is_disabled": True},
        }
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendRichMessage",
                json=payload,
            )
            if r.status_code != 200:
                logger.warning("sendRichMessage failed: %s %s", r.status_code, r.text[:200])
                return False
            return True
    except Exception as e:
        logger.warning("sendRichMessage exception: %s", e)
        return False


async def _send_entity_fallback(chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fallback: convert markdown → entities and send via sendMessage."""
    try:
        tg_text, entities = convert(text)
        await context.bot.send_message(
            chat_id=chat_id,
            text=tg_text[:TG_CHAR_LIMIT],
            entities=entities,
        )
    except Exception:
        # Last resort: plain text
        await context.bot.send_message(chat_id=chat_id, text=text[:TG_CHAR_LIMIT])


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
    await update.message.reply_text(
        f"🤖 **LowCostLLM**\n\n"
        f"Cheap: `{cheap}`\n"
        f"Expensive: `{expensive}`\n\n"
        f"/model -c <name> — swap cheap model\n"
        f"/model -e <name> — swap expensive\n"
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
        await update.message.reply_text(
            f"Current models:\n• Cheap: `{cheap}`\n• Expensive: `{expensive}`\n\n"
            f"Use /model list to see options.\n"
            f"/model -c <name> — swap cheap\n"
            f"/model -e <name> — swap expensive",
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

    # Fallback: bare key → set cheap model
    if sub in AVAILABLE_MODELS:
        full_id, provider = AVAILABLE_MODELS[sub]
        set_cheap_model(full_id)
        await update.message.reply_text(
            f"✅ Cheap → `{full_id}` ({provider})",
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
    await update.message.reply_text(
        f"🆕 Session cleared!\n"
        f"Cheap: `{cheap}`\n"
        f"Expensive: `{expensive}`",
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

    # Check allowed users
    if TELEGRAM_ALLOWED_USERS:
        allowed = [u.strip() for u in TELEGRAM_ALLOWED_USERS.split(",") if u.strip()]
        if allowed and user_id not in allowed and str(chat_id) not in allowed:
            stop_typing.set()
            await update.message.reply_text("⛔ Access denied.")
            return

    # Build chat history from user_data
    history = context.user_data.get("history", [])
    chat_history_str = "\n".join(
        f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
        for h in history[-10:]
    )

    # Process through the shared pipeline
    try:
        # Bind delivery target so image tools can self-deliver via sendPhoto
        set_delivery_context("telegram", chat_id, TELEGRAM_BOT_TOKEN)
        try:
            response, model_used, _images = await process_query(
                user_query=user_msg,
                chat_history=chat_history_str,
            )
        finally:
            clear_delivery_context()
    except Exception as e:
        logger.exception(f"Query processing failed: {e}")
        stop_typing.set()
        await update.message.reply_text(
            f"❌ Sorry, something went wrong: {type(e).__name__}\n\n"
            f"Details: {e}\n\nPlease try again in a moment."
        )
        return

    # Update history
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})
    if len(history) > 20:
        history = history[-20:]
    context.user_data["history"] = history

    # Build calling card with short model name and REAL cost
    from config import AVAILABLE_MODELS, MODEL_PRICING
    from llm import get_last_usage, _reset_usage

    # Get real token usage from the LLM call
    usage = get_last_usage()
    _reset_usage()

    # Shorten model name
    short_name = model_used
    for key, (full_id, _) in AVAILABLE_MODELS.items():
        if full_id in model_used:
            short_name = f"{key} (cached)" if "(cached)" in model_used else key
            break

    # Calculate real cost from actual tokens
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    if prompt_tokens or completion_tokens:
        # Find pricing for this model
        price_per_m = 0.28  # default
        for key, (full_id, _) in AVAILABLE_MODELS.items():
            if full_id in model_used:
                price_per_m = MODEL_PRICING.get(key, 0.28)
                break
        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1_000_000) * price_per_m
        cost_str = f" · ${cost:.6f} · {total_tokens} tok"
    else:
        cost_str = ""
    footer = f"\n\n---\n🤖 {short_name}{cost_str}"
    full_response = response + footer

    # Send — voice bubble + hidden transcript if voice mode is ON
    if get_voice_enabled():
        await _send_voice_response(update, chat_id, response, full_response, context)
    else:
        await _send_response(chat_id, full_response, context)
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
