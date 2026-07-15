"""Telegram bot integration for LowCostLLM — webhook mode.

Receives updates via POST to /telegram/webhook (routed through Traefik
at llm.smartdochub.net). Processes messages through the same two-tier
cache → cheap → expensive pipeline as the Flask Chat webhook.
"""
import asyncio
import logging
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS
from processor import process_query_stream
from memory import load_and_prepare_history, save_exchange
from db import clear_user_history
from llm import _wait_for_images

logger = logging.getLogger(__name__)

# Telegram message limit: 4096 characters
TG_CHAR_LIMIT = 4000  # leave some headroom


def _is_allowed(user_id: int) -> bool:
    """Check if user is in the allowed list. Empty list = allow all."""
    allowed = TELEGRAM_ALLOWED_USERS.strip()
    if not allowed:
        return True
    allowed_ids = {int(uid.strip()) for uid in allowed.split(",") if uid.strip()}
    return user_id in allowed_ids


def _split_long_message(text: str, limit: int = TG_CHAR_LIMIT) -> list[str]:
    """Split a long message into Telegram-safe chunks, breaking at newlines."""
    if len(text) <= limit:
        return [text]

    chunks = []
    while len(text) > limit:
        # Try to split at the last newline before the limit
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1 or split_at < limit // 2:
            split_at = limit
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def _model_short_label(model_used: str) -> str:
    """Turn a full model name into a short Telegram-friendly label."""
    m = model_used.lower()
    if not m:
        return "🔄 **Streaming...**"
    if "deepseek" in m and "flash" in m:
        if "cached" in m:
            return "⚡ **DeepSeek Flash** (cached)"
        return "⚡ **DeepSeek Flash**"
    if "deepseek" in m and "v4-pro" in m:
        return "🧠 **DeepSeek V4**"
    if "cached" in m:
        if "qwen" in m:
            return "⚡ **Qwen 3.5** (cached)"
        if "deepseek" in m:
            return "⚡ **DeepSeek** (cached)"
        return "⚡ **Cached**"
    if "fallback" in m:
        if "qwen" in m:
            return "⚡ **Qwen 3.5** (fallback)"
        return "⚡ **Fallback**"
    if "qwen" in m:
        return "⚡ **Qwen 3.5**"
    if "deepseek" in m:
        return "🧠 **DeepSeek**"
    return f"🤖 **{model_used.split('/')[-1][:30]}**"


def _md_to_telegram_html(text: str) -> str:
    """Convert common markdown to Telegram-compatible HTML for parse_mode='HTML'.

    Telegram HTML supports: <b>, <i>, <u>, <s>, <code>, <pre>, <a href>.
    Headings (#) become bold lines. Bullet lists work natively.
    Returns plain text (entities stripped) if HTML would be malformed.
    """
    import re

    # 0. Escape raw HTML entities first — otherwise &, <, > in source text
    #    would break Telegram's HTML parser
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 1. Code blocks (```) → <pre> (must be done before inline code)
    text = re.sub(r'```[^\n]*\n(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)

    # 2. Inline code (`) → <code>
    text = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', text)

    # 3. Bold (**text** or __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # 4. Italic (*text* or _text_)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)

    # 5. Headings (# at line start)
    text = re.sub(r'^#{1,4}\s+(.+)', r'<b>\1</b>', text, flags=re.MULTILINE)

    # 6. Links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # 7. Sanitize: strip any unmatched tags to prevent parse errors
    text = _sanitize_telegram_html(text)

    # 8. If sanitization removed everything useful, return plain text
    if not text.strip():
        # Start over with plain text (no markdown conversion)
        text2 = text  # already escaped above, just strip remaining markdown
        # Can't easily recover, just return original escaped text
        return "..."  # shouldn't happen

    return text


def _sanitize_telegram_html(text: str) -> str:
    """Remove unmatched HTML tags that would break Telegram's parser."""
    import re
    allowed = {"b", "i", "u", "s", "code", "pre", "a"}
    # Track tag stack
    stack = []
    result = []
    pos = 0
    for m in re.finditer(r'</?(\w+)(?:\s[^>]*)?>', text):
        tag = m.group(1)
        is_close = m.group(0).startswith("</")
        # Add text before this tag
        result.append(text[pos:m.start()])
        if tag in allowed:
            if is_close:
                if stack and stack[-1] == tag:
                    result.append(m.group(0))
                    stack.pop()
                # else: unmatched close — skip it
            else:
                result.append(m.group(0))
                stack.append(tag)
        # else: unknown tag — skip
        pos = m.end()
    result.append(text[pos:])
    # Close any remaining open tags
    for tag in reversed(stack):
        result.append(f"</{tag}>")
    return "".join(result)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — send welcome message."""
    user = update.effective_user
    if not _is_allowed(user.id):
        logger.warning(f"Unauthorized user {user.id} ({user.full_name}) tried /start")
        return

    await update.message.reply_text(
        f"👋 Hello {user.first_name}!\n\n"
        f"I'm **Kara** — your AI assistant powered by LowCostLLM.\n\n"
        f"• Smart caching: repeated questions use cheap model\n"
        f"• Fresh questions go to the full reasoning engine\n\n"
        f"Just send me a message and I'll respond!",
    )
    logger.info(f"User {user.id} ({user.full_name}) started the bot")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""
    user = update.effective_user
    if not _is_allowed(user.id):
        return

    await update.message.reply_text(
        "**Kara — LowCostLLM Bot**\n\n"
        "Send any question and I'll answer using the two-tier LLM pipeline:\n"
        "• Similar cached questions → cheap model (fast, low cost)\n"
        "• New questions → DeepSeek V4 Pro (thorough)\n\n"
        "Commands:\n"
        "/start — Welcome message\n"
        "/help — This help text\n"
        "/stats — Usage statistics",
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats — show cache and usage stats."""
    user = update.effective_user
    if not _is_allowed(user.id):
        return

    try:
        from db import get_cache_stats
        from stats import get_stats

        cache = get_cache_stats()
        usage = get_stats()

        msg = (
            f"📊 **LowCostLLM Stats**\n\n"
            f"**Cache**\n"
            f"• Entries: {cache['total_entries']} active / {cache.get('expired', 0)} expired\n"
            f"• TTL: {cache['ttl_days']} days\n\n"
            f"**Usage** (since {usage['uptime_started'][:10]})\n"
            f"• Requests: {usage['total_requests']}\n"
            f"• Cache hits: {usage['cache_hits']} ({usage['hit_rate_pct']}%)\n"
            f"• Expensive calls: {usage['expensive_calls']}\n"
            f"• IRRELEVANT escalations: {usage['irrelevant_escalations']} ({usage['irrelevant_rate_pct']}%)\n"
            f"• Tool calls: {usage.get('tool_calls_total', 0)}\n"
        )
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"Error fetching stats: {e}")


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new — clear conversation history and start fresh."""
    user = update.effective_user
    if not _is_allowed(user.id):
        return

    deleted = clear_user_history(user.id)
    await update.message.reply_text(
        f"🆕 Conversation reset.\n"
        f"{'Removed ' + str(deleted) + ' previous messages.' if deleted else 'No history to clear.'}\n\n"
        f"What would you like to talk about?"
    )
    logger.info(f"User {user.id} cleared {deleted} messages")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any text message — route through the LLM pipeline."""
    user = update.effective_user
    if not _is_allowed(user.id):
        logger.warning(f"Blocked message from unauthorized user {user.id}")
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return

    user_query = update.message.text.strip()
    if not user_query:
        return

    logger.info(f"Query from {user.id} ({user.full_name}): {user_query[:80]}...")

    # Keep typing indicator alive while LLM processes (Telegram expires after ~5s)
    chat_id = update.effective_chat.id
    typing_task = asyncio.create_task(_keep_typing(context.bot, chat_id))

    try:
        # Load conversation history for multi-turn context
        chat_history = await load_and_prepare_history(user.id)

        # Send placeholder that we'll stream into
        thinking_msg = await update.message.reply_text("🧠 Thinking...", parse_mode="HTML")
        accumulated = ""
        last_edit = asyncio.get_event_loop().time()
        EDIT_INTERVAL = 1.5  # seconds — avoid Telegram rate limits

        async def on_chunk(chunk_text: str):
            nonlocal accumulated, last_edit
            accumulated += chunk_text
            now = asyncio.get_event_loop().time()
            # Edit the message periodically to show live streaming
            if now - last_edit >= EDIT_INTERVAL or len(accumulated) < 100:
                last_edit = now
                try:
                    label = _model_short_label("")  # temp label during streaming
                    preview = accumulated[:3500]  # Telegram limit
                    await thinking_msg.edit_text(
                        _md_to_telegram_html(preview),
                        parse_mode="HTML",
                    )
                except Exception:
                    pass  # rate limit or parse error — skip this frame

        model_used, images = await process_query_stream(
            user_query, on_chunk, chat_history=chat_history
        )

        # Final edit with model label
        answer = accumulated
        if not answer.strip():
            await thinking_msg.edit_text("⚠️ No response from model. Please try again.")
        else:
            model_label = _model_short_label(model_used)
            full_text = f"{model_label}\n\n{answer}"
            # For long answers, send as new message; for short ones, edit
            if len(full_text) > 3500:
                await thinking_msg.delete()
                md_chunks = _split_long_message(full_text)
                for chunk_md in md_chunks:
                    chunk_html = _md_to_telegram_html(chunk_md)
                    if chunk_html.strip():
                        await update.message.reply_text(chunk_html, parse_mode="HTML")
            else:
                await thinking_msg.edit_text(
                    _md_to_telegram_html(full_text),
                    parse_mode="HTML",
                )

        # Save the exchange to conversation history
        await save_exchange(user.id, user_query, answer)

        # Wait for any async image generation to complete, then send
        images = await _wait_for_images()
        for img_path in images:
            try:
                with open(img_path, "rb") as f:
                    await update.message.reply_photo(
                        photo=f,
                        caption=f"🖼️ {user_query[:100]}",
                    )
            except Exception:
                logger.exception(f"Failed to send image {img_path}")
    except Exception as e:
        logger.exception(f"Error processing query from {user.id}")
        await update.message.reply_text(
            f"❌ Sorry, something went wrong: {type(e).__name__}\n\n"
            f"Details: {str(e)[:200]}\n\n"
            f"Please try again in a moment."
        )
    finally:
        typing_task.cancel()


async def _keep_typing(bot, chat_id: int, interval: float = 4.5) -> None:
    """Refresh typing indicator every `interval` seconds until cancelled.

    Telegram's typing status expires after ~5 seconds. This loop keeps it
    alive for the duration of LLM processing.
    """
    from telegram.constants import ChatAction
    try:
        while True:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


def build_application() -> Application:
    """Build and return the Telegram bot Application (not started yet)."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — bot will not start")
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("new", new_command))
    app.add_handler(CommandHandler("clear", new_command))  # alias
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app


async def start_bot() -> Application:
    """Start the bot in webhook mode. Returns the Application for shutdown."""
    from config import TELEGRAM_WEBHOOK_URL

    app = build_application()

    await app.initialize()
    await app.start()

    # Set webhook — Telegram will POST updates to this URL
    webhook_url = TELEGRAM_WEBHOOK_URL or f"https://llm.smartdochub.net/telegram/webhook"
    await app.bot.set_webhook(url=webhook_url, allowed_updates=["message"])
    logger.info(f"🤖 Telegram bot started (webhook → {webhook_url})")

    return app


async def stop_bot(app: Application) -> None:
    """Gracefully stop the bot."""
    await app.bot.delete_webhook()
    await app.stop()
    await app.shutdown()
    logger.info("🤖 Telegram bot stopped")


async def process_telegram_update(body: dict) -> None:
    """Process an incoming Telegram update from the webhook endpoint.

    Called by the FastAPI /telegram/webhook route. Gets the bot application
    from app.state and feeds the update into the handler pipeline.
    """
    import json
    from telegram import Update

    # Access the FastAPI app state to get the bot application
    # We store it on the bot module itself during startup
    update = Update.de_json(body, _tg_bot_app.bot)
    await _tg_bot_app.process_update(update)


# Module-level reference to the bot Application, set during startup
_tg_bot_app: Application | None = None
