"""Telegram bot integration for LowCostLLM — polling mode.

Uses the shared processor.process_query() pipeline so Telegram messages
go through the same cache → cheap → expensive flow as Flask Chat.
"""

import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from processor import process_query
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS, AVAILABLE_MODELS
from config import get_cheap_model, set_cheap_model, get_expensive_model, set_expensive_model

logger = logging.getLogger("lowcostllm.tg")

_tg_bot_app: Application | None = None

# ── Command handlers ────────────────────────────────────────────


async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Hello! I'm LowCostLLM — ask me anything!\n\n"
        "Commands:\n"
        "/model — show/switch models\n"
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
        f"/model <name> — switch cheap model\n"
        f"/model list — show all available\n"
        f"/new — clear chat context",
    )


async def _model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    if not args:
        cheap = get_cheap_model()
        expensive = get_expensive_model()
        await update.message.reply_text(
            f"Current models:\n• Cheap: `{cheap}`\n• Expensive: `{expensive}`\n\n"
            f"Use /model list to see options.\n"
            f"/model cheap <name> — swap cheap\n"
            f"/model expensive <name> — swap expensive"
        )
        return

    sub = args[0].lower()

    if sub == "list":
        lines = ["**Available models:**"]
        for key, (full_id, provider) in AVAILABLE_MODELS.items():
            lines.append(f"• `{key}` → {full_id} ({provider})")
        await update.message.reply_text("\n".join(lines))
        return

    if sub in ("cheap", "expensive"):
        if len(args) < 2:
            current = get_cheap_model() if sub == "cheap" else get_expensive_model()
            await update.message.reply_text(
                f"Current {sub} model: `{current}`\n"
                f"Usage: /model {sub} <name>\n"
                f"Use /model list to see options."
            )
            return

        key = args[1].lower()
        if key not in AVAILABLE_MODELS:
            await update.message.reply_text(
                f"❌ Unknown model: `{key}`\nUse /model list to see options."
            )
            return

        full_id, provider = AVAILABLE_MODELS[key]
        if sub == "cheap":
            set_cheap_model(full_id)
        else:
            set_expensive_model(full_id)
        await update.message.reply_text(
            f"✅ {sub.capitalize()} model set to `{full_id}` ({provider})"
        )
        return

    # Fallback: bare key → set cheap model
    if sub in AVAILABLE_MODELS:
        full_id, provider = AVAILABLE_MODELS[sub]
        set_cheap_model(full_id)
        await update.message.reply_text(
            f"✅ Cheap model set to `{full_id}` ({provider})"
        )
    else:
        await update.message.reply_text(
            f"❌ Unknown: `{sub}`\nUse /model list to see options."
        )


async def _new_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Clear any stored chat history in user_data
    context.user_data.clear()
    cheap = get_cheap_model()
    expensive = get_expensive_model()
    await update.message.reply_text(
        f"🆕 Session cleared!\n"
        f"Cheap: `{cheap}`\n"
        f"Expensive: `{expensive}`"
    )


# ── Message handler ─────────────────────────────────────────────


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message.text
    chat_id = update.effective_chat.id
    user_id = str(update.effective_user.id)

    if not user_msg or not user_msg.strip():
        return

    # Check allowed users
    if TELEGRAM_ALLOWED_USERS:
        allowed = [u.strip() for u in TELEGRAM_ALLOWED_USERS.split(",") if u.strip()]
        if allowed and user_id not in allowed and str(chat_id) not in allowed:
            await update.message.reply_text("⛔ Access denied.")
            return

    # Send typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Build chat history from user_data (last N exchanges)
    history = context.user_data.get("history", [])
    chat_history_str = "\n".join(
        f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
        for h in history[-10:]
    )

    # Process through the shared pipeline
    try:
        response, model_used = await process_query(
            user_query=user_msg,
            chat_history=chat_history_str,
        )
    except Exception as e:
        logger.exception(f"Query processing failed: {e}")
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

    # Send response (handle long messages)
    if len(response) > 4000:
        # Split into chunks
        for i in range(0, len(response), 4000):
            chunk = response[i:i + 4000]
            await update.message.reply_text(chunk)
    else:
        await update.message.reply_text(response)


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

    # Register handlers
    app.add_handler(CommandHandler("start", _start))
    app.add_handler(CommandHandler("help", _help_cmd))
    app.add_handler(CommandHandler("model", _model_cmd))
    app.add_handler(CommandHandler("new", _new_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    app.add_error_handler(_error_handler)

    # Initialize and start polling
    await app.initialize()
    await app.updater.start_polling(drop_pending_updates=True)
    await app.start()

    _tg_bot_app = app
    logger.info("Telegram bot started (polling mode)")
    return app


async def stop_bot(app: Application | None = None) -> None:
    """Stop the Telegram bot."""
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
    """Process a Telegram webhook update (alternative to polling)."""
    global _tg_bot_app
    if _tg_bot_app is None:
        raise RuntimeError("Bot not started")
    update = Update.de_json(body, _tg_bot_app.bot)
    await _tg_bot_app.process_update(update)
