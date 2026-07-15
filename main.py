"""FastAPI application for LowCostLLM."""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from db import init_db
from proxy import handle_chat_completion
from webhook import handle_webhook_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Track in-flight update IDs to prevent duplicate processing from Telegram retries
_inflight_updates: set[int] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + restore stats + launch Telegram bot. Shutdown: save stats + stop bot."""
    init_db()

    # Restore persisted stats from previous session
    from stats import init_from_db
    init_from_db()

    # Start Telegram bot in webhook mode (if token is configured)
    bot_app = None
    try:
        from telegram_bot import start_bot
        import telegram_bot as tb
        bot_app = await start_bot()
        tb._tg_bot_app = bot_app  # store for webhook endpoint
        app.state.tg_bot = bot_app
    except RuntimeError:
        logger.warning("Telegram bot not started — no token configured")
    except Exception:
        logger.exception("Telegram bot failed to start")

    yield

    # Save stats before shutdown
    from stats import flush_to_db
    flush_to_db()

    # Shutdown Telegram bot
    if bot_app:
        from telegram_bot import stop_bot
        await stop_bot(bot_app)


app = FastAPI(title="LowCostLLM", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "lowcostllm", "object": "model"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        result = await handle_chat_completion(request)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=500,
        )


@app.post("/webhook/chat")
async def webhook_chat(request: Request):
    """Flask Chat compatible webhook — n8n-style streaming."""
    return await handle_webhook_chat(request)


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """Receive Telegram updates via webhook.

    Returns 200 OK IMMEDIATELY (before processing) so Telegram doesn't time out
    and retry. Processes the update in the background with deduplication.
    """
    import telegram_bot as tb
    if tb._tg_bot_app is None:
        return JSONResponse({"error": "bot not started"}, status_code=503)

    body = await request.json()
    update_id = body.get("update_id", 0)

    # Deduplicate: if this update is already being processed, skip it
    if update_id in _inflight_updates:
        logger.info(f"Telegram webhook: update {update_id} already in flight, skipping")
        return {"ok": True}

    _inflight_updates.add(update_id)

    # Fire-and-forget: process in background, acknowledge immediately
    asyncio.create_task(_process_update_async(update_id, body))

    return {"ok": True}


async def _process_update_async(update_id: int, body: dict) -> None:
    """Process a Telegram update in the background, then clean up."""
    import telegram_bot as tb
    try:
        await tb.process_telegram_update(body)
    except Exception:
        logger.exception(f"Error processing Telegram update {update_id}")
    finally:
        _inflight_updates.discard(update_id)


@app.get("/admin")
async def admin_dashboard():
    """Usage dashboard — cache stats + request metrics."""
    from db import get_cache_stats
    from stats import get_stats

    cache = get_cache_stats()
    usage = get_stats()

    return {
        "cache": cache,
        "usage": usage,
    }
