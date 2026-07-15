"""FastAPI application for LowCostLLM."""
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from db import init_db
from proxy import handle_chat_completion, stream_chat_completion
from webhook import handle_webhook_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log"),
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + restore stats. Shutdown: save stats."""
    init_db()

    from stats import init_from_db
    init_from_db()

    yield

    from stats import flush_to_db
    flush_to_db()


app = FastAPI(title="LowCostLLM", version="0.3.0", lifespan=lifespan)


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
        body = await request.json()
        if body.get("stream"):
            return StreamingResponse(
                stream_chat_completion(body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        result = await handle_chat_completion(body)
        return JSONResponse(result)
    except Exception as e:
        body_preview = {}
        try:
            body_preview = await request.json()
        except Exception:
            pass
        if body_preview.get("stream"):
            error_chunk = json.dumps(
                {"error": {"message": str(e), "type": "server_error"}}
            )
            return StreamingResponse(
                iter([f"data: {error_chunk}\n\n", "data: [DONE]\n\n"]),
                media_type="text/event-stream",
            )
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=500,
        )


@app.post("/webhook/chat")
async def webhook_chat(request: Request):
    """Flask Chat compatible webhook — n8n-style streaming."""
    return await handle_webhook_chat(request)


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
