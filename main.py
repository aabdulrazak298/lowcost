"""FastAPI application for LowCostLLM."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from db import init_db
from proxy import handle_chat_completion, stream_chat_completion
from webhook import handle_webhook_chat
from config import AUTH_KEY

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
    """Startup: init DB + restore stats + periodic flush. Shutdown: save stats."""
    init_db()

    from stats import init_from_db, flush_to_db
    init_from_db()

    flush_task = asyncio.create_task(_periodic_flush())

    yield

    flush_task.cancel()
    try:
        await flush_task
    except asyncio.CancelledError:
        pass

    flush_to_db()


async def _periodic_flush(interval: int = 30):
    """Flush in-memory stats to DB every `interval` seconds."""
    while True:
        await asyncio.sleep(interval)
        try:
            from stats import flush_to_db
            flush_to_db()
        except Exception:
            logger.exception("Periodic stats flush failed")


app = FastAPI(title="LowCostLLM", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def auth_dependency(request: Request):
    """Optional auth — only enforced when AUTH_KEY is set in .env."""
    if AUTH_KEY:
        if request.headers.get("Authorization") != f"Bearer {AUTH_KEY}":
            raise HTTPException(status_code=401, detail="Unauthorized")


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
async def chat_completions(request: Request, auth=Depends(auth_dependency)):
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
async def webhook_chat(request: Request, auth=Depends(auth_dependency)):
    """Flask Chat compatible webhook — n8n-style streaming."""
    return await handle_webhook_chat(request)


@app.get("/admin")
async def admin_dashboard(auth=Depends(auth_dependency)):
    """Usage dashboard — cache stats + request metrics."""
    from db import get_cache_stats
    from stats import get_stats

    cache = get_cache_stats()
    usage = get_stats()

    return {
        "cache": cache,
        "usage": usage,
    }
