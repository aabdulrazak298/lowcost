"""FastAPI application for LowCostLLM."""
import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from db import init_db
from proxy import handle_chat_completion, stream_chat_completion
from webhook import handle_webhook_chat
from config import AUTH_KEY, CHEAP_MODEL, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
from schemas import ChatCompletionRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log"),
    ],
)
logger = logging.getLogger(__name__)

_start_time = time.monotonic()

# Rate limiting — per-IP sliding window
_rate_windows: dict[str, deque[float]] = defaultdict(lambda: deque())
_rate_lock = asyncio.Lock()

# Request dedup — fingerprint → expiry
_dedup: dict[str, float] = {}
_dedup_lock = asyncio.Lock()
_DEDUP_TTL = 5  # seconds


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


app = FastAPI(title="LowCostLLM", version="0.4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=512)


@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    """Sliding-window rate limiter per client IP."""
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = RATE_LIMIT_WINDOW
    limit = RATE_LIMIT_REQUESTS

    async with _rate_lock:
        dq = _rate_windows[client]
        while dq and dq[0] < now - window:
            dq.popleft()
        if len(dq) >= limit:
            return JSONResponse(
                {"error": {"message": "Rate limit exceeded", "type": "rate_limit"}},
                status_code=429,
            )
        dq.append(now)

    return await call_next(request)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    logger.info(json.dumps({
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "ms": elapsed_ms,
    }))
    return response


async def auth_dependency(request: Request):
    """Optional auth — only enforced when AUTH_KEY is set in .env."""
    if AUTH_KEY:
        if request.headers.get("Authorization") != f"Bearer {AUTH_KEY}":
            raise HTTPException(status_code=401, detail="Unauthorized")


async def _check_dedup(body: dict) -> bool:
    """Return True if this request is a duplicate within TTL window."""
    fingerprint = hashlib.sha256(
        json.dumps(body.get("messages", []), sort_keys=True).encode()
    ).hexdigest()
    now = time.monotonic()
    async with _dedup_lock:
        if fingerprint in _dedup and _dedup[fingerprint] > now:
            return True
        _dedup[fingerprint] = now + _DEDUP_TTL
    # Prune expired entries
    async with _dedup_lock:
        expired = [k for k, v in _dedup.items() if v <= now]
        for k in expired:
            del _dedup[k]
    return False


@app.get("/health")
async def health():
    try:
        from db import get_conn
        conn = get_conn()
        conn.execute("SELECT 1")
    except Exception:
        return JSONResponse(
            {"status": "unhealthy", "db": "unreachable"},
            status_code=503,
        )
    return {
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - _start_time),
    }


@app.get("/v1/models")
async def list_models(auth=Depends(auth_dependency)):
    return {
        "object": "list",
        "data": [
            {"id": CHEAP_MODEL, "object": "model"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, auth=Depends(auth_dependency)):
    try:
        body = await request.json()
        req = ChatCompletionRequest.model_validate(body)
        body_dict = req.model_dump(exclude_none=True)
    except ValidationError as e:
        return JSONResponse(
            {"error": {"message": str(e.errors()[0]["msg"]), "type": "invalid_request"}},
            status_code=422,
        )
    try:
        if body_dict.get("stream"):
            return StreamingResponse(
                stream_chat_completion(body_dict),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        if await _check_dedup(body_dict):
            return JSONResponse(
                {"error": {"message": "Duplicate request", "type": "dedup"}},
                status_code=409,
            )
        result = await handle_chat_completion(body_dict)
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
