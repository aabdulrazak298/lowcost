"""FastAPI application for LowCostLLM — cached two-tier LLM proxy."""
import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from db import init_db
from proxy import handle_chat_completion, stream_chat_completion
from code_proxy import handle_code_completion, stream_code_completion
from webhook import handle_webhook_chat
from config import (
    AUTH_KEY, API_KEY, CHEAP_MODEL,
    RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW,
    get_cheap_model, get_expensive_model,
)
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
_inflight_updates: set[int] = set()

# Rate limiting — per-IP sliding window
_rate_windows: dict[str, deque[float]] = defaultdict(lambda: deque())
_rate_lock = asyncio.Lock()

# Request dedup — fingerprint → expiry
_dedup: dict[str, float] = {}
_dedup_lock = asyncio.Lock()
_DEDUP_TTL = 5

# 404 debug log
_404_LOG = Path(__file__).parent / "404_debug.log"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + stats + Telegram bot. Shutdown: stop bot + flush."""
    init_db()

    from stats import init_from_db, flush_to_db
    init_from_db()

    # Restore model overrides from previous session
    from config import _load_overrides_from_db
    _load_overrides_from_db()

    # Restore voice settings from previous session
    from config import _load_voice_from_db
    _load_voice_from_db()

    flush_task = asyncio.create_task(_periodic_flush())

    # ── Telegram bot ──────────────────────────────────────────
    bot_app = None
    try:
        from telegram_bot import start_bot
        import telegram_bot as tb
        bot_app = await start_bot()
        tb._tg_bot_app = bot_app
        app.state.tg_bot = bot_app
        logger.info("Telegram polling started")
    except RuntimeError:
        logger.warning("Telegram bot not started — no token configured")
    except Exception:
        logger.exception("Telegram bot failed to start")

    yield

    # Shutdown
    flush_task.cancel()
    try:
        await flush_task
    except asyncio.CancelledError:
        pass

    flush_to_db()

    if bot_app:
        from telegram_bot import stop_bot
        await stop_bot(bot_app)


async def _periodic_flush(interval: int = 30):
    while True:
        await asyncio.sleep(interval)
        try:
            from stats import flush_to_db
            flush_to_db()
        except Exception:
            logger.exception("Periodic stats flush failed")


app = FastAPI(title="LowCostLLM", version="0.5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=512)


# ── Auth dependency ─────────────────────────────────────────────


async def _auth_dependency(request: Request):
    """Accepts Bearer token (AUTH_KEY or API_KEY) or Basic auth (password=API_KEY).

    Skips auth for localhost requests (FlaskChat admin dashboard).
    """
    # Allow localhost without auth
    client = request.client.host if request.client else ""
    if client in ("127.0.0.1", "::1", "localhost"):
        return

    key = AUTH_KEY or API_KEY
    if not key:
        return  # Auth disabled

    auth = request.headers.get("Authorization", "")

    # Bearer token
    if auth.startswith("Bearer "):
        token = auth[7:]
        if token in (AUTH_KEY, API_KEY):
            return

    # HTTP Basic
    if auth.startswith("Basic "):
        import base64
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            if ":" in decoded:
                _, pwd = decoded.split(":", 1)
                if pwd in (AUTH_KEY, API_KEY):
                    return
        except Exception:
            pass

    raise HTTPException(status_code=401, detail="Invalid API key")


# ── Middleware ──────────────────────────────────────────────────


@app.middleware("http")
async def rate_limiter(request: Request, call_next):
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

    # Log 404s to file for debugging
    if response.status_code >= 400:
        try:
            with open(_404_LOG, "a") as f:
                f.write(f"{time.time()} HTTP {response.status_code} {request.method} {request.url.path}\n")
        except Exception:
            pass

    return response


# ── Dedup helper ────────────────────────────────────────────────


async def _check_dedup(body: dict) -> bool:
    fingerprint = hashlib.sha256(
        json.dumps(body.get("messages", []), sort_keys=True).encode()
    ).hexdigest()
    now = time.monotonic()
    async with _dedup_lock:
        if fingerprint in _dedup and _dedup[fingerprint] > now:
            return True
        _dedup[fingerprint] = now + _DEDUP_TTL
    async with _dedup_lock:
        expired = [k for k, v in _dedup.items() if v <= now]
        for k in expired:
            del _dedup[k]
    return False


def _header_safe(value: str) -> str:
    """HTTP header values must be latin-1 encodable; replace any non-latin-1
    characters (e.g. em dashes) so routing headers never 500."""
    return value.encode("latin-1", "replace").decode("latin-1")


# ── Health ──────────────────────────────────────────────────────


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


@app.get("/health/readiness")
@app.get("/health/ready")
async def health_readiness():
    """LiteLLM-compatible readiness check — OpenClient calls this."""
    return {"status": "ok", "litellm_version": "0.0.0"}


# ── Models ──────────────────────────────────────────────────────


@app.get("/v1/models")
@app.get("/models")
async def list_models(_auth=Depends(_auth_dependency)):
    return {
        "object": "list",
        "data": [
            {"id": "lowcostllm", "object": "model", "owned_by": "lowcostllm"},
            {"id": "thinkllm", "object": "model", "owned_by": "thinkllm"},
        ],
    }


@app.get("/v1/code/models")
@app.get("/code/models")
async def list_code_models(_auth=Depends(_auth_dependency)):
    """OpenCode model-picker discovery alias — same shape as /v1/models."""
    return {
        "object": "list",
        "data": [
            {"id": "lowcostllm-code", "object": "model", "owned_by": "lowcostllm-code"},
        ],
    }


@app.get("/model/info")
@app.get("/v1/model/info")
async def model_info(_auth=Depends(_auth_dependency), model: str = ""):
    """LiteLLM-compatible model info — OpenClient calls this."""
    model_name = model or "lowcostllm"
    return {
        "data": [{
            "model_name": model_name,
            "model_info": {
                "id": model_name,
                "key": model_name,
                "max_tokens": 8192,
                "max_input_tokens": 8192,
                "max_output_tokens": 8192,
                "mode": "chat",
                "supports_vision": model_name == "qwen35b",
                "supports_function_calling": True,
                "supports_parallel_function_calling": False,
                "litellm_provider": "openai",
            },
        }],
    }


# ── OpenClient stubs ────────────────────────────────────────────


@app.post("/api/show")
@app.post("/v1/api/show")
async def api_show():
    """Stub — OpenClient calls this. No auth needed."""
    return {"status": "ok"}


@app.get("/v1/mcp/server")
@app.get("/mcp/server")
async def mcp_server():
    """Stub — OpenClient MCP tool discovery."""
    return {"data": []}


# ── Chat completions ────────────────────────────────────────────


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request, _auth=Depends(_auth_dependency)):
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
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        if await _check_dedup(body_dict):
            return JSONResponse(
                {"error": {"message": "Duplicate request", "type": "dedup"}},
                status_code=409,
            )

        result = await handle_chat_completion(body_dict)
        return JSONResponse(result)

    except Exception as e:
        logger.exception("Chat completion failed")
        if body_dict.get("stream"):
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


# ── Code completions (judge-based cache routing) ────────────────


@app.post("/v1/code/chat/completions")
@app.post("/code/chat/completions")
async def code_completions(request: Request, _auth=Depends(_auth_dependency)):
    try:
        body = await request.json()
        req = ChatCompletionRequest.model_validate(body)
        body_dict = req.model_dump(exclude_none=True)
        body_dict["x_session_id"] = request.headers.get("x-session-id")
    except ValidationError as e:
        return JSONResponse(
            {"error": {"message": str(e.errors()[0]["msg"]), "type": "invalid_request"}},
            status_code=422,
        )

    try:
        if body_dict.get("stream"):
            return StreamingResponse(
                stream_code_completion(body_dict),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        if await _check_dedup(body_dict):
            return JSONResponse(
                {"error": {"message": "Duplicate request", "type": "dedup"}},
                status_code=409,
            )

        result = await handle_code_completion(body_dict)
        routing_meta = result.pop("_routing_meta", {}) or {}
        headers = {}
        if routing_meta.get("selected_model"):
            headers["x-model-router-selected-model"] = _header_safe(str(routing_meta["selected_model"]))
        if routing_meta.get("rationale"):
            headers["x-model-router-rationale"] = _header_safe(str(routing_meta["rationale"]))
        return JSONResponse(result, headers=headers)

    except Exception as e:
        logger.exception("Code completion failed")
        if body_dict.get("stream"):
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


# ── Webhook endpoints ───────────────────────────────────────────


@app.post("/webhook/chat")
async def webhook_chat(request: Request):
    """Flask Chat compatible webhook — n8n-style streaming."""
    return await handle_webhook_chat(request)


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """Telegram webhook endpoint (alternative to polling)."""
    import telegram_bot as tb
    if tb._tg_bot_app is None:
        return JSONResponse({"error": "bot not started"}, status_code=503)

    body = await request.json()
    update_id = body.get("update_id", 0)

    if update_id in _inflight_updates:
        return {"ok": True}

    _inflight_updates.add(update_id)
    asyncio.create_task(_process_update_async(update_id, body))
    return {"ok": True}


async def _process_update_async(update_id: int, body: dict) -> None:
    import telegram_bot as tb
    try:
        await tb.process_telegram_update(body)
    except Exception:
        logger.exception(f"Error processing Telegram update {update_id}")
    finally:
        _inflight_updates.discard(update_id)


# ── Admin ───────────────────────────────────────────────────────


@app.get("/admin")
async def admin_dashboard(request: Request, _auth=Depends(_auth_dependency)):
    from db import get_cache_stats
    from stats import get_stats

    return {
        "cache": get_cache_stats(),
        "usage": get_stats(),
    }


# ── Catch-all (debug 404s) ──────────────────────────────────────


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(request: Request, path: str):
    logger.warning(f"404 NOT FOUND: {request.method} /{path}")
    return JSONResponse(
        {"detail": f"Not found: {request.method} /{path}"},
        status_code=404,
    )
