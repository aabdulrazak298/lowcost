"""Central configuration — all tunables live here."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DB_PATH = ROOT / "cache.db"

SIMILARITY_THRESHOLD = int(os.getenv("SIMILARITY_THRESHOLD", "48"))
CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "365"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "2000000"))  # ~4 GB

# Cheap model (configurable — currently Qwen 3.7 Flash via OpenRouter)
CHEAP_API_KEY = os.getenv("CHEAP_API_KEY", "")
CHEAP_BASE_URL = os.getenv("CHEAP_BASE_URL", "https://openrouter.ai/api/v1")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "qwen/qwen3.7-flash")

# Cheap fallback model — used when the primary cheap model keeps failing
# (rate limit, 5xx, timeout) after CHEAP_FALLBACK_RETRIES attempts. Defaults to
# a different vendor (DeepSeek via OpenRouter) so a Qwen rate limit doesn't
# take down the fallback too. Configurable via /model -cb; set to "" to
# disable. Retry semantics: the OpenAI client also retries 429/5xx once per
# attempt, so CHEAP_FALLBACK_RETRIES=2 ≈ up to 4 HTTP attempts before fallback.
CHEAP_FALLBACK_MODEL = os.getenv("CHEAP_FALLBACK_MODEL", "deepseek/deepseek-v4-flash")
CHEAP_FALLBACK_RETRIES = int(os.getenv("CHEAP_FALLBACK_RETRIES", "2"))

# Expensive model — DeepSeek V4 Pro direct API (native tool calling)
EXPENSIVE_API_KEY = os.getenv("EXPENSIVE_API_KEY", "")
EXPENSIVE_BASE_URL = os.getenv("EXPENSIVE_BASE_URL", "https://api.deepseek.com/v1")
EXPENSIVE_MODEL = os.getenv("EXPENSIVE_MODEL", "deepseek-v4-pro")

# Fallback model — used if expensive model API fails (via OpenRouter for reliability)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "deepseek/deepseek-v4-flash")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")
FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "https://openrouter.ai/api/v1")

# Engy — decentralized verified inference (Bittensor SN53). OpenAI-compatible.
ENGY_API_KEY = os.getenv("ENGY_API_KEY", "")
ENGY_BASE_URL = os.getenv("ENGY_BASE_URL", "https://api.engy.ai/v1")
ENGY_MODEL = os.getenv("ENGY_MODEL", "deepseek-v4-flash-0731")

# Model IDs served by Engy (routed to the Engy client, not DeepSeek/OpenRouter).
# Engy's native IDs carry no org prefix, so they never collide with the
# OpenRouter-style "org/model" IDs in AVAILABLE_MODELS.
ENGY_MODELS = {
    "deepseek-v4-flash-0731",
    "qwen3.6-35b-a3b",
    "qwen3.8-27b",
    "glm-5.2",
    "kimi-k3",
}

# OpenRouter equivalent used when an Engy model fails (OpenRouter has no 0731 build)
ENGY_FALLBACK_MODEL = "deepseek/deepseek-v4-flash"

# ── Runtime model overrides (set by /model command, persisted to DB) ──
_cheap_override: str | None = None
_expensive_override: str | None = None
# Cheap fallback: None = not configured (use env default), "" = explicitly
# disabled via /model -cb off, otherwise the model id.
_cheap_fallback_override: str | None = None


def _load_overrides_from_db() -> None:
    """Restore model overrides from the database after restart."""
    global _cheap_override, _expensive_override, _cheap_fallback_override
    try:
        from db import ensure_overrides_schema, get_conn
        ensure_overrides_schema()
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT cheap_override, expensive_override, cheap_fallback_override "
                "FROM model_overrides WHERE id = 1"
            ).fetchone()
            if row:
                _cheap_override = row["cheap_override"]
                _expensive_override = row["expensive_override"]
                _cheap_fallback_override = row["cheap_fallback_override"]
        except Exception:
            # Older DB without the fallback column — load the two known columns.
            row = conn.execute(
                "SELECT cheap_override, expensive_override FROM model_overrides WHERE id = 1"
            ).fetchone()
            if row:
                _cheap_override = row["cheap_override"]
                _expensive_override = row["expensive_override"]
    except Exception:
        pass


def get_cheap_model() -> str:
    return _cheap_override or CHEAP_MODEL


def get_expensive_model() -> str:
    return _expensive_override or EXPENSIVE_MODEL


def get_cheap_fallback_model() -> str | None:
    """Return the cheap fallback model id, or None/'' when disabled."""
    return _cheap_fallback_override if _cheap_fallback_override is not None else CHEAP_FALLBACK_MODEL


def set_cheap_model(model_id: str | None) -> None:
    global _cheap_override
    _cheap_override = model_id
    _save_overrides_to_db()


def set_expensive_model(model_id: str | None) -> None:
    global _expensive_override
    _expensive_override = model_id
    _save_overrides_to_db()


def set_cheap_fallback_model(model_id: str | None) -> None:
    """Set the cheap fallback model. None or '' disables fallback entirely."""
    global _cheap_fallback_override
    _cheap_fallback_override = (model_id or "").strip() or ""
    _save_overrides_to_db()


def _save_overrides_to_db() -> None:
    """Persist overrides so they survive restarts."""
    try:
        from db import ensure_overrides_schema, get_conn
        ensure_overrides_schema()
        conn = get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO model_overrides "
            "(id, cheap_override, expensive_override, cheap_fallback_override) "
            "VALUES (1, ?, ?, ?)",
            (_cheap_override, _expensive_override, _cheap_fallback_override),
        )
        conn.commit()
    except Exception:
        pass


# ── Available models registry ──────────────────────────────────────
AVAILABLE_MODELS = {
    "qwen":       ("qwen/qwen3.7-flash", "OpenRouter"),
    "qwen36":     ("qwen/qwen3.6-flash", "OpenRouter"),
    "qwen35f":    ("qwen/qwen3.5-flash-02-23", "OpenRouter"),
    "qwen35b":    ("qwen/qwen3.6-35b-a3b", "OpenRouter"),
    "qwen-plus":  ("qwen/qwen3.7-plus", "OpenRouter"),
    "flash":      ("deepseek/deepseek-v4-flash", "OpenRouter"),
    "flash-ds":   ("deepseek-v4-flash", "DeepSeek Direct"),
    "pro":        ("deepseek-v4-pro", "DeepSeek Direct"),
    "m3":         ("minimax/minimax-m3", "OpenRouter"),
    "gemini":     ("google/gemini-2.5-flash", "OpenRouter"),
    "gemini-lite":("google/gemini-3.1-flash-lite", "OpenRouter"),
    "llama":      ("meta-llama/llama-4-maverick", "OpenRouter"),
    "luna":       ("openai/gpt-5.6-luna", "OpenRouter"),
    "ling":       ("inclusionai/ling-3.0-flash", "OpenRouter"),
    "sonnet":     ("anthropic/claude-sonnet-4", "OpenRouter"),
    "nemotron":   ("nvidia/nemotron-3-ultra-550b-a55b", "OpenRouter"),
    "solar":      ("upstage/solar-pro4", "OpenRouter"),
    "flash-engy": ("deepseek-v4-flash-0731", "Engy"),
    "qwen3.8-27b-engy": ("qwen3.8-27b", "Engy"),
}

# Output pricing per 1M tokens (keyed by model key)
MODEL_PRICING: dict[str, float] = {
    "qwen": 0.13, "qwen36": 0.26, "qwen35f": 0.26, "qwen35b": 0.80,
    "qwen-plus": 1.28, "flash": 0.28, "flash-ds": 0.28, "pro": 0.87,
    "m3": 2.40, "gemini": 2.50, "gemini-lite": 1.50,
    "llama": 0.85, "luna": 6.00, "ling": 0.021, "sonnet": 15.00, "nemotron": 2.20, "solar": 0.12,
    "flash-engy": 0.09,
    "qwen3.8-27b-engy": 0.32,
}


def _resolve_key(model_id: str) -> str | None:
    """Map a model id back to its registry key via LONGEST full_id match.

    Longest-match (not first-match) so a shorter id that is a substring of a
    longer one (e.g. "deepseek-v4-flash" is a substring of
    "deepseek-v4-flash-0731") can't steal the match from the more specific
    model.
    """
    best_key, best_len = None, -1
    mid = (model_id or "").lower()
    for key, (full_id, _) in AVAILABLE_MODELS.items():
        fid = (full_id or "").lower()
        if fid and fid in mid and len(fid) > best_len:
            best_key, best_len = key, len(fid)
    return best_key


def estimate_cost(model_id: str, output_chars: int) -> float:
    """Estimate cost based on model output pricing and char count (3.5 chars ≈ 1 token)."""
    key = _resolve_key(model_id)
    if key:
        price = MODEL_PRICING.get(key, 0.28)
        tokens = output_chars / 3.5
        return (tokens / 1_000_000) * price
    return 0.0


def build_calling_card(model_used: str, usage: dict | None = None) -> str:
    """Build the '🤖 <short-key> · $<cost>' footer shown after an answer.

    Single source of truth for the Telegram + web calling card, so both
    platforms show the identical model label and real token cost.
    """
    usage = usage or {}

    # Shorten model id to its registry key ("qwen/qwen3.7-flash" -> "qwen").
    key = _resolve_key(model_used)
    if key:
        short_name = f"{key} (cached)" if "(cached)" in model_used else key
    else:
        short_name = model_used

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    if prompt_tokens or completion_tokens:
        price_per_m = MODEL_PRICING.get(key, 0.28) if key else 0.28
        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1_000_000) * price_per_m
        cost_str = f" · ${cost:.6f} · {total_tokens} tok"
    else:
        cost_str = ""
    return f"\n\n---\n🤖 {short_name}{cost_str}"


# Telegram bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USERS = os.getenv("TELEGRAM_ALLOWED_USERS", "")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "")

# ── Voice / TTS settings (set by /voice command, persisted to DB) ──
VOICE_ENABLED_DEFAULT = os.getenv("VOICE_ENABLED", "0") == "1"
VOICE_ENGINE_DEFAULT = os.getenv("VOICE_ENGINE", "edge")  # "edge" or "kokoro"
VOICE_NAME = os.getenv("VOICE_NAME", "en-US-JennyNeural")   # edge-tts voice (female)
VOICE_RATE = os.getenv("VOICE_RATE", "+0%")                 # edge-tts speed
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")        # kokoro voice id (female)

_voice_enabled_override: bool | None = None
_voice_engine_override: str | None = None


def _load_voice_from_db() -> None:
    """Restore voice settings from the database after restart."""
    global _voice_enabled_override, _voice_engine_override
    try:
        from db import get_conn
        row = get_conn().execute(
            "SELECT voice_enabled, voice_engine FROM app_settings WHERE id = 1"
        ).fetchone()
        if row:
            _voice_enabled_override = bool(row["voice_enabled"])
            _voice_engine_override = row["voice_engine"]
    except Exception:
        pass


def get_voice_enabled() -> bool:
    if _voice_enabled_override is not None:
        return _voice_enabled_override
    return VOICE_ENABLED_DEFAULT


def get_voice_engine() -> str:
    return _voice_engine_override or VOICE_ENGINE_DEFAULT


def set_voice_enabled(enabled: bool) -> None:
    global _voice_enabled_override
    _voice_enabled_override = enabled
    _save_voice_to_db()


def set_voice_engine(engine: str) -> None:
    global _voice_engine_override
    _voice_engine_override = engine
    _save_voice_to_db()


def _save_voice_to_db() -> None:
    """Persist voice settings so they survive restarts."""
    try:
        from db import get_conn
        conn = get_conn()
        conn.execute(
            "UPDATE app_settings SET voice_enabled = ?, voice_engine = ? WHERE id = 1",
            (
                1 if get_voice_enabled() else 0,
                get_voice_engine(),
            ),
        )
        conn.commit()
    except Exception:
        pass

# Server
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8800"))

# Upstream API
UPSTREAM_TIMEOUT = int(os.getenv("UPSTREAM_TIMEOUT", "120"))
UPSTREAM_MAX_RETRIES = int(os.getenv("UPSTREAM_MAX_RETRIES", "3"))

# Proxy auth (disabled by default — set AUTH_KEY to enable)
AUTH_KEY = os.getenv("AUTH_KEY", "")

# API key for OpenAI-compatible endpoint (clients send Authorization: Bearer ***)
API_KEY = os.getenv("API_KEY", "")

# Rate limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# ── Agentic cache retrieval (cheap model orchestrates search via tool) ──
# When enabled, the chat path lets the cheap model resolve implicit wording and
# query the cache itself via a search_cache tool, instead of the fixed
# matcher→context-prompt pipeline. Default off (A/B against the matcher).
AGENTIC_CACHE = os.getenv("AGENTIC_CACHE", "0") == "1"

# ── Code-path routing (Switchyard-inspired) ──
CODE_ROUTE_MODE = os.getenv("CODE_ROUTE_MODE", "auto")
SESSION_AFFINITY = os.getenv("SESSION_AFFINITY", "1") == "1"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
JUDGE_BASE_THRESHOLD = float(os.getenv("JUDGE_BASE_THRESHOLD", "0.5"))
JUDGE_THRESHOLD_STEP = float(os.getenv("JUDGE_THRESHOLD_STEP", "0.1"))
STAGE_CONFIDENCE_THRESHOLD = float(os.getenv("STAGE_CONFIDENCE_THRESHOLD", "0.5"))
STAGE_RECENT_TURN_WINDOW = int(os.getenv("STAGE_RECENT_TURN_WINDOW", "3"))
