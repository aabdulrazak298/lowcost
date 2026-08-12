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

# Cheap model (configurable — currently DeepSeek V4 Flash via OpenRouter)
CHEAP_API_KEY = os.getenv("CHEAP_API_KEY", "")
CHEAP_BASE_URL = os.getenv("CHEAP_BASE_URL", "https://openrouter.ai/api/v1")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "deepseek-v4-flash")

# Expensive model — DeepSeek V4 Pro direct API (native tool calling)
EXPENSIVE_API_KEY = os.getenv("EXPENSIVE_API_KEY", "")
EXPENSIVE_BASE_URL = os.getenv("EXPENSIVE_BASE_URL", "https://api.deepseek.com/v1")
EXPENSIVE_MODEL = os.getenv("EXPENSIVE_MODEL", "deepseek-v4-pro")

# Fallback model — used if expensive model API fails (via OpenRouter for reliability)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "deepseek/deepseek-v4-flash")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")
FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "https://openrouter.ai/api/v1")

# ── Runtime model overrides (set by /model command) ──────────────────
_cheap_override: str | None = None
_expensive_override: str | None = None


def get_cheap_model() -> str:
    return _cheap_override or CHEAP_MODEL


def get_expensive_model() -> str:
    return _expensive_override or EXPENSIVE_MODEL


def set_cheap_model(model_id: str | None) -> None:
    global _cheap_override
    _cheap_override = model_id


def set_expensive_model(model_id: str | None) -> None:
    global _expensive_override
    _expensive_override = model_id


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
    "r1":         ("deepseek-reasoner", "DeepSeek Direct"),
    "m3":         ("minimax/minimax-m3", "OpenRouter"),
    "gemini":     ("google/gemini-2.5-flash", "OpenRouter"),
    "gemini-lite":("google/gemini-3.1-flash-lite", "OpenRouter"),
    "llama":      ("meta-llama/llama-4-maverick", "OpenRouter"),
    "luna":       ("openai/gpt-5.6-luna", "OpenRouter"),
    "ling":       ("inclusionai/ling-3.0-flash", "OpenRouter"),
    "sonnet":     ("anthropic/claude-sonnet-4", "OpenRouter"),
    "nemotron":   ("nvidia/nemotron-3-ultra-550b-a55b", "OpenRouter"),
    "solar":      ("upstage/solar-pro4", "OpenRouter"),
}

# Output pricing per 1M tokens (keyed by model key)
MODEL_PRICING: dict[str, float] = {
    "qwen": 0.13, "qwen36": 0.26, "qwen35f": 0.26, "qwen35b": 0.80,
    "qwen-plus": 1.28, "flash": 0.28, "flash-ds": 0.28, "pro": 0.87,
    "r1": 2.19, "m3": 2.40, "gemini": 2.50, "gemini-lite": 1.50,
    "llama": 0.85, "luna": 6.00, "ling": 0.021, "sonnet": 15.00, "nemotron": 2.20, "solar": 0.12,
}


def estimate_cost(model_id: str, output_chars: int) -> float:
    """Estimate cost based on model output pricing and char count (3.5 chars ≈ 1 token)."""
    for key, (full_id, _) in AVAILABLE_MODELS.items():
        if full_id.lower() in model_id.lower():
            price = MODEL_PRICING.get(key, 0.28)
            tokens = output_chars / 3.5
            return (tokens / 1_000_000) * price
    return 0.0


# Telegram bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USERS = os.getenv("TELEGRAM_ALLOWED_USERS", "")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "")

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
