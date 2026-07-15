"""Central configuration — all tunables live here."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DB_PATH = ROOT / "cache.db"

SIMILARITY_THRESHOLD = int(os.getenv("SIMILARITY_THRESHOLD", "48"))
CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "365"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "1000000"))  # ~800 MB

# Cheap model (configurable — currently Qwen 3.5 Flash via OpenRouter)
CHEAP_API_KEY = os.getenv("CHEAP_API_KEY", "")
CHEAP_BASE_URL = os.getenv("CHEAP_BASE_URL", "https://openrouter.ai/api/v1")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "qwen/qwen3.5-flash-02-23")

# Expensive model — DeepSeek V4 Pro direct API (native tool calling)
EXPENSIVE_API_KEY = os.getenv("EXPENSIVE_API_KEY", "")
EXPENSIVE_BASE_URL = os.getenv(
    "EXPENSIVE_BASE_URL", "https://api.deepseek.com/v1"
)
EXPENSIVE_MODEL = os.getenv("EXPENSIVE_MODEL", "deepseek-v4-pro")

# Fallback model — used if expensive model API fails (circuit breaker)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "qwen/qwen3.5-flash-02-23")

# Telegram bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USERS = os.getenv("TELEGRAM_ALLOWED_USERS", "")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "")

# Server
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8800"))
