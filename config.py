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

# Cheap model (configurable — currently Qwen 3.5 Flash via OpenRouter)
CHEAP_API_KEY = os.getenv("CHEAP_API_KEY", "")
CHEAP_BASE_URL = os.getenv("CHEAP_BASE_URL", "https://openrouter.ai/api/v1")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "deepseek-v4-flash")

# Expensive model — DeepSeek V4 Pro direct API (native tool calling)
EXPENSIVE_API_KEY = os.getenv("EXPENSIVE_API_KEY", "")
EXPENSIVE_BASE_URL = os.getenv(
    "EXPENSIVE_BASE_URL", "https://api.deepseek.com/v1"
)
EXPENSIVE_MODEL = os.getenv("EXPENSIVE_MODEL", "deepseek-v4-pro")

# Fallback model — used if expensive model API fails (via OpenRouter for reliability)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "deepseek/deepseek-v4-flash")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")
FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "https://openrouter.ai/api/v1")

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
