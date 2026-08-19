"""Live E2E: full pipeline — model decides to call send_file for a .md file.

Runs process_query (cache → cheap → expensive) with Telegram delivery bound,
exactly like telegram_bot._handle_message does. Verifies the agentic loop
actually invokes send_file when asked for a file.
"""
import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

# Isolate from the LIVE service's cache.db (it holds a write lock via WAL).
import config  # noqa: E402

_live_test_db = Path("cache.db.live_test")
if _live_test_db.exists():
    _live_test_db.unlink()
if Path("cache.db").exists():
    shutil.copy2("cache.db", _live_test_db)
config.DB_PATH = _live_test_db

from llm import set_delivery_context, clear_delivery_context  # noqa: E402
from processor import process_query  # noqa: E402
from config import TELEGRAM_BOT_TOKEN  # noqa: E402

CHAT = 915519325


async def main() -> None:
    set_delivery_context("telegram", CHAT, TELEGRAM_BOT_TOKEN)
    try:
        response, model_used, _images, usage = await process_query(
            "Send this as a .md file please. Content:\n\n"
            "# Checklist\n\n"
            "- Verify 4-20mA loop at 4mA and 20mA\n"
            "- Check transmitter zero/span\n"
            "- Confirm wiring to correct AI channel\n"
        )
        print("MODEL:", model_used)
        print("USAGE:", usage)
        print("REPLY:", response[:500])
    finally:
        clear_delivery_context()


if __name__ == "__main__":
    asyncio.run(main())
