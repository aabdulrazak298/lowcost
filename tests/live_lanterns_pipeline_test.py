"""Full pipeline check with the fix (temp 0.3 + anti-denial prompt).

Uses an isolated DB copy so the live service isn't disturbed. Query = pasted
Lanterns doc (matches cache id 171) + leading 'is it fake?' phrasing — the
worst case from the repro.
"""
import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

import config  # noqa: E402

_repro_db = Path("cache.db.pipeline_test")
if _repro_db.exists():
    _repro_db.unlink()
if Path("cache.db").exists():
    shutil.copy2("cache.db", _repro_db)
config.DB_PATH = _repro_db

from llm import set_delivery_context, clear_delivery_context  # noqa: E402
from processor import process_query  # noqa: E402
from config import TELEGRAM_BOT_TOKEN  # noqa: E402

import sqlite3

conn = sqlite3.connect("cache.db")
CACHED_ANSWER = conn.execute("SELECT answer FROM qa_cache WHERE id=171").fetchone()[0]
conn.close()

QUERY = (
    "Someone sent me this document about a TV series called Lanterns and it "
    "looks AI-generated to me. Is this real or fabricated?\n\n" + CACHED_ANSWER
)


async def main() -> None:
    set_delivery_context("telegram", 915519325, TELEGRAM_BOT_TOKEN)
    try:
        resp, model_used, _imgs, usage = await process_query(QUERY)
        print("MODEL:", model_used)
        print(f"REPLY ({len(resp)} chars):\n{resp[:900]}")
        print("\n>>> markers:")
        low = resp.lower()
        print("  'fabricated':", "fabricated" in low)
        print("  'no such':", "no such" in low)
        print("  cites URL (searched):", "http" in low)
    finally:
        clear_delivery_context()


if __name__ == "__main__":
    asyncio.run(main())
