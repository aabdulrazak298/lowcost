"""Reproduce the user's exact transaction: cache-hit news query through the
real pipeline (agent + web tools + reasoning), measure output truncation."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor import process_query


async def main() -> None:
    print("process_query('August 2026 worldwide news reports') ...", flush=True)
    answer, model_used, _images, usage = await process_query(
        user_query="August 2026 worldwide news reports",
        chat_history="",
    )
    print("model_used:", model_used)
    print("usage:", usage)
    print("answer chars:", len(answer))
    print("answer words:", len(answer.split()))
    print("LAST 300 chars:", repr(answer[-300:]))
    ends_ok = answer.rstrip().endswith((".", "!", "?", '"', "”", "—"))
    print("ends with sentence-ending punctuation:", ends_ok)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
