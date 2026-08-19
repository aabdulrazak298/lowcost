"""Repro: does qwen3.7-flash (cheap, cached path) call a CORRECT cached answer 'fabricated'?

Part A: replicate processor.py's cached-path messages exactly (system + [Example] =
cache entry 171 = the real Lanterns answer) + user pasting that same doc for
verification. This is the path that produced the bad answer.
Part B: control — same query on the expensive model (deepseek-v4-pro) with no example.
"""
import asyncio
import sqlite3
import sys

sys.path.insert(0, ".")

from processor import _DATE_CONTEXT  # noqa: E402
from llm import call_cheap, call_expensive  # noqa: E402

conn = sqlite3.connect("cache.db")
row = conn.execute("SELECT answer FROM qa_cache WHERE id=171").fetchone()
CACHED_ANSWER = row[0]
conn.close()

# The user's query: pasted the previous answer (or similar doc) and asked to verify.
DOC = CACHED_ANSWER[:1200]
USER_QUERY = f"Is this document accurate? Please verify each claim.\n\n{DOC}"

CACHED_PATH_SYSTEM = (
    f"You are a helpful assistant. {_DATE_CONTEXT}\n"
    "Use web_search directly to get fresh, current information whenever "
    "the example is stale, time-sensitive, or incomplete — search and "
    "answer yourself; never ask the user for permission to search.\n"
    "Use run_code to execute Python for ANY arithmetic, calculation, "
    "enumeration, or multi-step computation — never compute in your head.\n"
    "If the user's message contains an [Example] block about a DIFFERENT "
    "topic than their actual question, ignore it and answer from the "
    "conversation instead. If you still cannot answer, reply with exactly "
    "one word: IRRELEVANT"
)


async def part_a() -> None:
    print("=" * 60)
    print("PART A — cheap model (qwen3.7-flash), cached path with [Example]")
    print("=" * 60)
    messages = [
        {"role": "system", "content": CACHED_PATH_SYSTEM},
        {"role": "user", "content": (
            "[Example — a similar question and how it was solved. Follow this "
            "pattern and adapt it to the new question.]\n"
            f"{CACHED_ANSWER}\n\n"
            f"{USER_QUERY}"
        )},
    ]
    answer = await call_cheap(messages, tools=None, reasoning=True)
    print(f"\n--- CHEAP ANSWER ({len(answer)} chars) ---\n{answer[:1500]}")
    print("\n>>> verdict markers:")
    print("  'fabricated' in answer:", "fabricated" in answer.lower())
    print("  'does not exist' in answer:", "does not exist" in answer.lower())
    print("  cites a URL (searched):", "http" in answer.lower())


async def part_b() -> None:
    print("\n" + "=" * 60)
    print("PART B — control: expensive model (deepseek-v4-pro), same query, no example")
    print("=" * 60)
    messages = [
        {"role": "system", "content": _DATE_CONTEXT},
        {"role": "user", "content": USER_QUERY},
    ]
    answer, model = await call_expensive(messages)
    print(f"\n--- EXPENSIVE ANSWER ({len(answer)} chars) — model={model} ---\n{answer[:1200]}")
    print("\n>>> verdict markers:")
    print("  'fabricated' in answer:", "fabricated" in answer.lower())
    print("  confirms real:", "real" in answer.lower() or "accurate" in answer.lower())
    print("  cites a URL (searched):", "http" in answer.lower())


if __name__ == "__main__":
    async def main() -> None:
        await part_a()
        await part_b()

    asyncio.run(main())
