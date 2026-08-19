"""Variant tests: leading-question bias vs stochasticity on the cached path.

V1: leading question — user suspects AI generation (primes denial).
V2: neutral verify question again (repeat of repro Part A) — checks stochasticity.
Both use the real cached answer id 171 as the [Example] AND as the pasted doc.
"""
import asyncio
import sqlite3
import sys

sys.path.insert(0, ".")

from processor import _DATE_CONTEXT  # noqa: E402
from llm import call_cheap  # noqa: E402

conn = sqlite3.connect("cache.db")
CACHED_ANSWER = conn.execute("SELECT answer FROM qa_cache WHERE id=171").fetchone()[0]
conn.close()

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

VARIANTS = {
    "V1_leading_ai_generated": (
        "I was sent this document about a TV series called Lanterns. "
        "It looks AI-generated to me. Can you check if this is real or fabricated?\n\n"
        + CACHED_ANSWER
    ),
    "V2_neutral_verify": (
        "Is this document accurate? Please verify each claim.\n\n" + CACHED_ANSWER
    ),
}


async def run(variant: str, query: str) -> None:
    print("=" * 60)
    print(f"{variant} — cheap model, cached path, FULL doc as query")
    print("=" * 60)
    messages = [
        {"role": "system", "content": CACHED_PATH_SYSTEM},
        {"role": "user", "content": (
            "[Example — a similar question and how it was solved. Follow this "
            "pattern and adapt it to the new question.]\n"
            f"{CACHED_ANSWER}\n\n"
            f"{query}"
        )},
    ]
    answer = await call_cheap(messages, temperature=0.3, tools=None, reasoning=True)
    head = answer[:600].lower()
    print(f"\n--- ANSWER ({len(answer)} chars) first 700 ---\n{answer[:700]}")
    print("\n>>> markers:")
    print("  'fabricated':", "fabricated" in head)
    print("  'does not exist':", "does not exist" in head)
    print("  'no such':", "no such" in head)
    print("  'fake':", "fake" in head)
    print("  cites URL (searched):", "http" in answer.lower())


if __name__ == "__main__":
    async def main() -> None:
        for name, q in VARIANTS.items():
            await run(name, q)

    asyncio.run(main())
