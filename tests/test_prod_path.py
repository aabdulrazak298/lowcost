"""Pre-deploy check: exercise the REAL call_cheap (Agents SDK) with reasoning=True
and the new example framing + run_code steer, exactly as process_query now does.

Loads the live model override from cache.db (read-only), then calls call_cheap
with a hard variant question + a base example, and checks the answer is correct.

Run: cd ~/cloud/projects/lowcostllm && .venv/bin/python tests/test_prod_path.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import _load_overrides_from_db, get_cheap_model
from llm import call_cheap

_load_overrides_from_db()
print(f"cheap model (from DB override): {get_cheap_model()}")

_DATE_CONTEXT = "Today's date is Sunday, 16 August 2026."

SYSTEM = (
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

CASES = [
    {
        "id": "tank",
        "correct": "5.5",
        "base_answer": ("EXAMPLE SOLUTION: V=pi*(1.0)^2*(3.0)=9.4248 m3. 80%=7.5398 m3, "
                        "25%=2.3562 m3. Drain 7.5398-2.3562=5.1836 m3. 12 L/s=0.012 m3/s. "
                        "t=5.1836/0.012=431.97 s = 7.2 minutes."),
        "variant_q": "A cylindrical tank (diameter 1.5 m, height 2.5 m) is 90% full. A pump drains it at 8 L/s. How many minutes (one decimal) to drain down to 30% full?",
    },
    {
        "id": "perm",
        "correct": "10",
        "base_answer": ("EXAMPLE SOLUTION: enumerate all 6!=720 permutations, keep those "
                        "satisfying the constraints, count survivors -> 24."),
        "variant_q": "Six people A, B, C, D, E, F line up in a row. How many valid orderings satisfy ALL of: A before B; B before C; C not first; D immediately after E; F before D?",
    },
]


async def main():
    for case in CASES:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                f"[Example — a similar question and how it was solved. Follow this "
                f"pattern and adapt it to the new question.]\n"
                f"{case['base_answer']}\n\n"
                f"{case['variant_q']}"
            )},
        ]
        print(f"\n=== [{case['id']}] expected {case['correct']} ===")
        ans = await call_cheap(messages, tools=None, reasoning=True)
        print(f"ANSWER: {ans[:300]}")
        print(f"correct? {'YES' if case['correct'] in ans else 'CHECK'}")

    print("\nDONE")


if __name__ == "__main__":
    asyncio.run(main())
