"""Verify the append-only rewrite prompt: stability + context resolution.

1. Stability: re-rewrite the same 12 stored keys as the earlier run
   (old results: id 281 cos 0.690 UNSTABLE, id 273 cos 0.772 UNSTABLE,
   id 274 cos 0.948, id 272 cos 0.931). Expect higher cosines now.
2. Referential: "number 5" / "that video" must still resolve from history.
3. Date anchoring: "today" must still become an absolute date.

Run: .venv/bin/python scripts/test_rewrite_stability.py  (uses new prompt)
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import processor
import matcher


async def stability_one(k: str) -> tuple[str, float]:
    k2 = await processor.generate_search_query(k, [])
    if not k2:
        return k2, 0.0
    a = matcher._embed([k])
    b = matcher._embed([k2])
    return k2, float(matcher._cosine_scores(a, b)[0])


async def main():
    c = sqlite3.connect("cache.db")
    rows = c.execute(
        "SELECT id, query FROM qa_cache "
        "WHERE created_at > '2026-08-20' AND length(query) > 20 "
        "ORDER BY id DESC LIMIT 12"
    ).fetchall()
    print(f"== rewrite self-stability (new append-only prompt, n={len(rows)}) ==")
    print(f"{'id':>4} {'cos':>6}  stored-key → re-rewritten")
    unstable = 0
    for rid, k in rows:
        k2, cos = await stability_one(k)
        flag = "UNSTABLE" if cos < 0.85 else ""
        if cos < 0.85:
            unstable += 1
        print(f"{rid:>4} {cos:6.3f}  {k[:45]!r} → {k2[:45]!r} {flag}")
    print(f"\nunstable (<0.85): {unstable}/{len(rows)}  (old: 2/12)")

    print("\n== referential resolution (history given) ==")
    turns = [
        {"role": "user", "content": "List the 5 largest rivers in the world by length"},
        {"role": "assistant", "content": "1. Nile 2. Amazon 3. Yangtze 4. Mississippi 5. Yenisei"},
    ]
    for q in ["what about number 5?", "tell me more about that one"]:
        key = await processor.generate_search_query(q, turns)
        print(f"  {q!r}\n    → {key!r}")

    print("\n== date anchoring ==")
    key = await processor.generate_search_query("what are today's top world news", [])
    print(f"  'what are today's top world news'\n    → {key!r}")
    key2 = await processor.generate_search_query("is there any sequel?", [])
    print(f"  'is there any sequel?' (no history)\n    → {key2!r}")


if __name__ == "__main__":
    asyncio.run(main())
