"""Verify SEM_THRESHOLD 0.50 vs 0.60 against today's real borderline queries.

Runs smart_cache_lookup with an env override on the REAL cache.db (read-only)
and reports which borderline misses become hits. Clearly-unrelated queries
must stay misses.

Run: LCLLM_SEM_THRESHOLD=0.50 .venv/bin/python scripts/test_threshold_050.py
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

THRESH = os.environ.get("LCLLM_SEM_THRESHOLD", "0.50")

# (query, expected at 0.50: hit/miss, note)
CASES = [
    # repeats / same-topic — SHOULD become hits at 0.50
    ("Latest developments and detailed analysis on the US Iran conflict and Strait of Hormuz", "hit", "repeat of id 267"),
    ("Summary of the YouTube video SILO - Season 3 Episode 8 - Everything You Missed!", "hit", "repeat of id 274"),
    ("summary of https://www.youtube.com/watch?v=eLs-BMXuaRs titled Gillian Anderson", "hit", "near-repeat, entry exists"),
    ("How does caching repeated LLM API requests save money on inference costs and reduce latency", "hit", "LLM-cost topic in cache"),
    # first-time / unrelated — MUST stay misses
    ("summarize the CBR article 7 Anime Arcs Where the Hero Dies https://www.cbr.com/anime-arcs-where-the-hero-dies/", "miss", "entry deleted"),
    ("what is the transporter actual limit https://www.youtube.com/watch?v=DpQuWaWA6MU", "miss", "unrelated"),
    ("information about the video game Lies of P", "miss", "pre-write miss (now cached but different follow-up)"),
    ("summarize article about Linus Torvalds fixing a Linux bug using AI highlighting the tools he used", "miss", "pre-write miss"),
]


async def main():
    import matcher

    print(f"== threshold = {THRESH} (matcher.SEM_THRESHOLD={matcher.SEM_THRESHOLD}) ==")
    ok = 0
    for q, expected, note in CASES:
        m = await matcher.smart_cache_lookup(q, purpose="chat")
        if m:
            print(f"HIT  cos~? matched={m.get('query','')[:60]!r}  [{note}]")
            got = "hit"
        else:
            print(f"MISS                                    [{note}]")
            got = "miss"
        if got == expected:
            ok += 1
    print(f"\n{ok}/{len(CASES)} as expected")


if __name__ == "__main__":
    asyncio.run(main())
