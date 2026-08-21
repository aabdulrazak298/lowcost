"""Cache false-positive gate verification (2026-08-21 fix).

Run: .venv/bin/python scripts/tests/test_cache_gates.py
Covers: LCLLM_SEM_THRESHOLD=0.60, ephemeral write/read gates,
media-no-URL skip, decade-mismatch skip, gatekeeper parse + eviction.

Live gatekeeper LLM calls (4 cheap calls) only when GATE_LIVE=1:
GATE_LIVE=1 .venv/bin/python scripts/tests/test_cache_gates.py

Exit 0 = all checks pass. Uses the LIVE cache.db read-only.
"""
import asyncio
import logging
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.WARNING)

import config  # noqa: F401  (load_dotenv before matcher reads env)
from matcher import smart_cache_lookup, SEM_THRESHOLD, EPHEMERAL_TTL_HOURS, _decade_buckets
from db import is_ephemeral_query, hot_cache_put, hot_cache_lookup, hot_cache_delete
from processor import _is_media_summary_no_url, _parse_gate_verdict

FAIL = []


def check(label, got, exp):
    ok = got == exp
    print(f"  {'OK ' if ok else 'FAIL'} {label}: {got}" + ("" if ok else f"  (exp {exp})"))
    if not ok:
        FAIL.append(label)


print(f"SEM_THRESHOLD={SEM_THRESHOLD} EPHEMERAL_TTL_HOURS={EPHEMERAL_TTL_HOURS}")
check("threshold >= 0.60", SEM_THRESHOLD >= 0.60, True)

print("--- ephemeral classifier ---")
check("latest news -> ephemeral", is_ephemeral_query("latest worldwide news reports"), True)
check("this month -> ephemeral", is_ephemeral_query("how many disasters this month alone?"), True)
check("news URL NOT ephemeral", is_ephemeral_query("verify this https://www.abc.net.au/news/2026-08-17/x"), False)
check("evergreen NOT ephemeral", is_ephemeral_query("how to renew chargeman a0"), False)

print("--- media-no-URL skip ---")
check("video by title -> skip", _is_media_summary_no_url("Summarize the Sean Foo video America Panics as Bonds"), True)
check("yt URL -> not skip", _is_media_summary_no_url("summarize https://www.youtube.com/watch?v=4EPW0Ht7XCc more"), False)
check("tv series info -> not skip", _is_media_summary_no_url("what is the latest season of futurama"), False)
check("meeting notes -> not skip", _is_media_summary_no_url("summarize our meeting notes from yesterday"), False)

print("--- decade buckets ---")
check("1980s -> {1980}", _decade_buckets("1980s"), {1980})
check("1990 -> {1990}", _decade_buckets("1990"), {1990})
check("2020 to 2024 -> {2020}", _decade_buckets("2020 to 2024"), {2020})
check("80s shorthand -> {}", _decade_buckets("the 80s"), set())
check("multi-year -> buckets", _decade_buckets("1978 and 1985"), {1970, 1980})

print("--- gatekeeper parse ---")
for raw, exp in [("YES", True), ("NO", False), (" yes.", True), ("NO — different topic", False),
                 ("maybe", None), ("", None)]:
    check(f"parse {raw!r}", _parse_gate_verdict(raw), exp)

print("--- hot-cache eviction (gate-rejected hit must not be re-served) ---")
async def eviction_test():
    fake = {"id": 999999, "query": "reject me", "answer": "x", "source": "semantic"}
    await hot_cache_put("reject me", fake)
    present = bool(await hot_cache_lookup("reject me"))
    await hot_cache_delete("reject me")
    gone = not bool(await hot_cache_lookup("reject me"))
    check("eviction removes hot entry", present and gone, True)
asyncio.run(eviction_test())


async def live_gate():
    """GATE_LIVE=1: one cheap call per pair, real qwen3.7-flash verdicts."""
    from processor import _relevance_gate
    import sqlite3
    con = sqlite3.connect('file:cache.db?mode=ro', uri=True)

    def row(q):
        r = con.execute("SELECT id, query, answer FROM qa_cache WHERE query LIKE ? ORDER BY id DESC LIMIT 1",
                        (f"%{q}%",)).fetchone()
        return {"id": r[0], "query": r[1], "answer": r[2][:1500]} if r else None

    pairs = [
        ("why does the united states government not declare bankruptcy", "bankruptcy", True),
        ("top 10 most popular white female baby names from the 1980s", "baby girl names in the United States during the 1990s", False),
        ("Summarize the Sean Foo video America Panics as Bonds", "FBI LETTERS: TRUMP TRAFFICKED", False),
        ("what is the latest season of futurama", "latest season of futurama", True),
    ]
    for q, pat, exp in pairs:
        m = row(pat)
        if not m:
            print(f"  SKIP no row for {pat!r}")
            continue
        got = await _relevance_gate(q, m)
        check(f"gate {q[:40]!r}", got, exp)


if os.environ.get("GATE_LIVE") == "1":
    print("--- live gatekeeper (GATE_LIVE=1) ---")
    asyncio.run(live_gate())
else:
    print("--- live gatekeeper skipped (set GATE_LIVE=1 to run 4 cheap calls) ---")


async def main():
    print("--- live semantic lookups (expect all MISS/SKIP except marked HIT) ---")
    cases = [
        ("latest worldwide news reports", None, "stale ephemeral read-gate"),
        ("latest earthquake verified news reports 21 August 2026", None, "threshold 0.60"),
        ("number of earthquakes magnitude 7.0 or greater that occurred in 2026", None, "threshold 0.60"),
        ("top 10 most popular white female baby names from the 1980s", None, "decade-mismatch vs 1990s"),
        ("Top 10 most popular white baby girl names from 2020 to 2024", None, "decade-mismatch vs 1990s"),
        ("why does the united states government not declare bankruptcy", "HIT", "evergreen legit hit"),
        ("top 10 most popular artists in the 80s", "HIT", "exact self, no century token"),
    ]
    for q, exp, note in cases:
        m = await smart_cache_lookup(q, purpose="chat")
        got = "MISS" if m is None else "HIT"
        ok = (got == exp) if exp else (got == "MISS")
        print(f"  {'OK ' if ok else 'FAIL'} {got} | {q[:60]!r} ({note})")
        if not ok:
            FAIL.append(q[:60])

asyncio.run(main())
print("---")
if FAIL:
    print(f"FAILED: {len(FAIL)} checks")
    sys.exit(1)
print("ALL CHECKS PASSED")
