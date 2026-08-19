"""Tests for the recency-tiebreaker ranking (_score_and_pick) + ephemeral
classification + cache_store policy wrapper.

Step 1+2 of the freshness plan (2026-08-19, Gemini-reviewed):
- recency is a SMALL tiebreaker (0.05) in the hybrid; the absolute cosine gate
  can never be overridden by how new an entry is
- is_ephemeral_query / cache_store classify writes (measurement only — expiry
  enforcement is step 3, gated on real numbers)

Run standalone:
    .venv/bin/python tests/test_matcher_rank.py
"""
import os
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matcher  # noqa: E402


def ts(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def pick(sems, bm25s, created, threshold=0.45):
    idx, _ = matcher._score_and_pick(
        np.array(sems, dtype=np.float32),
        np.array(bm25s, dtype=np.float32),
        created,
        threshold,
    )
    return idx


# ── Ranking behaviour ───────────────────────────────────────────────

def test_newest_wins_near_tie():
    """Two candidates with ~equal cosine + bm25: newest entry wins."""
    idx = pick(
        sems=[0.70, 0.71],        # cosine nearly tied (old slightly lower)
        bm25s=[-5.0, -4.5],       # bm25 nearly tied (old slightly lower)
        created=[ts(30), ts(1)],  # 30 days vs 1 day
    )
    assert idx == 1, f"newest must win the near-tie, got idx={idx}"


def test_semantic_dominates_recency():
    """A clearly better OLD answer must beat a mediocre NEW one."""
    idx = pick(
        sems=[0.90, 0.65],        # old is far more relevant
        bm25s=[-3.0, -6.0],       # old also better lexically
        created=[ts(60), ts(0.1)],
    )
    assert idx == 0, f"semantic must dominate recency, got idx={idx}"


def test_below_gate_never_wins_even_if_newest():
    """Sub-threshold candidate can NEVER win, no matter how new."""
    idx = pick(
        sems=[0.44, 0.90],        # candidate 0 just below gate, 1 above
        bm25s=[-9.0, -2.0],       # below-gate one is lexically strong
        created=[ts(0.01), ts(90)],
    )
    assert idx == 1, f"gate must never be overridden, got idx={idx}"


def test_all_below_gate_returns_something():
    """argmax still returns an index (caller gates separately) — sanity only."""
    idx = pick(sems=[0.3, 0.2], bm25s=[-1.0, -2.0], created=[ts(1), ts(2)])
    assert idx in (0, 1)


def test_single_candidate_picked():
    idx = pick(sems=[0.6], bm25s=[-1.0], created=[ts(5)])
    assert idx == 0


def test_equal_created_at_semantic_decides():
    """Same created_at → recency neutral → cosine decides."""
    idx = pick(
        sems=[0.70, 0.85],
        bm25s=[-5.0, -5.0],
        created=[ts(3), ts(3)],
    )
    assert idx == 1, "equal age → higher cosine wins"


def test_unparseable_created_at_neutral():
    """Malformed created_at must not crash or dominate."""
    idx = pick(sems=[0.8, 0.79], bm25s=[-4.0, -3.9], created=[None, ts(1)])
    assert idx in (0, 1)


# ── Ephemeral classification ────────────────────────────────────────

def test_is_ephemeral_query():
    import processor

    ephemeral = [
        "latest worldwide news reports",
        "how many disasters happened around the globe this month alone?",
        "is earthquake this year still common or frequent than normal",
        "what is the weather in Johor Bahru",
        "breaking: new update on the election",
        "what is the stock price of tesla today",
    ]
    evergreen = [
        "tell me about the tv series lantern",
        "summaries https://www.youtube.com/watch?v=YeZhfgoZgRw",
        "top 10 amy adams movies",
        "is qwen3.7 an open source model?",
        "write a good birthday wish for me, his name is John",
    ]
    for q in ephemeral:
        assert processor.is_ephemeral_query(q) is True, f"ephemeral: {q!r}"
    for q in evergreen:
        assert processor.is_ephemeral_query(q) is False, f"evergreen: {q!r}"


def test_cache_store_writes_and_classifies():
    import processor
    import db

    td = tempfile.TemporaryDirectory()
    try:
        db.DB_PATH = Path(td.name) / "test.db"
        db._conn_local = threading.local()
        db.init_db()

        processor.cache_store("latest news reports", "answer news " * 10, "m1")
        processor.cache_store("tell me about the tv series lantern", "answer tv " * 10, "m1")
        processor.cache_store("say OK", "OK", "m1")  # junk — skipped by guard
        assert db.get_conn().execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0] == 2

        rows = db.get_conn().execute(
            "SELECT query FROM qa_cache ORDER BY query"
        ).fetchall()
        assert rows[0]["query"] == "latest news reports"
        assert rows[1]["query"] == "tell me about the tv series lantern"
    finally:
        td.cleanup()


# ── Runner ───────────────────────────────────────────────────────────

def main():
    tests = [
        test_newest_wins_near_tie,
        test_semantic_dominates_recency,
        test_below_gate_never_wins_even_if_newest,
        test_all_below_gate_returns_something,
        test_single_candidate_picked,
        test_equal_created_at_semantic_decides,
        test_unparseable_created_at_neutral,
        test_is_ephemeral_query,
        test_cache_store_writes_and_classifies,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
