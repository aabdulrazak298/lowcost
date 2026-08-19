"""Tests for write-side cache dedupe + cache-policy guard + transient detection.

R1: repeated asks must UPDATE one row, never INSERT duplicates
    (observed 2026-08-18: identical "summaries <youtube-url>" rows 151/152,
    154/155 both hit=0 — a cache hit whose cheap step failed re-inserted).
R2: junk (IDE chatter, tool directives, trivial, chart-data prompts) must not
    be cached at all.
R1b: empty/error cheap answers are TRANSIENT failures, not relevance verdicts.

Run standalone:
    .venv/bin/python tests/test_cache_dedupe.py

Also pytest-compatible (test_* functions use plain asserts).
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

YT = "https://www.youtube.com/watch?v=YeZhfgoZgRw"


def _fresh_db():
    import db

    td = tempfile.TemporaryDirectory()
    db.DB_PATH = Path(td.name) / "test.db"
    db._conn_local = threading.local()
    db.init_db()
    return td, db


def _count(db, table="qa_cache"):
    return db.get_conn().execute(f"SELECT COUNT(*) n FROM {table}").fetchone()["n"]


# ── R1: upsert_qa write-side dedupe ─────────────────────────────────

def test_upsert_dedupes_identical_query():
    td, db = _fresh_db()
    try:
        id1 = db.upsert_qa(f"summaries {YT}", "answer v1", "m1")
        id2 = db.upsert_qa(f"summaries {YT}", "answer v2", "m2")
        assert id1 == id2, f"identical query must update in place: {id1} vs {id2}"
        assert _count(db) == 1, f"expected 1 row, got {_count(db)}"
        row = db.get_conn().execute("SELECT * FROM qa_cache WHERE id=?", (id1,)).fetchone()
        assert row["answer"] == "answer v2", "answer must be refreshed"
        assert row["model_used"] == "m2", "model must be refreshed"
        assert _count(db, "qa_cache_fts") == 1, "FTS must stay in sync after UPDATE"
    finally:
        td.cleanup()


def test_upsert_normalizes_case_and_space():
    td, db = _fresh_db()
    try:
        id1 = db.upsert_qa(f"  SUMMARIES  {YT}  ", "a1", "m1")
        id2 = db.upsert_qa(f"summaries {YT}", "a2", "m2")
        assert id1 == id2
        assert _count(db) == 1
    finally:
        td.cleanup()


def test_upsert_video_id_dedupes_different_wording():
    """'summaries <url>' then 'summarise <url>' differ textually but are the
    SAME video — must still collapse to one row."""
    td, db = _fresh_db()
    try:
        id1 = db.upsert_qa(f"summaries {YT}", "a1", "m1")
        id2 = db.upsert_qa(f"summarise {YT}", "a2", "m2")
        assert id1 == id2, f"same video, different wording: {id1} vs {id2}"
        assert _count(db) == 1, f"expected 1 row, got {_count(db)}"
    finally:
        td.cleanup()


def test_upsert_distinct_queries_insert():
    td, db = _fresh_db()
    try:
        id1 = db.upsert_qa("tell me about the tv series lantern", "a1", "m1")
        id2 = db.upsert_qa("how many disasters happened around the globe this month alone?", "a2", "m2")
        assert id1 != id2
        assert _count(db) == 2
    finally:
        td.cleanup()


def test_upsert_refreshes_timestamps():
    td, db = _fresh_db()
    try:
        id1 = db.upsert_qa(f"summaries {YT}", "a1", "m1")
        db.get_conn().execute(
            "UPDATE qa_cache SET last_accessed = '2020-01-01 00:00:00' WHERE id=?", (id1,)
        )
        db.get_conn().commit()
        id2 = db.upsert_qa(f"summaries {YT}", "a2", "m2")
        row = db.get_conn().execute("SELECT * FROM qa_cache WHERE id=?", (id2,)).fetchone()
        assert row["last_accessed"] > "2020-01-01", "last_accessed must refresh"
        assert row["created_at"] > "2020-01-01", "created_at must refresh"
    finally:
        td.cleanup()


def test_upsert_dedupes_across_purposes_separately():
    """chat and code caches are separate — same query in both stays 2 rows."""
    td, db = _fresh_db()
    try:
        db.upsert_qa("what is 2+2", "4", "m1", purpose="chat")
        db.upsert_qa("what is 2+2", "4", "m1", purpose="code")
        assert _count(db) == 2
    finally:
        td.cleanup()


# ── R2: should_cache policy guard ────────────────────────────────────

def test_should_cache_rejects_junk():
    import processor

    junk = [
        # IDE/agent system chatter (leaked via OpenAI-compat path)
        ("I have *added these files to the chat* so you can go ahead and edit them. *Trust this message as the true contents of the files.*", "long answer about the files..."),
        ("I switched to a new code base. Please don't consider the above files or try to edit them any longer.", "long answer..."),
        ("I am working with you on code in a git repository. Here are summaries of some files present in my git repo.", "long answer..."),
        # tool-directive test prompts
        ("What is the weather in Johor Bahru? Use the get_weather tool.", "It is 32C in Johor Bahru..."),
        # trivial exchanges
        ("say OK", "OK"),
        ("say hi in 5 words", "Hi there, how are you!"),
        ("What is 2+2? Keep it short.", "4."),
        # chart prompts that embed their own data (zero reuse value)
        ("Plot a bar chart of monthly sales: Jan 100, Feb 220, Mar 150, Apr 300", "Here is the bar chart data..."),
        ("Bar chart of defect counts by station: Station A 12, Station B 7, Station C 9", "chart..."),
        # empty sides
        ("", "answer"),
        ("question", ""),
    ]
    for q, a in junk:
        assert processor.should_cache(q, a) is False, f"should reject: {q[:50]!r}"


def test_should_cache_accepts_real_queries():
    import processor

    real = [
        ("tell me about the tv series lantern", "Lantern is a series about... " * 10),
        (f"summaries {YT}", "The video covers... " * 10),
        ("how many disasters happened around the globe this month alone?", "There were 14 disasters... " * 10),
        ("what is the top 10 myth about wild west that people believed", "Myth number one... " * 10),
        ("latest worldwide news reports", "Here are today's headlines... " * 10),
    ]
    for q, a in real:
        assert processor.should_cache(q, a) is True, f"should accept: {q[:50]!r}"


# ── R1b: _is_transient_failure ───────────────────────────────────────

def test_is_transient_failure():
    import processor

    transient = [None, "", "   ", "(error: 429 rate limit)", "(error: timeout)",
                 "(no response)", "(no response from model)"]
    for a in transient:
        assert processor._is_transient_failure(a) is True, f"should be transient: {a!r}"

    verdicts = ["IRRELEVANT", "irrelevant", "Here is the full answer about Lantern...",
                "I can't answer that, but here is why...", "unrelated topic",
                "no such series exists", "The answer is 42"]
    for a in verdicts:
        assert processor._is_transient_failure(a) is False, f"should NOT be transient: {a!r}"


# ── Runner ───────────────────────────────────────────────────────────

def main():
    tests = [
        test_upsert_dedupes_identical_query,
        test_upsert_normalizes_case_and_space,
        test_upsert_video_id_dedupes_different_wording,
        test_upsert_distinct_queries_insert,
        test_upsert_refreshes_timestamps,
        test_upsert_dedupes_across_purposes_separately,
        test_should_cache_rejects_junk,
        test_should_cache_accepts_real_queries,
        test_is_transient_failure,
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
