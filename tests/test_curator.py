"""Tests for the cache curator (poison-eviction) layer.

Layer 1 of the cache-poisoning fix: the expensive model acts as curator.
When the cheap model rejects a cache match (IRRELEVANT), the expensive model
issues a verdict on the rejected entry — EVICT (poisoned) or KEEP (valid but
mismatched) — and EVICT deletes the row.

Run standalone:
    .venv/bin/python tests/test_curator.py

Also pytest-compatible (test_* functions use plain asserts).
"""
import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path

# Make project root importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Pure logic: verdict parsing ──────────────────────────────────────

def test_parse_curator_verdict():
    import curator

    cases = {
        "EVICT": "EVICT",
        "evict": "EVICT",
        "EVICT this entry": "EVICT",
        "evict because the answer is wrong": "EVICT",
        "KEEP": "KEEP",
        "keep": "KEEP",
        "KEEP, entry is valid": "KEEP",
        "this is fine, KEEP": "KEEP",   # first token isn't EVICT → safe default
        "unclear": "KEEP",
        "": "KEEP",
        "   ": "KEEP",
        "EVICT.": "EVICT",              # trailing punctuation stripped
    }
    for raw, want in cases.items():
        got = curator.parse_curator_verdict(raw)
        assert got == want, f"parse({raw!r}) = {got!r}, want {want!r}"

    # None safety — a malformed/absent verdict must never evict.
    assert curator.parse_curator_verdict(None) == "KEEP"


def test_build_curator_messages():
    import curator

    msgs = curator.build_curator_messages("poisoned question", "poisoned answer")
    assert isinstance(msgs, list) and len(msgs) == 2, msgs
    assert msgs[0]["role"] == "system", msgs
    assert msgs[1]["role"] == "user", msgs
    # Both the cached question and answer must reach the model, plus the
    # two valid verdict words.
    body = msgs[1]["content"]
    assert "poisoned question" in body
    assert "poisoned answer" in body
    # Verdict words (EVICT/KEEP) live in the SYSTEM prompt — the task
    # instructions — not in the user data message.
    sys_prompt = msgs[0]["content"]
    assert "EVICT" in sys_prompt and "KEEP" in sys_prompt


# ── DB layer: delete_cache_entry ─────────────────────────────────────

def _fresh_temp_db():
    import db

    td = tempfile.TemporaryDirectory()
    db.DB_PATH = Path(td.name) / "test.db"
    db._conn_local = threading.local()  # force fresh per-thread connections
    db.init_db()
    return td, db


def test_delete_cache_entry():
    td, db = _fresh_temp_db()
    try:
        rid = db.insert_qa("poison q", "poison a", "test-model")
        assert rid > 0

        assert db.delete_cache_entry(rid) == 1
        n = db.get_conn().execute(
            "SELECT COUNT(*) FROM qa_cache WHERE id=?", (rid,)
        ).fetchone()[0]
        assert n == 0, "row should be gone after delete"

        # Deleting a non-existent id returns 0, not an error.
        assert db.delete_cache_entry(rid) == 0
    finally:
        td.cleanup()


# ── Orchestration: run_curator evicts / keeps ────────────────────────

def test_run_curator_evicts():
    import curator
    import db

    td = tempfile.TemporaryDirectory()
    try:
        db.DB_PATH = Path(td.name) / "test.db"
        db._conn_local = threading.local()
        db.init_db()
        rid = db.insert_qa("poison q", "poison a", "test-model")
        match = {"id": rid, "query": "poison q", "answer": "poison a"}

        async def fake_verdict(cached_q, cached_a):
            assert cached_q == "poison q"
            assert cached_a == "poison a"
            return "EVICT"

        async def _run():
            return await curator.run_curator(match, verdict_fn=fake_verdict)

        verdict = asyncio.run(_run())
        assert verdict == "EVICT"

        n = db.get_conn().execute(
            "SELECT COUNT(*) FROM qa_cache WHERE id=?", (rid,)
        ).fetchone()[0]
        assert n == 0, "EVICT verdict must delete the poisoned row"
    finally:
        td.cleanup()


def test_run_curator_keeps():
    import curator
    import db

    td = tempfile.TemporaryDirectory()
    try:
        db.DB_PATH = Path(td.name) / "test.db"
        db._conn_local = threading.local()
        db.init_db()
        rid = db.insert_qa("valid q", "valid a", "test-model")
        match = {"id": rid, "query": "valid q", "answer": "valid a"}

        async def fake_verdict(cached_q, cached_a):
            return "KEEP"

        verdict = asyncio.run(curator.run_curator(match, verdict_fn=fake_verdict))
        assert verdict == "KEEP"

        n = db.get_conn().execute(
            "SELECT COUNT(*) FROM qa_cache WHERE id=?", (rid,)
        ).fetchone()[0]
        assert n == 1, "KEEP verdict must not delete the row"
    finally:
        td.cleanup()


def test_run_curator_unclear_keeps():
    import curator
    import db

    td = tempfile.TemporaryDirectory()
    try:
        db.DB_PATH = Path(td.name) / "test.db"
        db._conn_local = threading.local()
        db.init_db()
        rid = db.insert_qa("q", "a", "m")
        match = {"id": rid, "query": "q", "answer": "a"}

        async def fake_verdict(cached_q, cached_a):
            return "some rambling answer, not a clean verdict"

        verdict = asyncio.run(curator.run_curator(match, verdict_fn=fake_verdict))
        assert verdict == "KEEP"  # unclear → safe default
        n = db.get_conn().execute(
            "SELECT COUNT(*) FROM qa_cache WHERE id=?", (rid,)
        ).fetchone()[0]
        assert n == 1
    finally:
        td.cleanup()


# ── Runner ───────────────────────────────────────────────────────────

def main():
    tests = [
        test_parse_curator_verdict,
        test_build_curator_messages,
        test_delete_cache_entry,
        test_run_curator_evicts,
        test_run_curator_keeps,
        test_run_curator_unclear_keeps,
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
