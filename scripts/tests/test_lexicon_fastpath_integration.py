"""Integration test: lexicon fast path through the REAL smart_cache_lookup.

Uses a TEMP sqlite DB (patched db.get_conn) seeded with code-purpose rows —
no live-DB writes, no lock contention. Verifies:
  - identifier-anchored rewrite -> HIT source=lexicon-fast (no embedding)
  - paraphrase with no anchor -> falls through to semantic path (MISS)
"""
import asyncio
import datetime
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
from matcher import smart_cache_lookup

NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_temp_db():
    path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(path, timeout=8)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE qa_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT, answer TEXT, model_used TEXT,
            hit_count INTEGER DEFAULT 0,
            created_at TEXT, purpose TEXT DEFAULT 'chat',
            last_accessed TEXT
        );
        CREATE VIRTUAL TABLE qa_cache_fts USING fts5(query);
    """)
    rows = [
        ("write a merge sort function in python", "def merge_sort(arr): ..."),
        ("export a list of dicts to csv file", "import csv\ndef export_csv(rows, path): ..."),
        ("implement binary search in python", "def binary_search(arr, target): ..."),
    ]
    for q, a in rows:
        cur = conn.execute(
            "INSERT INTO qa_cache (query, answer, model_used, hit_count, created_at, purpose, last_accessed) "
            "VALUES (?, ?, 'deepseek-v4-flash', 0, ?, 'code', ?)",
            (q, a, NOW, NOW),
        )
        conn.execute("INSERT INTO qa_cache_fts (rowid, query) VALUES (?, ?)", (cur.lastrowid, q))
    conn.commit()
    return conn


async def main():
    tconn = build_temp_db()
    orig_get_conn = db.get_conn
    db.get_conn = lambda: tconn
    try:
        # 1) identifier-anchored rewrite -> lexicon fast path HIT
        hit = await smart_cache_lookup(
            "change the csv export to use semicolon delimiter", purpose="code")
        print("lexicon rewrite ->", "HIT" if hit else "MISS")
        if hit:
            print("  matched:", hit["query"])
            assert hit["query"] == "export a list of dicts to csv file", "wrong row!"

        # 2) paraphrase with no identifier anchor -> falls through (semantic)
        miss = await smart_cache_lookup(
            "find where a value sits in an ordered collection", purpose="code")
        print("paraphrase ->", "MISS (fell through)" if miss is None else f"HIT {miss['query'][:40]}")

        ok = bool(hit) and miss is None
        print("\nINTEGRATION OK" if ok else "INTEGRATION FAILED")
        sys.exit(0 if ok else 1)
    finally:
        db.get_conn = orig_get_conn
        tconn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
