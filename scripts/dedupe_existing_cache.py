#!/usr/bin/env python3
"""One-time cleanup: collapse duplicate qa_cache rows in the LIVE cache.db.

Observed 2026-08-18: identical "summaries <youtube-url>" rows (151/152,
154/155) both hit=0 — a cache hit whose cheap step failed re-inserted.

Rule: for each (normalized query, purpose) group with >1 row, keep the row
with the most hits (tie → newest created_at), delete the rest. FTS stays in
sync via the AFTER DELETE trigger. Runs safely against the live service
(WAL mode + busy_timeout) — but do it right before a restart anyway.

Usage:
    .venv/bin/python scripts/dedupe_existing_cache.py [--dry-run]
"""
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "cache.db"
DRY = "--dry-run" in sys.argv


def norm(q: str) -> str:
    return " ".join((q or "").lower().split())


def main():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=20000")
    c.execute("PRAGMA journal_mode=WAL")
    rows = c.execute(
        "SELECT id, query, answer, model_used, hit_count, created_at, purpose "
        "FROM qa_cache ORDER BY created_at"
    ).fetchall()

    groups = defaultdict(list)
    for r in rows:
        groups[(norm(r["query"]), r["purpose"])].append(r)

    to_delete: list[int] = []
    for (q, purpose), members in sorted(groups.items()):
        if len(members) <= 1:
            continue
        keep = max(members, key=lambda r: (r["hit_count"], r["created_at"]))
        for r in members:
            if r["id"] != keep["id"]:
                to_delete.append(r["id"])
                print(
                    f"  dup id={r['id']} (hits={r['hit_count']}) → keep id={keep['id']} "
                    f"(hits={keep['hit_count']}) | {q[:60]!r}"
                )

    print(f"\n{len(to_delete)} duplicate row(s) to delete, {len(rows) - len(to_delete)} will remain")
    if DRY:
        print("DRY RUN — no changes made")
        return

    # video-ID dedupe: same video, different wording (e.g. summaries vs summarise)
    vid_re = re.compile(r"(?:youtube\.com/(?:watch\?v=|shorts/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})")
    by_vid = defaultdict(list)
    for r in rows:
        if r["id"] in to_delete:
            continue
        for v in set(vid_re.findall(r["query"])):
            by_vid[v].append(r)
    for v, members in by_vid.items():
        if len(members) <= 1:
            continue
        keep = max(members, key=lambda r: (r["hit_count"], r["created_at"]))
        for r in members:
            if r["id"] != keep["id"] and r["id"] not in to_delete:
                to_delete.append(r["id"])
                print(
                    f"  vid-dup id={r['id']} (hits={r['hit_count']}) → keep id={keep['id']} "
                    f"| video {v} | {norm(r['query'])[:60]!r}"
                )

    if to_delete:
        cur = c.executemany("DELETE FROM qa_cache WHERE id = ?", [(i,) for i in to_delete])
        c.commit()
        print(f"deleted {cur.rowcount} rows")
    else:
        print("nothing to delete")

    after = c.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
    fts = c.execute("SELECT COUNT(*) FROM qa_cache_fts").fetchone()[0]
    print(f"qa_cache now: {after} rows, FTS: {fts} rows ({'OK' if after == fts else 'MISMATCH!'})")
    c.close()


if __name__ == "__main__":
    main()
