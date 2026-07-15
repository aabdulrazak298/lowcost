"""SQLite cache for Q&A pairs + conversation memory."""
import asyncio
import sqlite3
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from config import CACHE_MAX_ENTRIES, CACHE_TTL_DAYS, DB_PATH

_conn_local = threading.local()
_hot_cache: OrderedDict[str, dict] = OrderedDict()
_hot_cache_lock = asyncio.Lock()
HOT_CACHE_MAX = 2000


def get_conn() -> sqlite3.Connection:
    """Return a persistent thread-local connection with WAL mode."""
    conn = getattr(_conn_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-65536")  # 64MB page cache
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
        conn.execute("PRAGMA busy_timeout=5000")
        _conn_local.conn = conn
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT    NOT NULL,
            answer      TEXT    NOT NULL,
            model_used  TEXT    NOT NULL,
            hit_count   INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    # Migration: add hit_count column if upgrading from older schema
    try:
        conn.execute("ALTER TABLE qa_cache ADD COLUMN hit_count INTEGER NOT NULL DEFAULT 0")
    except Exception:
        pass

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_qa_created
        ON qa_cache(created_at DESC)
    """)

    # FTS5 full-text index for fast candidate pre-filtering
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS qa_cache_fts USING fts5(
            query,
            content='qa_cache',
            content_rowid='id',
            tokenize='porter unicode61'
        )
    """)

    # Triggers to keep FTS index in sync
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS qa_cache_ai AFTER INSERT ON qa_cache BEGIN
            INSERT INTO qa_cache_fts(rowid, query) VALUES (new.id, new.query);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS qa_cache_ad AFTER DELETE ON qa_cache BEGIN
            INSERT INTO qa_cache_fts(qa_cache_fts, rowid, query) VALUES('delete', old.id, old.query);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS qa_cache_au AFTER UPDATE ON qa_cache BEGIN
            INSERT INTO qa_cache_fts(qa_cache_fts, rowid, query) VALUES('delete', old.id, old.query);
            INSERT INTO qa_cache_fts(rowid, query) VALUES (new.id, new.query);
        END
    """)

    _init_conversations_table(conn)
    _init_stats_table(conn)


def _init_conversations_table(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
    ).fetchone()

    if row is None:
        conn.execute("""
            CREATE TABLE conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                created_at  REAL    NOT NULL DEFAULT (unixepoch('subsec'))
            )
        """)
        conn.execute("""
            CREATE INDEX idx_conv_user ON conversations(user_id, created_at)
        """)
        return

    sample = conn.execute(
        "SELECT created_at FROM conversations LIMIT 1"
    ).fetchone()

    if sample and sample[0] == int(sample[0]):
        conn.execute("ALTER TABLE conversations RENAME TO conversations_old")
        conn.execute("""
            CREATE TABLE conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                created_at  REAL    NOT NULL DEFAULT (unixepoch('subsec'))
            )
        """)
        conn.execute("""
            INSERT INTO conversations (id, user_id, role, content, created_at)
            SELECT id, user_id, role, content, created_at FROM conversations_old
        """)
        conn.execute("DROP TABLE conversations_old")
        conn.execute("""
            CREATE INDEX idx_conv_user ON conversations(user_id, created_at)
        """)


def insert_qa(query: str, answer: str, model_used: str) -> int:
    """Insert a Q&A pair. Evicts oldest if over max. Returns the new row ID."""
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO qa_cache (query, answer, model_used) VALUES (?, ?, ?)",
        (query, answer, model_used),
    )
    rid = cur.lastrowid

    count = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
    if count > CACHE_MAX_ENTRIES:
        excess = count - CACHE_MAX_ENTRIES
        conn.execute(
            "DELETE FROM qa_cache WHERE id IN "
            "(SELECT id FROM qa_cache ORDER BY created_at ASC LIMIT ?)",
            (excess,),
        )
    conn.commit()

    # Hot cache insert is best-effort — skip if lock contention
    try:
        _hot_cache[str(rid)] = {"id": rid, "query": query, "answer": answer, "model_used": model_used, "hit_count": 0}
        if len(_hot_cache) >= HOT_CACHE_MAX:
            _hot_cache.popitem(last=False)
    except Exception:
        pass
    return rid


def search_candidates(query: str, limit: int = 100) -> list[dict]:
    """FTS5 pre-filter: find top N candidates by text relevance.
    Only these candidates will be fuzzy-matched by RapidFuzz."""
    conn = get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()

    # Sanitize FTS5 query - escape special chars, wrap terms for better matching
    safe = query.replace('"', '""')
    terms = [f'"{t}"' if len(t) > 2 else t for t in safe.split() if len(t) > 1]

    if not terms:
        return []

    fts_query = " OR ".join(terms[:20])  # max 20 terms

    rows = conn.execute(
        "SELECT qa.id, qa.query, qa.answer, qa.model_used, qa.hit_count "
        "FROM qa_cache_fts fts "
        "JOIN qa_cache qa ON fts.rowid = qa.id "
        "WHERE qa_cache_fts MATCH ? AND qa.created_at >= ? "
        "ORDER BY rank LIMIT ?",
        (fts_query, cutoff, limit),
    ).fetchall()

    return [dict(r) for r in rows]


def get_all_queries() -> list[dict]:
    """Return all non-expired cached queries (used when FTS returns no results)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, query, answer, model_used, hit_count, created_at "
        "FROM qa_cache WHERE created_at >= ? "
        "ORDER BY created_at DESC",
        (cutoff,),
    ).fetchall()
    return [dict(r) for r in rows]


def increment_hit_count(cache_id: int) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE qa_cache SET hit_count = hit_count + 1 WHERE id = ?",
        (cache_id,),
    )
    conn.commit()
    _hot_cache_bump(cache_id)


def get_cache_stats() -> dict:
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()
    active = conn.execute(
        "SELECT COUNT(*) FROM qa_cache WHERE created_at >= ?", (cutoff,)
    ).fetchone()[0]
    expired = total - active
    top = conn.execute(
        "SELECT query, hit_count FROM qa_cache "
        "WHERE created_at >= ? "
        "ORDER BY hit_count DESC LIMIT 5",
        (cutoff,),
    ).fetchall()

    return {
        "total_entries": total,
        "active": active,
        "expired": expired,
        "ttl_days": CACHE_TTL_DAYS,
        "max_entries": CACHE_MAX_ENTRIES,
        "hot_cache_size": len(_hot_cache),
        "top_reused": [{"query": r["query"][:80], "hits": r["hit_count"]} for r in top],
    }


# -- Hot cache (LRU via OrderedDict) --------------------------------

def _hot_cache_bump(cache_id: int) -> None:
    for k in list(_hot_cache.keys()):
        v = _hot_cache[k]
        if v.get("id") == cache_id:
            v["hit_count"] = v.get("hit_count", 0) + 1
            _hot_cache.move_to_end(k)
            break


async def hot_cache_lookup(query: str) -> dict | None:
    async with _hot_cache_lock:
        entry = _hot_cache.get(query)
        if entry:
            _hot_cache.move_to_end(query)
        return entry


async def hot_cache_put(query: str, entry: dict) -> None:
    async with _hot_cache_lock:
        if len(_hot_cache) >= HOT_CACHE_MAX:
            _hot_cache.popitem(last=False)
        _hot_cache[query] = entry


async def cache_lookup(match_query: str) -> dict | None:
    """Unified cache lookup: hot cache → FTS5 → RapidFuzz → fallback full scan.
    Used by both proxy.py and processor.py."""
    from matcher import find_best_match

    hot = await hot_cache_lookup(match_query)
    if hot:
        return hot

    candidates = search_candidates(match_query, limit=100)
    if candidates:
        match = find_best_match(match_query, candidates)
        if match:
            await hot_cache_put(match_query, match)
            return match

    all_entries = get_all_queries()
    if all_entries:
        match = find_best_match(match_query, all_entries)
        if match:
            await hot_cache_put(match_query, match)
            return match

    return None


# -- Conversation memory ------------------------------------------

def save_message(user_id: int, role: str, content: str, created_at: float | None = None) -> None:
    conn = get_conn()
    if created_at is not None:
        conn.execute(
            "INSERT INTO conversations (user_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, role, content, created_at),
        )
    else:
        conn.execute(
            "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
    conn.commit()


def get_history(user_id: int, limit: int = 30) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, role, content, created_at "
        "FROM conversations WHERE user_id = ? "
        "ORDER BY created_at ASC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_message_count(user_id: int) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return row[0]


def delete_messages(user_id: int, message_ids: list[int]) -> None:
    if not message_ids:
        return
    conn = get_conn()
    placeholders = ",".join("?" * len(message_ids))
    conn.execute(
        f"DELETE FROM conversations WHERE user_id = ? AND id IN ({placeholders})",
        (user_id, *message_ids),
    )
    conn.commit()


def build_history_string(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        if m["role"] == "user":
            lines.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"Assistant: {m['content']}")
    return "\n".join(lines)


def clear_user_history(user_id: int) -> int:
    conn = get_conn()
    cur = conn.execute(
        "DELETE FROM conversations WHERE user_id = ?", (user_id,)
    )
    conn.commit()
    return cur.rowcount


# -- Stats persistence ---------------------------------------------

def _init_stats_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stats_snapshot (
            id          INTEGER PRIMARY KEY CHECK (id = 1),
            total_requests          INTEGER NOT NULL DEFAULT 0,
            cache_hits              INTEGER NOT NULL DEFAULT 0,
            cache_misses            INTEGER NOT NULL DEFAULT 0,
            irrelevant_escalations  INTEGER NOT NULL DEFAULT 0,
            expensive_calls         INTEGER NOT NULL DEFAULT 0,
            cheap_calls             INTEGER NOT NULL DEFAULT 0,
            tool_calls_total        INTEGER NOT NULL DEFAULT 0,
            updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("INSERT OR IGNORE INTO stats_snapshot (id) VALUES (1)")


def save_stats(stats: dict) -> None:
    conn = get_conn()
    conn.execute("""
        UPDATE stats_snapshot SET
            total_requests = ?,
            cache_hits = ?,
            cache_misses = ?,
            irrelevant_escalations = ?,
            expensive_calls = ?,
            cheap_calls = ?,
            tool_calls_total = ?,
            updated_at = datetime('now')
        WHERE id = 1
    """, (
        stats["total_requests"],
        stats["cache_hits"],
        stats["cache_misses"],
        stats["irrelevant_escalations"],
        stats["expensive_calls"],
        stats["cheap_calls"],
        stats["tool_calls_total"],
    ))
    conn.commit()


def load_stats() -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM stats_snapshot WHERE id = 1"
    ).fetchone()
    if row is None:
        return {}
    return dict(row)
