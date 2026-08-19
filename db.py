"""SQLite cache for Q&A pairs + conversation memory."""
import asyncio
import logging
import re
import sqlite3
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from config import CACHE_MAX_ENTRIES, CACHE_TTL_DAYS, DB_PATH

logger = logging.getLogger(__name__)

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
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            last_accessed TEXT  NOT NULL DEFAULT (datetime('now'))
        )
    """)
    # Migration: add hit_count column if upgrading from older schema
    try:
        conn.execute("ALTER TABLE qa_cache ADD COLUMN hit_count INTEGER NOT NULL DEFAULT 0")
    except Exception:
        pass
    # Migration: add purpose column (chat | code) for cache separation
    try:
        conn.execute("ALTER TABLE qa_cache ADD COLUMN purpose TEXT NOT NULL DEFAULT 'chat'")
    except Exception:
        pass
    # Migration: add last_accessed column (sliding TTL — refreshed on hit).
    # SQLite ALTER ADD COLUMN can't carry a datetime('now') default, so add a
    # nullable column and backfill existing rows from created_at.
    try:
        conn.execute("ALTER TABLE qa_cache ADD COLUMN last_accessed TEXT")
    except Exception:
        pass
    conn.execute("UPDATE qa_cache SET last_accessed = created_at WHERE last_accessed IS NULL")
    conn.commit()

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_qa_created
        ON qa_cache(created_at DESC)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_qa_last_accessed
        ON qa_cache(last_accessed)
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
    _init_overrides_table(conn)
    _init_app_settings_table(conn)


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


def insert_qa(query: str, answer: str, model_used: str, purpose: str = "chat") -> int:
    """Insert a Q&A pair. Evicts least-recently-accessed if over max. Returns the new row ID."""
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO qa_cache (query, answer, model_used, purpose, last_accessed) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (query, answer, model_used, purpose),
    )
    rid = cur.lastrowid

    count = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
    if count > CACHE_MAX_ENTRIES:
        excess = count - CACHE_MAX_ENTRIES
        conn.execute(
            "DELETE FROM qa_cache WHERE id IN "
            "(SELECT id FROM qa_cache ORDER BY last_accessed ASC LIMIT ?)",
            (excess,),
        )
    conn.commit()

    # Hot cache insert is best-effort — skip if lock contention
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        _hot_cache[str(rid)] = {"id": rid, "query": query, "answer": answer, "model_used": model_used, "hit_count": 0, "purpose": purpose, "created_at": now}
        if len(_hot_cache) >= HOT_CACHE_MAX:
            _hot_cache.popitem(last=False)
    except Exception:
        pass
    return rid


_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def _normalize_query(query: str) -> str:
    """Canonical form for duplicate detection: lowercase, whitespace-collapsed."""
    return " ".join((query or "").lower().split())


def _video_ids_in(query: str) -> list[str]:
    """Unique YouTube video IDs referenced in a query."""
    return list(dict.fromkeys(_VIDEO_ID_RE.findall(query or "")))


def _refresh_row(rid: int, query: str, answer: str, model_used: str, conn) -> int:
    """Update an existing cache row in place (FTS stays in sync via trigger)."""
    conn.execute(
        "UPDATE qa_cache SET query=?, answer=?, model_used=?, "
        "created_at=datetime('now'), last_accessed=datetime('now') WHERE id=?",
        (query, answer, model_used, rid),
    )
    conn.commit()
    return rid


def upsert_qa(query: str, answer: str, model_used: str, purpose: str = "chat") -> int:
    """Insert a Q&A pair, deduping against an existing active row.

    One row per distinct question (write-side dedupe):
      1. same normalized query  -> UPDATE in place (refreshes answer + timestamps)
      2. same YouTube video ID  -> UPDATE in place (catches wording variants
         like "summaries <url>" vs "summarise <url>")
      3. otherwise              -> plain INSERT

    Fixes the 2026-08-18 duplicate-rows bug: a cache hit whose cheap step
    failed (429 window) escalated to the expensive path and inserted a fresh
    copy of an identical query. Repeated asks now refresh instead of stacking.
    Returns the row id (existing or new).
    """
    conn = get_conn()
    norm = _normalize_query(query)
    if norm:
        row = conn.execute(
            "SELECT id FROM qa_cache WHERE lower(trim(query)) = ? AND purpose = ? "
            "ORDER BY id DESC LIMIT 1",
            (norm, purpose),
        ).fetchone()
        if row:
            logger.info("upsert dedupe: same query → UPDATE row %d", row["id"])
            return _refresh_row(row["id"], query, answer, model_used, conn)
    for vid in _video_ids_in(query):
        existing = lookup_by_video_id(vid, purpose)
        if existing:
            logger.info(
                "upsert dedupe: same video id %s → UPDATE row %d", vid, existing["id"],
            )
            return _refresh_row(existing["id"], query, answer, model_used, conn)
    return insert_qa(query, answer, model_used, purpose)


_FTS_STOP_WORDS = frozenset({
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their",
    # question words
    "what", "which", "who", "whom", "when", "where", "why", "how",
    # copula / basic auxiliaries
    "is", "are", "am", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    # basic prepositions
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "into",
    "from", "during", "between", "through", "over", "under",
    # conversational fillers (heavy in this cache)
    "tell", "more", "please", "get", "got", "want", "need", "just", "like",
    "there", "here", "know", "think",
})


def age_days(created_at: str | None) -> float | None:
    """Age of a cache entry in days (UTC created_at). None when unparseable."""
    if not created_at:
        return None
    try:
        ts = datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return None


def search_candidates(query: str, limit: int = 100, purpose: str = "chat") -> list[dict]:
    """FTS5 pre-filter: find top N candidates by text relevance.

    Only these candidates are scored by the embedding model. Stop words are
    dropped so the OR query matches on content words — otherwise "what is the
    butlerian jihad in the dune universe" would match every entry containing
    "the"/"is"/"in" and surface unrelated candidates.
    """
    conn = get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()

    # Sanitize FTS5 query - escape special chars, wrap terms for better matching
    safe = query.replace('"', '""')
    terms = [
        f'"{t}"'
        for t in safe.split()
        if len(t) > 1 and t.lower() not in _FTS_STOP_WORDS
    ]

    if not terms:
        return []

    fts_query = " OR ".join(terms[:20])  # max 20 terms

    rows = conn.execute(
        "SELECT qa.id, qa.query, qa.answer, qa.model_used, qa.hit_count, qa.created_at, "
        "       bm25(qa_cache_fts) AS rank "
        "FROM qa_cache_fts fts "
        "JOIN qa_cache qa ON fts.rowid = qa.id "
        "WHERE qa_cache_fts MATCH ? AND qa.last_accessed >= ? AND qa.purpose = ? "
        "ORDER BY rank LIMIT ?",
        (fts_query, cutoff, purpose, limit),
    ).fetchall()

    return [dict(r) for r in rows]


def get_all_queries() -> list[dict]:
    """Return all non-expired cached queries (used when FTS returns no results)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, query, answer, model_used, hit_count, created_at "
        "FROM qa_cache WHERE last_accessed >= ? "
        "ORDER BY created_at DESC",
        (cutoff,),
    ).fetchall()
    return [dict(r) for r in rows]


def lookup_by_video_id(video_id: str, purpose: str = "chat") -> dict | None:
    """Return the most recent cached entry whose query contains this exact video ID.

    Used by the agentic cache search for same-video reuse: an exact 11-char
    YouTube ID is the true identity of a video, so a substring match is safe.
    """
    conn = get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()
    row = conn.execute(
        "SELECT id, query, answer, model_used, hit_count, created_at "
        "FROM qa_cache "
        "WHERE query LIKE ? AND last_accessed >= ? AND purpose = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (f"%{video_id}%", cutoff, purpose),
    ).fetchone()
    return dict(row) if row else None


def delete_cache_entry(cache_id: int) -> int:
    """Delete a cached Q&A row by id. FTS is kept in sync via the delete trigger.

    Returns the number of rows deleted (0 if the id no longer exists).
    """
    conn = get_conn()
    cur = conn.execute("DELETE FROM qa_cache WHERE id = ?", (cache_id,))
    conn.commit()
    return cur.rowcount


def increment_hit_count(cache_id: int) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE qa_cache SET hit_count = hit_count + 1, "
        "last_accessed = datetime('now') WHERE id = ?",
        (cache_id,),
    )
    conn.commit()
    _hot_cache_bump(cache_id)


def get_cache_stats() -> dict:
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()
    active = conn.execute(
        "SELECT COUNT(*) FROM qa_cache WHERE last_accessed >= ?", (cutoff,)
    ).fetchone()[0]
    expired = total - active
    top = conn.execute(
        "SELECT query, hit_count FROM qa_cache "
        "WHERE last_accessed >= ? "
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
        "top_queries": [{"query": r["query"][:80], "count": r["hit_count"]} for r in top],
    }


# -- Hot cache (LRU via OrderedDict) --------------------------------

def _hot_cache_bump(cache_id: int) -> None:
    for k in list(_hot_cache.keys()):
        v = _hot_cache[k]
        if v.get("id") == cache_id:
            v["hit_count"] = v.get("hit_count", 0) + 1
            _hot_cache.move_to_end(k)
            break


async def hot_cache_lookup(query: str, purpose: str = "chat") -> dict | None:
    async with _hot_cache_lock:
        key = f"{purpose}:{query}"
        entry = _hot_cache.get(key)
        if entry:
            _hot_cache.move_to_end(key)
        return entry


async def hot_cache_put(query: str, entry: dict, purpose: str = "chat") -> None:
    async with _hot_cache_lock:
        key = f"{purpose}:{query}"
        if len(_hot_cache) >= HOT_CACHE_MAX:
            _hot_cache.popitem(last=False)
        _hot_cache[key] = entry


async def cache_lookup(match_query: str, purpose: str = "chat") -> dict | None:
    """Unified cache lookup: hot cache → FTS5 → LLM smart match → fallback full scan.
    Uses LLM-based semantic matching instead of RapidFuzz string distance."""
    from matcher import smart_cache_lookup

    hot = await hot_cache_lookup(match_query, purpose)
    if hot:
        logger.info(
            "cache verdict=HIT source=hot age_days=%s query=%r",
            age_days(hot.get("created_at")), match_query[:80],
        )
        return hot

    match = await smart_cache_lookup(match_query, purpose)
    if match:
        await hot_cache_put(match_query, match, purpose)
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

def _init_overrides_table(conn: sqlite3.Connection) -> None:
    """Create model_overrides table for persisting /model choices."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_overrides (
            id                  INTEGER PRIMARY KEY CHECK (id = 1),
            cheap_override      TEXT,
            expensive_override  TEXT,
            cheap_fallback_override TEXT
        )
    """)
    conn.execute("INSERT OR IGNORE INTO model_overrides (id) VALUES (1)")
    _migrate_overrides_table(conn)


def _migrate_overrides_table(conn: sqlite3.Connection) -> None:
    """Add cheap_fallback_override column if upgrading from an older schema."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(model_overrides)").fetchall()}
    if "cheap_fallback_override" not in cols:
        try:
            conn.execute("ALTER TABLE model_overrides ADD COLUMN cheap_fallback_override TEXT")
            conn.commit()
        except Exception:
            pass


def ensure_overrides_schema() -> None:
    """Self-healing: make sure model_overrides has the fallback column.

    Called from config's load/save paths so overrides survive even when
    init_db hasn't run yet (e.g. standalone scripts that load config).
    """
    try:
        conn = get_conn()
        _migrate_overrides_table(conn)
    except Exception:
        pass


def _init_app_settings_table(conn: sqlite3.Connection) -> None:
    """Create app_settings table for persisting voice (/voice) choices."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            id            INTEGER PRIMARY KEY CHECK (id = 1),
            voice_enabled INTEGER NOT NULL DEFAULT 0,
            voice_engine  TEXT    NOT NULL DEFAULT 'edge'
        )
    """)
    conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")


# Table names for per-purpose stats snapshots (chat vs code).
_STATS_TABLES = {"chat": "stats_snapshot", "code": "stats_snapshot_code"}


def _init_stats_table(conn: sqlite3.Connection) -> None:
    for table in _STATS_TABLES.values():
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                total_requests          INTEGER NOT NULL DEFAULT 0,
                cache_hits              INTEGER NOT NULL DEFAULT 0,
                cache_misses            INTEGER NOT NULL DEFAULT 0,
                irrelevant_escalations  INTEGER NOT NULL DEFAULT 0,
                expensive_calls         INTEGER NOT NULL DEFAULT 0,
                cheap_calls             INTEGER NOT NULL DEFAULT 0,
                tool_calls_total        INTEGER NOT NULL DEFAULT 0,
                prompt_tokens           INTEGER NOT NULL DEFAULT 0,
                completion_tokens       INTEGER NOT NULL DEFAULT 0,
                total_tokens            INTEGER NOT NULL DEFAULT 0,
                updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute(f"INSERT OR IGNORE INTO {table} (id) VALUES (1)")
        _migrate_stats_table(conn, table)


def _migrate_stats_table(conn: sqlite3.Connection, table: str) -> None:
    """Add token columns if missing from older schema."""
    existing = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols = {r[1] for r in existing}
    for col in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")


def save_stats(stats: dict, purpose: str = "chat") -> None:
    table = _STATS_TABLES.get(purpose, "stats_snapshot")
    conn = get_conn()
    conn.execute(f"""
        UPDATE {table} SET
            total_requests = ?,
            cache_hits = ?,
            cache_misses = ?,
            irrelevant_escalations = ?,
            expensive_calls = ?,
            cheap_calls = ?,
            tool_calls_total = ?,
            prompt_tokens = ?,
            completion_tokens = ?,
            total_tokens = ?,
            updated_at = datetime('now')
        WHERE id = 1
    """, (
        stats.get("total_requests", 0),
        stats.get("cache_hits", 0),
        stats.get("cache_misses", 0),
        stats.get("irrelevant_escalations", 0),
        stats.get("expensive_calls", 0),
        stats.get("cheap_calls", 0),
        stats.get("tool_calls_total", 0),
        stats.get("prompt_tokens", 0),
        stats.get("completion_tokens", 0),
        stats.get("total_tokens", 0),
    ))
    conn.commit()


def load_stats(purpose: str = "chat") -> dict:
    table = _STATS_TABLES.get(purpose, "stats_snapshot")
    conn = get_conn()
    row = conn.execute(
        f"SELECT * FROM {table} WHERE id = 1"
    ).fetchone()
    if row is None:
        return {}
    return dict(row)
