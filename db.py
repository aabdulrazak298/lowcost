"""SQLite cache for Q&A pairs + conversation memory."""
import sqlite3
from datetime import datetime, timedelta, timezone
from config import CACHE_MAX_ENTRIES, CACHE_TTL_DAYS, DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_conn() as conn:
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
            pass  # column already exists
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_qa_created
            ON qa_cache(created_at DESC)
        """)
        _init_conversations_table(conn)
        _init_stats_table(conn)


def _init_conversations_table(conn: sqlite3.Connection) -> None:
    """Create or migrate the conversations table with sub-second timestamps."""
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

    # Check if migration needed (integer timestamps → sub-second)
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
    with get_conn() as conn:
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

        return rid


def get_all_queries() -> list[dict]:
    """Return all non-expired cached queries, newest first."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, query, answer, model_used, hit_count, created_at "
            "FROM qa_cache WHERE created_at >= ? "
            "ORDER BY created_at DESC",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]


def increment_hit_count(cache_id: int) -> None:
    """Increment the hit count for a cached entry."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE qa_cache SET hit_count = hit_count + 1 WHERE id = ?",
            (cache_id,),
        )


def get_cache_stats() -> dict:
    """Return cache statistics."""
    with get_conn() as conn:
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
        "top_reused": [{"query": r["query"][:80], "hits": r["hit_count"]} for r in top],
    }


# ── Conversation memory ──────────────────────────────────────────


def save_message(user_id: int, role: str, content: str, created_at: float | None = None) -> None:
    """Append a message to the user's conversation history."""
    with get_conn() as conn:
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


def get_history(user_id: int, limit: int = 30) -> list[dict]:
    """Return recent messages for a user, oldest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at "
            "FROM conversations WHERE user_id = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_message_count(user_id: int) -> int:
    """Return total message count for a user."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row[0]


def delete_messages(user_id: int, message_ids: list[int]) -> None:
    """Delete specific messages by ID."""
    if not message_ids:
        return
    with get_conn() as conn:
        placeholders = ",".join("?" * len(message_ids))
        conn.execute(
            f"DELETE FROM conversations WHERE user_id = ? AND id IN ({placeholders})",
            (user_id, *message_ids),
        )


def build_history_string(messages: list[dict]) -> str:
    """Convert message rows into a chat_history string matching FlaskChat format."""
    lines = []
    for m in messages:
        if m["role"] == "user":
            lines.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"Assistant: {m['content']}")
    return "\n".join(lines)


def clear_user_history(user_id: int) -> int:
    """Delete all messages for a user. Returns count of deleted rows."""
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE user_id = ?", (user_id,)
        )
        return cur.rowcount


# ── Stats persistence ──────────────────────────────────────────


def _init_stats_table(conn: sqlite3.Connection) -> None:
    """Create the stats snapshot table if it doesn't exist."""
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
    # Ensure exactly one row exists
    conn.execute("""
        INSERT OR IGNORE INTO stats_snapshot (id) VALUES (1)
    """)


def save_stats(stats: dict) -> None:
    """Persist in-memory stats to DB."""
    with get_conn() as conn:
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


def load_stats() -> dict:
    """Load persisted stats from DB. Returns defaults if no snapshot exists."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM stats_snapshot WHERE id = 1"
        ).fetchone()
    if row is None:
        return {}
    return dict(row)
