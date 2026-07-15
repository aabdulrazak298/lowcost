"""In-memory usage statistics with SQLite persistence."""
import threading
from datetime import datetime, timezone


_lock = threading.Lock()
_started_at = datetime.now(timezone.utc).isoformat()

_stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "irrelevant_escalations": 0,
    "expensive_calls": 0,
    "cheap_calls": 0,
    "tool_calls_total": 0,
    "models": {},
}

_save_count = 0


def init_from_db() -> None:
    """Restore stats from previous session."""
    from db import load_stats
    saved = load_stats()
    if not saved:
        return
    with _lock:
        for key in ("total_requests", "cache_hits", "cache_misses",
                     "irrelevant_escalations", "expensive_calls",
                     "cheap_calls", "tool_calls_total"):
            if key in saved:
                _stats[key] = saved[key]


def record_request(hit: bool, model: str, tool_calls: int = 0) -> None:
    """Record a processed request. Persists every 10 requests."""
    global _save_count
    with _lock:
        _stats["total_requests"] += 1
        if hit:
            _stats["cache_hits"] += 1
            _stats["cheap_calls"] += 1
        else:
            _stats["cache_misses"] += 1
            _stats["expensive_calls"] += 1
        if model == "irrelevant-escalated":
            _stats["irrelevant_escalations"] += 1
        _stats["tool_calls_total"] += tool_calls
        _stats["models"][model] = _stats["models"].get(model, 0) + 1
        _save_count += 1

    # Persist every 10 requests to survive crashes between restarts
    if _save_count % 10 == 0:
        _flush_db()


def _flush_db() -> None:
    """Write current stats to SQLite."""
    from db import save_stats
    with _lock:
        data = dict(_stats)
    save_stats(data)


def flush_to_db() -> None:
    """Force a save (call on shutdown)."""
    _flush_db()


def get_stats() -> dict:
    """Return current statistics snapshot."""
    with _lock:
        total = _stats["total_requests"]
        hits = _stats["cache_hits"]
        irrelevant = _stats["irrelevant_escalations"]
        return {
            "uptime_started": _started_at,
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": _stats["cache_misses"],
            "hit_rate_pct": round(hits / total * 100, 1) if total > 0 else 0,
            "expensive_calls": _stats["expensive_calls"],
            "cheap_calls": _stats["cheap_calls"],
            "irrelevant_escalations": irrelevant,
            "irrelevant_rate_pct": round(irrelevant / total * 100, 1) if total > 0 else 0,
            "tool_calls_total": _stats["tool_calls_total"],
            "models": _stats["models"],
        }
