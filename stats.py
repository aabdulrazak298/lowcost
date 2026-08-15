"""In-memory usage statistics with SQLite persistence.

Maintains two independent stat buckets: chat (`_stats`) and code (`_code_stats`).
The code path (judge/router) records with `purpose="code"` so its metrics are
tracked separately from general chat.
"""
import threading
from datetime import datetime, timezone


_lock = threading.Lock()
_started_at = datetime.now(timezone.utc).isoformat()

_STATS_KEYS = (
    "total_requests", "cache_hits", "cache_misses", "irrelevant_escalations",
    "expensive_calls", "cheap_calls", "tool_calls_total",
    "prompt_tokens", "completion_tokens", "total_tokens",
)


def _make_stats() -> dict:
    return {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "irrelevant_escalations": 0,
        "expensive_calls": 0,
        "cheap_calls": 0,
        "tool_calls_total": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "models": {},
    }


_stats = _make_stats()       # chat
_code_stats = _make_stats()  # code

_save_count = 0


def init_from_db() -> None:
    """Restore stats from previous session."""
    from db import load_stats
    saved = load_stats(purpose="chat")
    if saved:
        with _lock:
            for key in _STATS_KEYS:
                if key in saved:
                    _stats[key] = saved[key]
    code_saved = load_stats(purpose="code")
    if code_saved:
        with _lock:
            for key in _STATS_KEYS:
                if key in code_saved:
                    _code_stats[key] = code_saved[key]


def record_request(
    hit: bool,
    model: str,
    tool_calls: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    purpose: str = "chat",
) -> None:
    """Record a processed request. Persists every 10 requests."""
    global _save_count
    target = _code_stats if purpose == "code" else _stats
    with _lock:
        target["total_requests"] += 1
        if hit:
            target["cache_hits"] += 1
            target["cheap_calls"] += 1
        else:
            target["cache_misses"] += 1
            target["expensive_calls"] += 1
        if model == "irrelevant-escalated":
            target["irrelevant_escalations"] += 1
        target["tool_calls_total"] += tool_calls
        target["prompt_tokens"] += prompt_tokens
        target["completion_tokens"] += completion_tokens
        target["total_tokens"] += prompt_tokens + completion_tokens
        target["models"][model] = target["models"].get(model, 0) + 1
        _save_count += 1

    if _save_count % 10 == 0:
        _flush_db()


def _flush_db() -> None:
    """Write current stats to SQLite."""
    from db import save_stats
    with _lock:
        chat_data = dict(_stats)
        code_data = dict(_code_stats)
    save_stats(chat_data, purpose="chat")
    save_stats(code_data, purpose="code")


def flush_to_db() -> None:
    """Force a save (call on shutdown)."""
    _flush_db()


def _snapshot(target: dict) -> dict:
    total = target["total_requests"]
    hits = target["cache_hits"]
    irrelevant = target["irrelevant_escalations"]
    return {
        "uptime_started": _started_at,
        "total_requests": total,
        "cache_hits": hits,
        "cache_misses": target["cache_misses"],
        "hit_rate_pct": round(hits / total * 100, 1) if total > 0 else 0,
        "expensive_calls": target["expensive_calls"],
        "cheap_calls": target["cheap_calls"],
        "irrelevant_escalations": irrelevant,
        "irrelevant_rate_pct": round(irrelevant / total * 100, 1) if total > 0 else 0,
        "tool_calls_total": target["tool_calls_total"],
        "tokens": {
            "prompt": target["prompt_tokens"],
            "completion": target["completion_tokens"],
            "total": target["total_tokens"],
        },
        "models": target["models"],
    }


def get_stats() -> dict:
    """Return current chat statistics snapshot."""
    with _lock:
        return _snapshot(_stats)


def get_code_stats() -> dict:
    """Return current code-path statistics snapshot."""
    with _lock:
        return _snapshot(_code_stats)
