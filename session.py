import asyncio
import hashlib
import time

_affinity: dict[str, tuple[str, float]] = {}
_lock = asyncio.Lock()


def session_id_from(messages: list[dict], header_value: str | None = None) -> str:
    if header_value and header_value.strip():
        return header_value.strip()[:32]
    first_user = ""
    for m in messages:
        if m.get("role") == "user":
            first_user = m.get("content", "")
            if isinstance(first_user, list):
                first_user = " ".join(
                    str(item.get("text", item.get("content", "")))
                    for item in first_user if isinstance(item, dict)
                )
            break
    if isinstance(first_user, str):
        return hashlib.sha256(first_user.encode()).hexdigest()[:32]
    return hashlib.sha256(str(first_user).encode()).hexdigest()[:32]


async def get_affinity(session_id: str, ttl: float) -> str | None:
    async with _lock:
        entry = _affinity.get(session_id)
        if entry is None:
            return None
        target, ts = entry
        if time.monotonic() - ts > ttl:
            del _affinity[session_id]
            return None
        return target


async def set_affinity(session_id: str, target: str) -> None:
    async with _lock:
        _affinity[session_id] = (target, time.monotonic())


async def evict_stale(ttl: float) -> None:
    now = time.monotonic()
    async with _lock:
        stale = [k for k, (_, ts) in _affinity.items() if now - ts > ttl]
        for k in stale:
            del _affinity[k]