"""Similarity matching using RapidFuzz."""
import logging
from rapidfuzz import fuzz
from config import SIMILARITY_THRESHOLD

_log = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "what", "when", "where", "which", "who", "how", "why",
    "i", "you", "we", "they", "he", "she", "it",
    "to", "for", "of", "in", "on", "at", "with", "by", "from",
    "and", "or", "but", "if", "so", "as", "than", "that", "this",
    "do", "does", "did", "can", "could", "will", "would", "should",
    "key", "please", "just", "tell", "me", "explain", "describe",
})


def _clean(text: str) -> str:
    return " ".join(
        w for w in text.lower().split() if w not in _STOP_WORDS
    )


def find_best_match(
    query: str,
    candidates: list[dict],
    threshold: int | None = None,
) -> dict | None:
    """Return the best matching cached entry, or None.

    Uses a weighted combination of token_set_ratio (0.6) and partial_ratio
    (0.4). Only scans candidates that pass a cheap pre-filter.

    Args:
        query: The incoming user query string.
        candidates: List of dicts with 'id', 'query', 'answer'.
        threshold: Override similarity threshold (defaults to config).

    Returns:
        The best matching dict if above threshold, else None.
    """
    if not candidates:
        return None

    clean_query = _clean(query)
    if not clean_query:
        return None

    min_score = threshold if threshold is not None else SIMILARITY_THRESHOLD
    best = None
    best_score = 0
    query_tokens = set(clean_query.split())

    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        cached_query = entry.get("query", "")
        if not cached_query:
            continue

        # Cheap pre-filter: must share at least 1 token
        cached_tokens = set(_clean(cached_query).split())
        if not query_tokens & cached_tokens:
            continue

        ts2 = fuzz.token_set_ratio(clean_query, _clean(cached_query))
        pr = fuzz.partial_ratio(clean_query, _clean(cached_query))
        score = ts2 * 0.6 + pr * 0.4

        if score > best_score:
            best_score = score
            best = entry

    if best and best_score >= min_score:
        _log.debug("Cache match score=%.1f query=%s", best_score, query[:60])
        return best

    return None
