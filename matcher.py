"""Similarity matching using RapidFuzz."""
from rapidfuzz import fuzz
from config import SIMILARITY_THRESHOLD

# Common English stop words that add noise to similarity matching
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
    """Lowercase and remove stop words."""
    return " ".join(
        w for w in text.lower().split() if w not in _STOP_WORDS
    )


def find_best_match(query: str, candidates: list[dict]) -> dict | None:
    """Return the best matching cached entry, or None.

    Uses a weighted combination of token_set_ratio (0.6) and partial_ratio
    (0.4). Token set handles subset/overlap well; partial catches substrings.
    Weighting favors token_set to avoid false positives from very short cached
    queries matching on a single shared word (e.g. "PLC").

    The cheap model's IRRELEVANT check is the safety net for any false-positive
    matches that slip through.

    Args:
        query: The incoming user query string.
        candidates: List of dicts from get_all_queries(), each with 'id', 'query', 'answer'.

    Returns:
        The best matching dict if above threshold, else None.
    """
    if not candidates:
        return None

    clean_query = _clean(query)
    if not clean_query:
        return None

    best = None
    best_score = 0

    for entry in candidates:
        clean_cached = _clean(entry["query"])
        if not clean_cached:
            continue
        ts2 = fuzz.token_set_ratio(clean_query, clean_cached)
        pr = fuzz.partial_ratio(clean_query, clean_cached)
        # Weighted: token_set penalizes size mismatch, partial catches substrings
        score = ts2 * 0.6 + pr * 0.4
        if score > best_score:
            best_score = score
            best = entry

    if best and best_score >= SIMILARITY_THRESHOLD:
        return best

    return None
