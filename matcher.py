"""Semantic cache matching — local transformer embeddings (rag-kit approach).

Uses all-MiniLM-L6-v2 (384-dim, 80MB) for semantic scoring of cache candidates.
No API calls, no GPU needed. Fully offline after first model download.

Pipeline: FTS5 pre-filter → embedding cosine → absolute-threshold gate.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from db import search_candidates, age_days

logger = logging.getLogger(__name__)

# ── Embedding model (singleton, lazy-loaded) ──────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384
SEM_THRESHOLD = float(os.environ.get("LCLLM_SEM_THRESHOLD", "0.45"))

_model: Any = None
_model_failed: bool = False
_model_tried: bool = False


def _load_model():
    """Load the embedding model (singleton, process-global)."""
    global _model, _model_failed, _model_tried
    if _model_tried:
        return _model
    _model_tried = True

    if os.environ.get("LOWCOST_NO_MODEL") == "1":
        _model_failed = True
        return None

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Loaded embedding model: %s", MODEL_NAME)
        return _model
    except Exception:
        _model_failed = True
        logger.warning("Embedding model unavailable — cache lookups will miss (no lexical fallback)")
        return None


def _embed(texts: list[str]) -> np.ndarray:
    """Embed texts → (n, 384) float32, L2-normalized."""
    model = _load_model()
    if model is None or not texts:
        return np.zeros((0, DIM), dtype=np.float32)

    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    ).astype(np.float32)


def _cosine_scores(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity. Both must be L2-normalized. Returns (n,) float32."""
    if chunk_vecs.size == 0:
        return np.array([], dtype=np.float32)
    return np.clip(chunk_vecs @ query_vec.T, 0, None).flatten().astype(np.float32)


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1].

    ONLY for the FTS5 BM25 term — BM25 is unbounded (negative rank values), so
    it must be rescaled relative to the candidate batch to be comparable with
    the 0..1 cosine. NEVER apply this to the cosine itself: cosine is already
    an absolute relevance measure, and rescaling it relative to the batch is
    what caused the 2026-08-14 false-hit bug (0.077 → 1.0).
    """
    scores = np.asarray(scores, dtype="float64")
    lo, hi = scores.min(), scores.max()
    if hi <= lo:
        return np.where(scores > 0, 1.0, 0.0).astype(np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def _parse_ts(created_at: str | None):
    """Parse an SQLite UTC timestamp; unparseable → now (age 0, neutral)."""
    if not created_at:
        return datetime.now(timezone.utc)
    try:
        return datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _score_and_pick(
    sem_scores: np.ndarray,
    fts5_raw: np.ndarray,
    created_ats: list[str | None],
    threshold: float,
) -> tuple[int, np.ndarray]:
    """Hybrid ranking + pick: 0.70×sem + 0.25×bm25 + 0.05×recency.

    sem_scores: RAW cosine (absolute, this is the gate). fts5_raw: negated
    BM25 ranks (higher = better). created_ats: candidate created_at strings
    (UTC) — the recency term (newest → 1.0, min-max over the batch, same
    treatment as BM25). The gate is applied LAST: sub-threshold candidates can
    never win, no matter how new or lexically strong they are. The recency
    weight is deliberately small — it breaks near-ties only.

    Returns (best_idx, hybrid_scores).
    """
    fts5_norm = _minmax_normalize(fts5_raw)
    ages = np.array(
        [(datetime.now(timezone.utc) - _parse_ts(c)).total_seconds() for c in created_ats],
        dtype="float64",
    )
    if ages.size and ages.max() > ages.min():
        date_norm = 1.0 - (ages - ages.min()) / (ages.max() - ages.min())
    else:
        date_norm = np.ones_like(ages)  # tie / single candidate → neutral
    hybrid = 0.70 * sem_scores + 0.25 * fts5_norm + 0.05 * date_norm
    hybrid[sem_scores < threshold] = -np.inf
    return int(np.argmax(hybrid)), hybrid


# ── Main lookup ──────────────────────────────────────────────


async def smart_cache_lookup(match_query: str, purpose: str = "chat") -> dict | None:
    """Semantic cache lookup using local transformer embeddings.

    1. FTS5 pre-filter → top N candidates (lexical pre-filter)
    2. Embed query + candidates → raw cosine similarity (absolute relevance)
    3. Gate: the best raw cosine must clear SEM_THRESHOLD
    4. Rank survivors by hybrid = 0.7×raw_sem + 0.3×minmax(FTS5 BM25)

    The weighted blend is intentional (ported from rag-kit hybrid retrieval):
    semantic (0.7) is the dominant signal, lexical (0.3) is a tiebreaker. The
    semantic term is the RAW cosine (absolute, 0..1) — it is the GATE. The
    FTS5 term is min-max normalized (BM25 is unbounded) and used only to rank
    among candidates that already cleared the semantic gate. It can never
    admit an unrelated candidate: the 2026-08-14 bug was min-maxing the
    cosine itself (0.077 → 1.0 → false hit).
    """
    # FTS5 pre-filter (lexical) — limits how many candidates we embed
    candidates = search_candidates(match_query, limit=20, purpose=purpose)
    if not candidates:
        logger.info(
            "cache verdict=MISS source=semantic candidates=0 query=%r", match_query[:80],
        )
        return None

    # Without an embedding model we cannot do semantic matching. Return a
    # miss rather than fall back to lexical-only matching — lexical overlap
    # on stop words means nothing and re-introduces false hits.
    if _model_failed:
        return None

    try:
        queries = [c.get("query", "") for c in candidates]
        query_vec = _embed([match_query])
        candidate_vecs = _embed(queries)

        if query_vec.size == 0 or candidate_vecs.size == 0:
            return None

        sem_scores = _cosine_scores(query_vec, candidate_vecs)  # RAW, absolute

        # Absolute relevance gate — on the RAW cosine, never a normalized one.
        if float(sem_scores.max()) < SEM_THRESHOLD:
            logger.info(
                "cache verdict=MISS source=semantic best_cosine=%.3f threshold=%.2f "
                "candidates=%d query=%r",
                float(sem_scores.max()), SEM_THRESHOLD, len(candidates), match_query[:80],
            )
            return None

        # Hybrid ranking (0.70 sem + 0.25 bm25 + 0.05 recency — see
        # _score_and_pick). FTS5 bm25 rank is negative (more negative = better
        # match): negate so higher = better, then min-max to [0,1]. (NOTE: the
        # old code used -abs(rank), a no-op on negatives that left the FTS5
        # term useless. The recency term is batch-relative too and only breaks
        # near-ties — the gate below stays absolute.)
        fts5_raw = np.array([
            -float(c.get("rank", 0) or 0) for c in candidates
        ], dtype=np.float32)
        created = [c.get("created_at") for c in candidates]

        best_idx, hybrid = _score_and_pick(sem_scores, fts5_raw, created, SEM_THRESHOLD)
        logger.info(
            "cache verdict=HIT source=semantic cosine=%.3f hybrid=%.3f candidates=%d "
            "age_days=%s query=%r matched=%r",
            float(sem_scores[best_idx]), float(hybrid[best_idx]), len(candidates),
            age_days(created[best_idx]),
            match_query[:80], candidates[best_idx].get("query", "")[:80],
        )
        return candidates[best_idx]

    except Exception:
        logger.exception("Embedding scoring failed — treating as cache miss")
        return None
