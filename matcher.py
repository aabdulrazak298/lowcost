"""Semantic cache matching — local transformer embeddings (rag-kit approach).

Uses all-MiniLM-L6-v2 (384-dim, 80MB) for semantic scoring of cache candidates.
No API calls, no GPU needed. Fully offline after first model download.

Pipeline: FTS5 pre-filter → embedding cosine → hybrid score with FTS5 BM25
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from config import SIMILARITY_THRESHOLD
from db import search_candidates

logger = logging.getLogger(__name__)

# ── Embedding model (singleton, lazy-loaded) ──────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384
SEM_FLOOR = float(os.environ.get("LOWCOST_SEM_FLOOR", "0.25"))

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
        logger.warning("Embedding model unavailable — falling back to FTS5-only matching")
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
    """Min-max normalize to [0, 1]."""
    scores = np.asarray(scores, dtype="float64")
    lo, hi = scores.min(), scores.max()
    if hi <= lo:
        return np.where(scores > 0, 1.0, 0.0).astype(np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def _fts5_normalize(raw_scores: np.ndarray) -> np.ndarray:
    """Normalize FTS5 BM25 scores to [0, 1]."""
    if raw_scores.size == 0:
        return np.array([], dtype=np.float32)
    return np.clip(raw_scores / raw_scores.max(), 0, 1).astype(np.float32)


def _hybrid_score(
    sem_scores: np.ndarray,
    fts5_scores: np.ndarray,
    sem_weight: float = 0.7,
    fts5_weight: float = 0.3,
) -> np.ndarray:
    """Blend semantic + FTS5 scores (rag-kit style)."""
    if sem_scores.size == 0:
        return _minmax_normalize(fts5_scores)

    sem_norm = _minmax_normalize(sem_scores)
    fts5_norm = _minmax_normalize(fts5_scores)
    return (sem_weight * sem_norm + fts5_weight * fts5_norm).astype(np.float32)


# ── Main lookup ──────────────────────────────────────────────


async def smart_cache_lookup(match_query: str) -> dict | None:
    """Semantic cache lookup using local transformer embeddings.

    1. FTS5 pre-filter → top N candidates
    2. Embed query + candidate questions → cosine similarity
    3. Hybrid score = 0.7 × semantic + 0.3 × FTS5 BM25
    4. Return best match above threshold (or None)
    """
    threshold = SIMILARITY_THRESHOLD / 100.0  # config is 0-100, we need 0-1

    # FTS5 pre-filter
    candidates = search_candidates(match_query, limit=20)
    if not candidates:
        return None

    # Try embedding-based scoring
    if not _model_failed:
        try:
            queries = [c.get("query", "") for c in candidates]
            query_vec = _embed([match_query])
            candidate_vecs = _embed(queries)

            if query_vec.size > 0 and candidate_vecs.size > 0:
                sem_scores = _cosine_scores(query_vec, candidate_vecs)

                # Normalize FTS5 BM25 scores
                # sqlite FTS5 rank is negative (lower=better), flip it
                fts5_raw = np.array([
                    -abs(float(c.get("rank", 0) or 0)) for c in candidates
                ], dtype=np.float32)
                fts5_norm = _minmax_normalize(fts5_raw)

                # Hybrid scoring
                hybrid = _hybrid_score(sem_scores, fts5_norm)

                best_idx = int(np.argmax(hybrid))
                best_score = float(hybrid[best_idx])

                if best_score >= threshold:
                    logger.debug(
                        "Cache HIT: score=%.3f (sem=%.3f fts5=%.3f) query=%s",
                        best_score,
                        float(sem_scores[best_idx]),
                        float(fts5_norm[best_idx]),
                        match_query[:60],
                    )
                    return candidates[best_idx]

                logger.debug(
                    "Cache MISS: best=%.3f below threshold=%.3f",
                    best_score, threshold,
                )
                return None

        except Exception:
            logger.exception("Embedding scoring failed, falling back to FTS5-only")

    # Fallback: FTS5-only (when model not available or scoring fails)
    # Just take the top FTS5 result with normalized score
    if candidates:
        fts5_raw = np.array([
            -abs(float(c.get("rank", 0) or 0)) for c in candidates
        ], dtype=np.float32)
        fts5_norm = _minmax_normalize(fts5_raw)
        best_idx = int(np.argmax(fts5_norm))
        best_score = float(fts5_norm[best_idx])

        if best_score >= threshold:
            logger.debug("Cache HIT (FTS5 fallback): score=%.3f", best_score)
            return candidates[best_idx]

    return None
