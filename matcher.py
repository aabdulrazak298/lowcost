"""Semantic cache matching — local transformer embeddings (rag-kit approach).

Uses all-MiniLM-L6-v2 (384-dim, 80MB) for semantic scoring of cache candidates.
No API calls, no GPU needed. Fully offline after first model download.

Pipeline: FTS5 pre-filter → embedding cosine → absolute-threshold gate.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np

from db import search_candidates, age_days, is_ephemeral_query

logger = logging.getLogger(__name__)

# ── Era gate ────────────────────────────────────────────────
# MiniLM scores "top 10 baby names in the 1980s" ≈ "…in the 1990s" at ~0.84
# — era qualifiers are invisible to cosine. If BOTH query and candidate
# carry explicit decade/year tokens and their decade buckets are DISJOINT,
# the answer is wrong data by definition: reject (never serve).

_DECADE_RE = re.compile(r"\b(?:19|20)\d{2}s?\b")


def _decade_buckets(text: str | None) -> set[int]:
    """Decade buckets (floor(year/10)*10) for any years/decades in text.

    "1980s" → {1980}; "1990" → {1990}; "2020 to 2024" → {2020}; none → {}.
    """
    out: set[int] = set()
    for m in _DECADE_RE.findall(text or ""):
        out.add((int(m[:4]) // 10) * 10)
    return out

# ── Embedding model (singleton, lazy-loaded) ──────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384
SEM_THRESHOLD = float(os.environ.get("LCLLM_SEM_THRESHOLD", "0.45"))
# Time-sensitive entries older than this are never served (stale by
# definition — "latest news reports" from 4 days ago is not the latest).
EPHEMERAL_TTL_HOURS = float(os.environ.get("EPHEMERAL_TTL_HOURS", "24"))
CODE_LEXICON_FAST_PATH = os.environ.get("CODE_LEXICON_FAST_PATH", "1") == "1"

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


# ── Code lexicon fast path ────────────────────────────────────────
#
# For purpose="code" lookups, run a rare-token lexicon pre-check BEFORE the
# embedding pipeline. Exact-token rewrites ("change the csv export to use
# semicolon") share identifiers that appear in exactly ONE cached question
# (df==1: csv, yaml, sha256...) — a strong lexical anchor that makes the
# embedding step unnecessary. If no rare-token anchor exists, fall through to
# the semantic path (paraphrased rewrites need recall, not precision — the
# code router's p_solve judge is the final precision gate anyway).
#
# Benchmark: scripts/test_code_matchers.py + scripts/test_judge_veto.py
# (2026-08-20) — rare-token gate recalls exact rewrites, misses paraphrases
# (semantic catches those), and the judge vetoes same-family-different-task.

_CODE_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "for", "on", "with", "write", "implement",
    "create", "make", "use", "add", "set", "up", "and", "or", "not", "my", "i",
    "we", "you", "can", "could", "would", "should", "this", "that", "is", "are",
    "was", "were", "be", "been", "it", "its", "as", "at", "by", "from", "into",
    "over", "under", "again", "further", "then", "once", "here", "there", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "so", "than", "too", "very", "code", "function", "python",
    # generic words that pollute df==1 on small corpora (rare only at scale:
    # real identifiers like csv/yaml/sha256/binary stay meaningful)
    "file", "list", "value", "data", "number", "format", "output", "input",
    "way", "get", "put", "need", "want", "like", "just", "also", "new", "old",
}


def _code_tokenize(text: str) -> list[str]:
    """Code-aware tokenizer: camelCase split, snake_case split, keep digits."""
    text = text.lower()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [t for t in text.split() if t and t not in _CODE_STOPWORDS and len(t) > 1]


def _build_code_lexicon(rows: list[dict]) -> dict:
    """Build the rare-token index from cached code questions.

    rows: list of dicts with at least id + query (answer/model_used/etc kept
    for the returned candidate). Returns index dict:
      df: Counter token -> doc frequency
      rare: {row_id: set(rare tokens)}  (df == 1 = strong identifier)
      rows: {row_id: row}
    """
    toks = {r["id"]: set(_code_tokenize(r.get("query", ""))) for r in rows}
    df = Counter()
    for ts in toks.values():
        df.update(ts)
    rare = {rid: {t for t in ts if df[t] == 1} for rid, ts in toks.items()}
    return {
        "df": df,
        "rare": rare,
        "rows": {r["id"]: r for r in rows},
    }


_code_index: dict | None = None
_code_index_sig: tuple | None = None
_code_index_checked: float = 0.0
_CODE_INDEX_TTL = 30.0  # seconds between DB signature checks


def _refresh_code_index() -> dict:
    """Load code-purpose cache rows and rebuild the lexicon index on change.

    Signature check (COUNT + MAX id) is a cheap read; full rebuild only when
    the code cache actually changed. TTL-bounded to avoid a SELECT per call.
    """
    global _code_index, _code_index_sig, _code_index_checked
    now = time.time()
    if _code_index is not None and (now - _code_index_checked) < _CODE_INDEX_TTL:
        return _code_index
    _code_index_checked = now
    try:
        from db import get_conn
        conn = get_conn()
        sig_row = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM qa_cache WHERE purpose = 'code'"
        ).fetchone()
        sig = (sig_row[0], sig_row[1])
        if sig == _code_index_sig:
            return _code_index
        rows = [dict(r) for r in conn.execute(
            "SELECT id, query, answer, model_used, hit_count, created_at "
            "FROM qa_cache WHERE purpose = 'code'"
        ).fetchall()]
        _code_index = _build_code_lexicon(rows)
        _code_index_sig = sig
    except Exception:
        logger.exception("Code lexicon index refresh failed — using stale index")
    return _code_index or {"df": Counter(), "rare": {}, "rows": {}}


def _code_lexicon_lookup(match_query: str, index: dict | None = None) -> dict | None:
    """Rare-token lexicon pre-check. Returns best code-cache row (≥1 shared
    rare token) or None (fall through to semantic path).

    index: optional prebuilt index (testing); defaults to the live DB index.
    """
    if index is None:
        index = _refresh_code_index()
    qset = set(_code_tokenize(match_query))
    best_id, best_score = None, 0
    for rid, rare in index["rare"].items():
        shared = len(qset & rare)
        if shared > best_score:
            best_score, best_id = shared, rid
    if best_id is None or best_score < 1:
        return None
    row = dict(index["rows"][best_id])
    row["rank"] = 0.0  # shape parity with search_candidates rows
    row["_lexicon_score"] = best_score
    return row


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
    # Code fast path: rare-token lexicon pre-check (skips embedding entirely).
    if purpose == "code" and CODE_LEXICON_FAST_PATH:
        fast = _code_lexicon_lookup(match_query)
        if fast:
            score = fast.pop("_lexicon_score", 0)
            logger.info(
                "cache verdict=HIT source=lexicon-fast score=%s query=%r matched=%r",
                score, match_query[:80], fast.get("query", "")[:80],
            )
            return fast

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
        best = candidates[best_idx]

        # Era gate: disjoint decade/year buckets = wrong data, never serve.
        q_dec = _decade_buckets(match_query)
        c_dec = _decade_buckets(best.get("query", ""))
        if q_dec and c_dec and q_dec.isdisjoint(c_dec):
            logger.info(
                "cache verdict=SKIP source=decade-mismatch query_decades=%s "
                "cached_decades=%s query=%r matched=%r",
                sorted(q_dec), sorted(c_dec), match_query[:80], best.get("query", "")[:80],
            )
            return None

        # Freshness gate (read side): an ephemeral entry older than
        # EPHEMERAL_TTL_HOURS is stale by definition — never serve it.
        # (Hot-cache exact repeats are unaffected; this is semantic-only.)
        if is_ephemeral_query(best.get("query", "")):
            age_h = (age_days(created[best_idx]) or 0.0) * 24.0
            if age_h > EPHEMERAL_TTL_HOURS:
                logger.info(
                    "cache verdict=SKIP source=ephemeral-stale age_hours=%.1f "
                    "query=%r matched=%r",
                    age_h, match_query[:80], best.get("query", "")[:80],
                )
                return None

        logger.info(
            "cache verdict=HIT source=semantic cosine=%.3f hybrid=%.3f candidates=%d "
            "age_days=%s query=%r matched=%r",
            float(sem_scores[best_idx]), float(hybrid[best_idx]), len(candidates),
            age_days(created[best_idx]),
            match_query[:80], best.get("query", "")[:80],
        )
        return best

    except Exception:
        logger.exception("Embedding scoring failed — treating as cache miss")
        return None
