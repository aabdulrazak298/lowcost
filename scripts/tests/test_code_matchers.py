"""Code-path matcher benchmark: semantic vs lexicon (weighted token overlap)
vs FTS5 BM25 — which retrieves the RIGHT cached source for rewrite/modify
queries and REJECTS genuinely different queries?

Setup (Azuan's spec): 7 code sources (original questions + code), a few
similar modifications/rewrites per source, 1-2 completely different queries.

Ground truth:
  - M-queries (modifications) must retrieve THEIR source, rank #1 ideally
  - D-queries (different) must retrieve NOTHING (no false hit)

Metrics per method:
  - Mean rank of the correct source (1 = perfect)
  - False hits on D-queries (must be 0)
  - Score separation: min(correct scores) vs max(wrong scores)
"""
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matcher import _embed, _cosine_scores

# ── Corpus: 7 code sources (question that produced them, then the code) ──
SOURCES = [
    {
        "q": "write a merge sort function in python",
        "code": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i]); i += 1\n        else:\n            result.append(right[j]); j += 1\n    result.extend(left[i:]); result.extend(right[j:])\n    return result",
    },
    {
        "q": "write an http get client with timeout and retry",
        "code": "import requests\n\ndef http_get(url, timeout=5):\n    for attempt in range(3):\n        try:\n            resp = requests.get(url, timeout=timeout)\n            resp.raise_for_status()\n            return resp.text\n        except requests.RequestException:\n            if attempt == 2:\n                raise",
    },
    {
        "q": "create a config loader that parses ini files",
        "code": "import configparser\n\ndef load_config(path):\n    parser = configparser.ConfigParser()\n    parser.read(path)\n    return {s: dict(parser.items(s)) for s in parser.sections()}",
    },
    {
        "q": "export a list of dicts to csv file",
        "code": "import csv\n\ndef export_csv(rows, path):\n    with open(path, 'w', newline='') as f:\n        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))\n        writer.writeheader()\n        writer.writerows(rows)",
    },
    {
        "q": "write a retry decorator with fixed delay",
        "code": "import time\nfrom functools import wraps\n\ndef retry(times=3, delay=1):\n    def deco(fn):\n        @wraps(fn)\n        def wrapper(*args, **kwargs):\n            for _ in range(times):\n                try:\n                    return fn(*args, **kwargs)\n                except Exception:\n                    time.sleep(delay)\n            raise\n        return wrapper\n    return deco",
    },
    {
        "q": "implement binary search in python",
        "code": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    },
    {
        "q": "validate email and phone number format",
        "code": "import re\n\ndef is_valid_email(s):\n    return bool(re.match(r'^[\\w.+-]+@[\\w-]+\\.[\\w.]+$', s))\n\ndef is_valid_phone(s):\n    return bool(re.match(r'^\\+?[\\d\\s-]{8,15}$', s))",
    },
]

# Modifications/rewrites: query -> index of the source it must hit
# (easy = keyword-aligned; hard = paraphrased, less lexical overlap)
MODS = [
    (0, "rewrite the merge sort to sort in descending order instead of ascending"),
    (1, "modify the http get client to add a timeout parameter and retry on connection errors"),
    (2, "refactor the config loader to support yaml format"),
    (3, "change the csv export to use semicolon delimiter and include header"),
    (4, "update the retry decorator to exponential backoff with max attempts"),
    (5, "rewrite binary search to work on a list of floats"),
    (6, "add validation for malaysian phone numbers to the validator"),
    # hard / paraphrased — the cheap-model rewrite is still appropriate
    (0, "make my sorting function faster and handle already-sorted input without extra work"),
    (1, "the client hangs when the server is slow, add a timeout so it fails fast"),
    (2, "the config file has sections and comments, parse those too"),
    (3, "the spreadsheet output is missing the column names row"),
    (5, "find the position of a value in a sorted list without scanning everything"),
]

# Completely different: must retrieve NOTHING
DIFFS = [
    "implement a rate limiter using the token bucket algorithm",
    "write a function that computes the sha256 checksum of a large file",
    # adversarial: same task family, DIFFERENT task — semantic blur risk
    "implement quicksort in python",
    "write an http post client that sends json and retries on failure",
    "parse a json config file into a dictionary",
    "generate an excel xlsx report from a database query",
]

# ── Tokenizer: code-aware ──────────────────────────────────────────
STOPWORDS = set(
    "the a an to of in for on with write implement create make use add set up out "
    "and or not my i we you can could would should this that is are was were be been "
    "it its as at by from into over under again further then once here there all any "
    "both each few more most other some such only own same so than too very".split()
)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)  # camelCase split
    text = re.sub(r"[^a-z0-9]+", " ", text)  # non-alnum -> space
    toks = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return toks


def is_identifier_like(tok: str) -> bool:
    """Code-y tokens: contain digits or underscore (error codes, var names)."""
    return any(ch.isdigit() for ch in tok) or "_" in tok


def lex_jaccard(q_toks: list[str], c_toks: list[str]) -> float:
    """Weighted token overlap (identifier-like tokens weigh 2x). 0..1."""
    qs, cs = set(q_toks), set(c_toks)
    if not qs or not cs:
        return 0.0
    inter = qs & cs
    w_inter = sum(2 if is_identifier_like(t) else 1 for t in inter)
    w_union = (
        sum(2 if is_identifier_like(t) else 1 for t in qs | cs)
    )
    return w_inter / w_union if w_union else 0.0


def lex_idf(q_toks: list[str], c_toks: list[str], idf: dict[str, float]) -> float:
    """IDF-weighted token overlap: rare tokens dominate, generic words ~0."""
    qs, cs = set(q_toks), set(c_toks)
    inter = qs & cs
    if not inter:
        return 0.0
    w_inter = sum(idf.get(t, 0.0) for t in inter)
    # normalize by the query's total IDF mass so long queries don't inflate
    w_total = sum(idf.get(t, 0.0) for t in qs)
    return w_inter / w_total if w_total else 0.0


def fts5_bm25_scores(q_toks: list[str], corpus_toks: list[str]) -> np.ndarray:
    """In-memory SQLite FTS5 BM25 ranking (OR query, min-max normalized)."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE docs USING fts5(body)")
    for ct in corpus_toks:
        con.execute("INSERT INTO docs(body) VALUES (?)", (" ".join(ct),))
    if not q_toks:
        return np.zeros(len(corpus_toks))
    query = " OR ".join(f'"{t}"' for t in q_toks)
    try:
        rows = con.execute(
            "SELECT rowid, bm25(docs) FROM docs WHERE body MATCH ? ORDER BY rank",
            (query,),
        ).fetchall()
    except sqlite3.OperationalError:
        return np.zeros(len(corpus_toks))
    scores = np.zeros(len(corpus_toks))
    for rowid, rank in rows:
        scores[rowid - 1] = -rank  # bm25: lower (more negative) = better
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores


# ── Build representations ───────────────────────────────────────────
queries = [s["q"] for s in SOURCES]
q_toks = [tokenize(q) for q in queries]
corpus_code_toks = [tokenize(s["code"]) for s in SOURCES]

# IDF over the question corpus (rare tokens = discriminative)
from collections import Counter
_df = Counter()
for qt in q_toks:
    for t in set(qt):
        _df[t] += 1
N = len(q_toks)
idf = {t: max(0.0, (N - df + 0.5) / (df + 0.5) + 1.0) for t, df in _df.items()}

# Semantic vectors (real production embedding model)
sem_queries = _embed(queries)
sem_codes = _embed([s["code"] for s in SOURCES])

# ── Evaluate one method ─────────────────────────────────────────────
def evaluate(name, score_fn, threshold):
    """score_fn(query) -> np.ndarray of scores over the corpus (higher=better)."""
    results = []
    for (correct, q) in MODS:
        scores = score_fn(q)
        pick = int(np.argmax(scores))
        results.append((q[:45], correct, pick, float(scores[pick]), pick == correct and scores[pick] >= threshold))

    diff_results = []
    for q in DIFFS:
        scores = score_fn(q)
        best = int(np.argmax(scores))
        diff_results.append((q[:45], best, float(scores[best]), scores[best] >= threshold))

    recall_top1 = sum(1 for r in results if r[2] == r[1]) / len(results)
    hit_top1 = sum(1 for r in results if r[4]) / len(results)
    false_hits = sum(1 for d in diff_results if d[3])

    correct_scores = [r[3] for r in results]
    wrong_best = []
    for (correct, q) in MODS:
        s = score_fn(q)
        wrong_best.append(float(np.max(np.delete(s, correct))))
    margin = (min(correct_scores) - max(wrong_best)) if correct_scores and wrong_best else float("nan")

    print(f"\n=== {name} (threshold {threshold}) ===")
    for r in results:
        mark = "✓" if r[4] else ("~" if r[2] == r[1] else "✗")
        print(f"  {mark} mod: {r[0]:<60} src={r[1]} pick={r[2]} score={r[3]:.3f}")
    for d in diff_results:
        mark = "✗ FALSE HIT" if d[3] else "✓ rejected"
        print(f"  {mark}: {d[0]:<55} best_src={d[1]} score={d[2]:.3f}")
    print(f"  -> recall@1={recall_top1:.2f} hit@1={hit_top1:.2f} "
          f"false_hits={false_hits} margin(min_correct-max_wrong)={margin:+.3f}")
    return {"recall_top1": recall_top1, "false_hits": false_hits, "margin": margin}


# Method 1: SEMANTIC — query emb vs cached-CODE emb
score_fn_sem = lambda q: _cosine_scores(_embed([q]), sem_codes)
# Method 2: SEMANTIC — query emb vs cached-QUESTION emb (what production does)
score_fn_sem_q = lambda q: _cosine_scores(_embed([q]), sem_queries)
# Method 3: LEXICON — weighted Jaccard vs cached QUESTION
score_fn_lex = lambda q: np.array([lex_jaccard(tokenize(q), qt) for qt in q_toks])
# Method 4: LEXICON — weighted Jaccard vs cached CODE
score_fn_lex_c = lambda q: np.array([lex_jaccard(tokenize(q), ct) for ct in corpus_code_toks])
# Method 5: FTS5 BM25 vs cached QUESTION tokens
score_fn_bm = lambda q: fts5_bm25_scores(tokenize(q), q_toks)
# Method 6: HYBRID (production chat blend) — 0.7×cosine + 0.3×bm25norm, cosine gate
def score_fn_hybrid(q):
    sem = _cosine_scores(_embed([q]), sem_queries)
    bm = fts5_bm25_scores(tokenize(q), q_toks)
    return 0.7 * sem + 0.3 * bm
# Method 7: LEXICON-IDF — rare-token weighted overlap vs QUESTION
score_fn_lexidf = lambda q: np.array([lex_idf(tokenize(q), qt, idf) for qt in q_toks])
# Method 8: LEXICON-IDF vs CODE
score_fn_lexidf_c = lambda q: np.array([lex_idf(tokenize(q), ct, idf) for ct in corpus_code_toks])
# Method 9: RARE-TOKEN GATE — count shared tokens that identify ONE source (df==1)
rare_toks = [set(t for t in qt if _df[t] == 1) for qt in q_toks]

def score_fn_rare(q):
    qset = set(tokenize(q))
    return np.array([len(qset & r) for r in rare_toks])


print("Semantic threshold 0.45 (production SEM_THRESHOLD-ish)")
evaluate("SEMANTIC vs CODE", score_fn_sem, 0.45)
evaluate("SEMANTIC vs QUESTION", score_fn_sem_q, 0.45)
evaluate("LEXICON (weighted Jaccard) vs QUESTION", score_fn_lex, 0.15)
evaluate("LEXICON (weighted Jaccard) vs CODE", score_fn_lex_c, 0.15)
evaluate("FTS5 BM25 vs QUESTION", score_fn_bm, 0.20)
evaluate("HYBRID 0.7sem+0.3bm25 (cosine gate)", score_fn_hybrid, 0.45)
evaluate("LEXICON-IDF vs QUESTION", score_fn_lexidf, 0.25)
evaluate("LEXICON-IDF vs CODE", score_fn_lexidf_c, 0.25)
evaluate("RARE-TOKEN GATE (df==1, >=1 shared)", score_fn_rare, 1.0)
