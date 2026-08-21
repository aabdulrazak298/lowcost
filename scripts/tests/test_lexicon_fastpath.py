"""Unit tests for the code lexicon fast path (matcher.py).

Uses a synthetic index built from benchmark sources — NO live DB writes.
Verifies:
  - exact/identifier-anchored rewrite queries hit the right source
  - paraphrased rewrites and adversarial same-family queries get NO lexicon
    hit (fall through to the semantic path)
  - tokenizer handles camelCase / snake_case / digits
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matcher import _build_code_lexicon, _code_lexicon_lookup, _code_tokenize

ROWS = [
    {"id": 1, "query": "write a merge sort function in python",
     "answer": "def merge_sort(arr): ...", "model_used": "deepseek-v4-flash"},
    {"id": 2, "query": "write an http get client with timeout and retry",
     "answer": "import requests ...", "model_used": "deepseek-v4-flash"},
    {"id": 3, "query": "create a config loader that parses ini files",
     "answer": "import configparser ...", "model_used": "deepseek-v4-flash"},
    {"id": 4, "query": "export a list of dicts to csv file",
     "answer": "import csv ...", "model_used": "deepseek-v4-flash"},
    {"id": 5, "query": "write a retry decorator with fixed delay",
     "answer": "import time ...", "model_used": "deepseek-v4-flash"},
    {"id": 6, "query": "implement binary search in python",
     "answer": "def binary_search ...", "model_used": "deepseek-v4-flash"},
    {"id": 7, "query": "validate email and phone number format",
     "answer": "import re ...", "model_used": "deepseek-v4-flash"},
]

INDEX = _build_code_lexicon(ROWS)

# (query, expected_source_id or None if should fall through)
CASES = [
    # exact / identifier-anchored rewrites -> lexicon HIT
    ("rewrite the merge sort to sort in descending order instead of ascending", 1),
    ("modify the http get client to add a timeout parameter and retry on connection errors", 2),
    ("refactor the config loader to support yaml format", 3),
    ("change the csv export to use semicolon delimiter and include header", 4),
    ("update the retry decorator to exponential backoff with max attempts", 5),
    ("rewrite binary search to work on a list of floats", 6),
    ("add validation for malaysian phone numbers to the validator", 7),
    # paraphrased rewrites -> semantic path, EXCEPT identifier-anchored ones
    # ("client"/"timeout" are identifiers — fast path legitimately catches them)
    ("the client hangs when the server is slow, add a timeout so it fails fast", 2),
    ("make my sorting function faster and handle already-sorted input without extra work", None),
    ("the spreadsheet output is missing the column names row", None),
    ("find the position of a value in a sorted list without scanning everything", None),
    # adversarial same-family -> NO lexicon hit (judge would veto anyway)
    ("implement quicksort in python", None),
    ("implement a rate limiter using the token bucket algorithm", None),
    ("write a function that computes the sha256 checksum of a large file", None),
]


def main():
    passed = failed = 0
    for query, expected in CASES:
        hit = _code_lexicon_lookup(query, index=INDEX)
        got = hit["id"] if hit else None
        ok = got == expected
        passed += ok
        failed += (not ok)
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}: {query[:60]:<62} -> {got} (expected {expected})")
        if hit:
            print(f"      score={hit.get('_lexicon_score')} matched={hit['query'][:50]}")
    print(f"\n{passed}/{len(CASES)} passed, {failed} failed")

    # tokenizer sanity
    assert _code_tokenize("httpGetClient") == ["httpgetclient"], _code_tokenize("httpGetClient")
    assert _code_tokenize("binary_search") == ["binary", "search"], _code_tokenize("binary_search")
    assert "csv" in _code_tokenize("export to csv")
    print("tokenizer sanity: PASS")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
