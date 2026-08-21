"""Judge (p_solve) veto test on code pairs.

The matcher surfaces candidates; the judge decides whether the CHEAP model can
safely ADAPT the cached answer to the new query (p_solve >= 0.5 -> cheap
writes it; below -> expensive answers fresh).

VETO pairs (dangerous): cheap must NOT adapt — different task despite
same-family wording (semantic matcher false-hits these).
ACCEPT pairs (good rewrites): cheap SHOULD adapt — genuine rewrite/modify.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_proxy import _judge

# (cached_question, cached_answer_code, new_query, expected)
PAIRS = [
    # ── VETO: same family, different task ────────────────────────────
    (
        "write a merge sort function in python",
        "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr)//2\n    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))\n\ndef merge(l, r):\n    out=[]; i=j=0\n    while i<len(l) and j<len(r):\n        if l[i]<=r[j]: out.append(l[i]); i+=1\n        else: out.append(r[j]); j+=1\n    out+=l[i:]; out+=r[j:]\n    return out",
        "implement quicksort in python",
        "VETO",
    ),
    (
        "write a merge sort function in python",
        "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr)//2\n    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))\n\ndef merge(l, r):\n    out=[]; i=j=0\n    while i<len(l) and j<len(r):\n        if l[i]<=r[j]: out.append(l[i]); i+=1\n        else: out.append(r[j]); j+=1\n    out+=l[i:]; out+=r[j:]\n    return out",
        "write a function that computes the sha256 checksum of a large file",
        "VETO",
    ),
    (
        "write an http get client with timeout and retry",
        "import requests\ndef http_get(url, timeout=5):\n    for attempt in range(3):\n        try:\n            r = requests.get(url, timeout=timeout); r.raise_for_status()\n            return r.text\n        except requests.RequestException:\n            if attempt == 2: raise",
        "implement a rate limiter using the token bucket algorithm",
        "VETO",
    ),
    (
        "export a list of dicts to csv file",
        "import csv\ndef export_csv(rows, path):\n    with open(path,'w',newline='') as f:\n        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))\n        w.writeheader(); w.writerows(rows)",
        "generate an excel xlsx report from a database query",
        "VETO",
    ),
    # ── ACCEPT: genuine rewrites / modifications ─────────────────────
    (
        "write an http get client with timeout and retry",
        "import requests\ndef http_get(url, timeout=5):\n    for attempt in range(3):\n        try:\n            r = requests.get(url, timeout=timeout); r.raise_for_status()\n            return r.text\n        except requests.RequestException:\n            if attempt == 2: raise",
        "write an http post client that sends json and retries on failure",
        "ACCEPT",
    ),
    (
        "create a config loader that parses ini files",
        "import configparser\ndef load_config(path):\n    p = configparser.ConfigParser(); p.read(path)\n    return {s: dict(p.items(s)) for s in p.sections()}",
        "parse a json config file into a dictionary",
        "ACCEPT",
    ),
    (
        "write a merge sort function in python",
        "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr)//2\n    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))\n\ndef merge(l, r):\n    out=[]; i=j=0\n    while i<len(l) and j<len(r):\n        if l[i]<=r[j]: out.append(l[i]); i+=1\n        else: out.append(r[j]); j+=1\n    out+=l[i:]; out+=r[j:]\n    return out",
        "rewrite the merge sort to sort in descending order instead of ascending",
        "ACCEPT",
    ),
    (
        "export a list of dicts to csv file",
        "import csv\ndef export_csv(rows, path):\n    with open(path,'w',newline='') as f:\n        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))\n        w.writeheader(); w.writerows(rows)",
        "change the csv export to use semicolon delimiter and include header",
        "ACCEPT",
    ),
    (
        "implement binary search in python",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid]==target: return mid\n        if arr[mid]<target: lo=mid+1\n        else: hi=mid-1\n    return -1",
        "find the position of a value in a sorted list without scanning everything",
        "ACCEPT",
    ),
]

THRESHOLD = 0.5


async def main() -> None:
    results = []
    for cached_q, cached_a, new_q, expected in PAIRS:
        verdict = await _judge(cached_q, cached_a, new_q)
        if verdict is None:
            p = None
            note = "PARSE FAIL"
        else:
            p = verdict.get("p_solve")
            note = f"boundary={verdict.get('capability_boundary')}"
        actual = "ACCEPT" if p is not None and p >= THRESHOLD else "VETO"
        ok = actual == expected
        results.append(ok)
        mark = "✓" if ok else "✗ MISMATCH"
        print(f"{mark} [{expected:6s}->{actual:6s}] p_solve={p}  {note}")
        print(f"      new: {new_q[:70]}")
        print(f"      cached: {cached_q[:70]}")

    print(f"\n{sum(results)}/{len(results)} correct")
    print("(VETO = expensive answers fresh; ACCEPT = cheap model rewrites cached)")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
