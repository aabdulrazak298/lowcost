"""A/B: does the cached-code example improve the cheap model's code output?

For each pair, run the cheap model TWICE:
  WITH example  -> WRITER_PROMPT (cached task + cached solution)
  WITHOUT       -> same new task, no example (from scratch)
Then check whether the WITH output (a) keeps the cached structure and
(b) actually implements the requested change.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_proxy import WRITER_PROMPT
from llm import _client_for_model

MODEL = "qwen/qwen3.7-flash"  # pinned — no silent fallback

PAIRS = [
    {
        "cached_q": "export a list of dicts to csv file",
        "cached_a": (
            "import csv\n"
            "def export_csv(rows, path):\n"
            "    with open(path, 'w', newline='') as f:\n"
            "        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))\n"
            "        writer.writeheader()\n"
            "        writer.writerows(rows)"
        ),
        "new_q": "change the csv export to use semicolon delimiter and include header",
        "checks": ["delimiter", ";", "header"],
    },
    {
        "cached_q": "write a merge sort function in python",
        "cached_a": (
            "def merge_sort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    mid = len(arr) // 2\n"
            "    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))\n"
            "def merge(left, right):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] <= right[j]:\n"
            "            result.append(left[i]); i += 1\n"
            "        else:\n"
            "            result.append(right[j]); j += 1\n"
            "    result.extend(left[i:]); result.extend(right[j:])\n"
            "    return result"
        ),
        "new_q": "rewrite the merge sort to sort in descending order instead of ascending",
        "checks": [">=", "descend", "reverse"],
    },
]

NO_EXAMPLE_PROMPT = (
    "Write a python function for this task. Output ONLY the function code — "
    "no explanation, no tests, no markdown fences.\n\nTask: {new_q}"
)


async def run_with(model_prompt: str) -> str:
    client = _client_for_model(MODEL)
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": model_prompt}],
        temperature=0.2,
        max_tokens=1500,
    )
    return (resp.choices[0].message.content or "").strip()


async def main():
    for p in PAIRS:
        print(f"\n{'='*70}\nNEW TASK: {p['new_q']}")
        print(f"CACHED:  {p['cached_q']}")

        with_ex = await run_with(WRITER_PROMPT.format(
            cached_q=p["cached_q"], cached_a=p["cached_a"], new_q=p["new_q"]))
        without = await run_with(NO_EXAMPLE_PROMPT.format(new_q=p["new_q"]))

        print(f"\n--- WITH EXAMPLE ({len(with_ex)} chars) ---")
        print(with_ex[:600])
        print(f"\n--- WITHOUT EXAMPLE ({len(without)} chars) ---")
        print(without[:600])

        w_hits = [c for c in p["checks"] if c in with_ex.lower()]
        o_hits = [c for c in p["checks"] if c in without.lower()]
        print(f"\nrequirement markers -> WITH: {w_hits}  WITHOUT: {o_hits}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
