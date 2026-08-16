"""Few-shot A/B: does a fully WORKED example (not a formula) let the cheap model adapt?

Earlier tests gave an abstract METHOD. This gives a complete worked example —
base problem + every number plugged in + final answer — then asks the variant.
This is true few-shot / in-context learning, which is a different capability.

Arms:
  DIRECT  = cheap model on the variant, bare.
  EXAMPLE = cheap model on the variant + a fully worked base example.

Correct answers computed in-script. Reasoning OFF (the real cheap config).

Run: cd ~/cloud/projects/lowcostllm && .venv/bin/python tests/test_fewshot.py
"""
import asyncio
import itertools
import math
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
load_dotenv(Path.home() / ".hermes" / ".env", override=False)

OR_KEY = os.getenv("OPENROUTER_API_KEY", "")
OR_BASE = "https://openrouter.ai/api/v1"
OR_HEADERS = {
    "Authorization": f"Bearer {OR_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8800",
    "X-Title": "LowCostLLM-FewShot",
}
CHEAP_MODEL = os.getenv("TEST_CHEAP_MODEL", "qwen/qwen3.7-flash")
REASONING_ON = os.getenv("REASONING", "0") == "1"
_EXTRA = {"reasoning": {"enabled": True}} if REASONING_ON else {"reasoning": {"enabled": False}}
_MAX_TOK = 8000 if REASONING_ON else 1500


def solve_bayes(pa, pa_def, pb, pb_def, pc, pc_def):
    p_good = (1 - pa_def) * pa + (1 - pb_def) * pb + (1 - pc_def) * pc
    return round(((1 - pa_def) * pa / p_good) * 100, 1)


def solve_tank(d, h, fill_hi, fill_lo, flow_lps):
    V = math.pi * (d / 2) ** 2 * h
    dV = V * (fill_hi - fill_lo) / 100
    return round(dV / (flow_lps / 1000) / 60, 1)


def solve_perm(constraints):
    return sum(1 for p in itertools.permutations("ABCDEF") if all(c(list(p)) for c in constraints))


def solve_grid(size):
    from functools import lru_cache
    N = size - 1

    @lru_cache(None)
    def go(x, y):
        if (x, y) == (N, N):
            return 1
        if x > N or y > N:
            return 0
        t = 0
        if y % 2 == 0:
            t += go(x + 1, y)
        if x % 2 == 1:
            t += go(x, y + 1)
        return t
    return go(0, 0)


# Fully worked examples: base problem -> complete solution with numbers -> answer.
PAIRS = [
    {
        "id": "bayes", "answer": solve_bayes(0.60, 0.02, 0.25, 0.04, 0.15, 0.08),
        "example": ("EXAMPLE:\n"
                    "Q: Three machines: A makes 50% (1% defective), B 30% (3%), C 20% (6%). A part is NOT defective. P(came from A)?\n"
                    "A: P(good|A)=0.99, P(good|B)=0.97, P(good|C)=0.94.\n"
                    "P(A|good) = (0.99)(0.5) / [(0.99)(0.5)+(0.97)(0.3)+(0.94)(0.2)]\n"
                    "          = 0.495 / (0.495 + 0.291 + 0.188) = 0.495/0.974 = 50.8%."),
        "variant_q": "Three machines make parts: A makes 60% with 2% defective, B 25% with 4% defective, C 15% with 8% defective. A part is found NOT defective. What is the probability (percentage, one decimal) it came from machine A? Give only the final answer.",
    },
    {
        "id": "tank", "answer": solve_tank(1.5, 2.5, 90, 30, 8),
        "example": ("EXAMPLE:\n"
                    "Q: Cylindrical tank d=2.0m, h=3.0m, 80% full, drained at 12 L/s to 25% full. Minutes?\n"
                    "A: V=pi*(1.0)^2*(3.0)=9.4248 m3. 80%=7.5398 m3, 25%=2.3562 m3.\n"
                    "Drain 7.5398-2.3562=5.1836 m3. 12 L/s=0.012 m3/s.\n"
                    "t=5.1836/0.012=431.97 s = 7.2 minutes."),
        "variant_q": "A cylindrical tank (diameter 1.5 m, height 2.5 m) is 90% full. A pump drains it at 8 L/s. How many minutes (one decimal) to drain down to 30% full? Give only the final answer.",
    },
    {
        "id": "bayes2", "answer": round((0.03 * 0.80) / (0.03 * 0.80 + 0.10 * 0.20) * 100, 1),
        "example": ("EXAMPLE:\n"
                    "Q: Machine A makes 70% (2% defective), B 30% (5%). A part is defective. P(came from A)?\n"
                    "A: P(def|A)=0.02, P(def|B)=0.05.\n"
                    "P(A|def) = (0.02)(0.7) / [(0.02)(0.7)+(0.05)(0.3)] = 0.014/0.029 = 48.3%."),
        "variant_q": "Two machines: X makes 80% of parts with 3% defective, Y makes 20% with 10% defective. A part is found defective. What is the probability (percentage, one decimal) it came from X? Give only the final answer.",
    },
    {
        "id": "perm", "answer": solve_perm([
            lambda p: p.index("A") < p.index("B"),
            lambda p: p.index("B") < p.index("C"),
            lambda p: p[0] != "C",
            lambda p: p.index("D") == p.index("E") + 1,
            lambda p: p.index("F") < p.index("D")]),
        "example": ("EXAMPLE:\n"
                    "Q: 6 people A-F line up. A before B; C not first; D immediately after E; F before D. How many orderings?\n"
                    "A: Enumerate all 6!=720 permutations, keep those satisfying all constraints -> 24."),
        "variant_q": "Six people A, B, C, D, E, F line up in a row. How many valid orderings satisfy ALL of: A before B; B before C; C not first; D immediately after E; F before D? Give only the final answer.",
    },
    {
        "id": "grid", "answer": solve_grid(6),
        "example": ("EXAMPLE:\n"
                    "Q: 5x5 grid (0,0)->(4,4), Right/Up only; even x cannot go Up, odd y cannot go Right. How many shortest paths?\n"
                    "A: Recursively count valid moves to the corner -> 3."),
        "variant_q": "A car navigates a 6x6 grid from (0,0) to (5,5), moving only Right (+x) or Up (+y). When on an even x-coordinate it cannot move Up, when on an odd y-coordinate it cannot move Right. How many distinct shortest paths exist? Give only the final answer.",
    },
]


async def chat(client, messages):
    payload = {"model": CHEAP_MODEL, "messages": messages, "max_tokens": _MAX_TOK, "temperature": 0.7}
    if _EXTRA:
        payload.update(_EXTRA)
    for attempt in range(6):
        resp = await client.post(f"{OR_BASE}/chat/completions", headers=OR_HEADERS, json=payload)
        if resp.status_code in (429, 500, 502, 503, 504):
            await asyncio.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        data = resp.json()
        text = (data["choices"][0]["message"].get("content") or "").strip()
        if text:
            return text
        await asyncio.sleep(1)
    return ""


async def run():
    print("=" * 82)
    print("FEW-SHOT A/B — does a WORKED EXAMPLE let the cheap model adapt?")
    print(f"  model : {CHEAP_MODEL}  reasoning={'ON' if REASONING_ON else 'OFF'}")
    print("=" * 82)

    async with httpx.AsyncClient(timeout=120) as client:
        for item in PAIRS:
            print(f"\n{'─' * 82}")
            print(f"[{item['id']}]  CORRECT = {item['answer']}")
            print(f"{'─' * 82}")

            direct = await chat(client, [{"role": "user", "content": item["variant_q"]}])
            fewshot = await chat(client, [{"role": "user", "content": (
                f"{item['example']}\n\n"
                f"Now solve this the same way:\n{item['variant_q']}"
            )}])

            print(f"  DIRECT (bare)   : {direct[:140]}")
            print(f"  EXAMPLE (fewshot): {fewshot[:140]}")
            print()

    print("=" * 82)
    print("DONE — grade each arm against CORRECT.")


if __name__ == "__main__":
    asyncio.run(run())
