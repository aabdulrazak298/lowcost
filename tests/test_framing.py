"""Framing A/B: does "use the expert answer as a GUIDE" fix adaptation?

Same variant questions and cheap model as test_adaptation.py, but THREE arms:
  DIRECT  = cheap model bare (no reference).
  DEMOTED = [Reference — background only, do not cite] (current production framing).
  GUIDE   = "use the expert's method as your guide, substitute the new values".

Correct answers computed in-script (exact ground truth). Reasoning OFF (the real
cheap config).

Run: cd ~/cloud/projects/lowcostllm && .venv/bin/python tests/test_framing.py
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
    "X-Title": "LowCostLLM-Framing",
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
    return sum(1 for p in itertools.permutations("ABCDEF")
               if all(c(list(p)) for c in constraints))


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


PAIRS = [
    {"id": "bayes", "base_q": "Three machines: A 50% (1% def), B 30% (3%), C 20% (6%). Part NOT defective -> P(from A)?",
     "ref": "Bayes with the 'not defective' flip: P(A|good) = P(good|A)*P(A) / [P(good|A)*P(A) + P(good|B)*P(B) + P(good|C)*P(C)], where P(good|m)=1-defect rate.",
     "variant_q": "Three machines make parts: A makes 60% with 2% defective, B makes 25% with 4% defective, C makes 15% with 8% defective. A part is found NOT defective. What is the probability (percentage, one decimal) it came from machine A? Give only the final answer.",
     "answer": solve_bayes(0.60, 0.02, 0.25, 0.04, 0.15, 0.08)},
    {"id": "tank", "base_q": "Cylindrical tank d=2.0m h=3.0m, 80% full, drain 12 L/s to 25% -> minutes?",
     "ref": "Volume = pi*(d/2)^2*h. Drain volume = V*(fill_hi - fill_lo)/100. Flow m3/s = L/s / 1000. Time = volume/flow, then /60 to minutes.",
     "variant_q": "A cylindrical tank (diameter 1.5 m, height 2.5 m) is 90% full of liquid. A pump drains it at 8 L/s. How many minutes (one decimal) to drain down to 30% full? Give only the final answer.",
     "answer": solve_tank(1.5, 2.5, 90, 30, 8)},
    {"id": "perm", "base_q": "6 people A-F line up. A before B; C not first; D immediately after E; F before D. Count orderings.",
     "ref": "Enumerate 6!=720 permutations, filter by the constraints, count survivors.",
     "variant_q": "Six people A, B, C, D, E, F line up in a row. How many valid orderings satisfy ALL of: A before B; B before C; C is not first; D is immediately after E; F is before D? Give only the final answer.",
     "answer": solve_perm([
         lambda p: p.index("A") < p.index("B"),
         lambda p: p.index("B") < p.index("C"),
         lambda p: p[0] != "C",
         lambda p: p.index("D") == p.index("E") + 1,
         lambda p: p.index("F") < p.index("D")])},
    {"id": "grid", "base_q": "5x5 grid (0,0)->(4,4), Right/Up; even x no Up, odd y no Right. Count shortest paths.",
     "ref": "Shortest path = N Right + N Up. Recurse: Right allowed only when y even; Up allowed only when x odd. Count paths to the corner.",
     "variant_q": "A car navigates a 6x6 grid from (0,0) to (5,5), moving only Right (+x) or Up (+y). When on an even x-coordinate it cannot move Up, when on an odd y-coordinate it cannot move Right. How many distinct shortest paths exist? Give only the final answer.",
     "answer": solve_grid(6)},
    {"id": "bayes2", "base_q": "Bayes (defective): A 70% (2% def), B 30% (5% def). Part defective -> P(from A)?",
     "ref": "Bayes with 'defective' condition: P(A|def) = P(def|A)*P(A) / [P(def|A)*P(A) + P(def|B)*P(B)].",
     "variant_q": "Two machines: X makes 80% of parts with 3% defective, Y makes 20% with 10% defective. A part is found defective. What is the probability (percentage, one decimal) it came from X? Give only the final answer.",
     "answer": round((0.03 * 0.80) / (0.03 * 0.80 + 0.10 * 0.20) * 100, 1)},
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
    print("FRAMING A/B — bare vs 'do not cite' vs 'use as guide' (adaptation)")
    print(f"  cheap model : {CHEAP_MODEL}  reasoning={'ON' if REASONING_ON else 'OFF'}")
    print("=" * 82)

    async with httpx.AsyncClient(timeout=120) as client:
        for item in PAIRS:
            print(f"\n{'─' * 82}")
            print(f"[{item['id']}]  CORRECT = {item['answer']}")
            print(f"{'─' * 82}")

            direct = await chat(client, [{"role": "user", "content": item["variant_q"]}])

            demoted = await chat(client, [{"role": "user", "content": (
                f"[Reference — background only, may be outdated, do not cite]\n"
                f"Similar question: {item['base_q']}\n{item['ref']}\n\n"
                f"Now answer: {item['variant_q']}"
            )}])

            guide = await chat(client, [{"role": "user", "content": (
                f"An expert already solved a SIMILAR question. Use their METHOD as your guide.\n"
                f"Similar question: {item['base_q']}\n"
                f"Expert method: {item['ref']}\n\n"
                f"Apply the SAME method to the new question below, substituting its values. "
                f"Give the correct answer for the NEW question:\n{item['variant_q']}"
            )}])

            print(f"  DIRECT (bare)     : {direct[:120]}")
            print(f"  DEMOTED (no cite) : {demoted[:120]}")
            print(f"  GUIDE (use method): {guide[:120]}")
            print()

    print("=" * 82)
    print("DONE — grade each arm against its CORRECT answer.")


if __name__ == "__main__":
    asyncio.run(run())
