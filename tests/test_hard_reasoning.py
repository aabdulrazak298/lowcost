"""Hard-reasoning cache A/B — grade correctness MYSELF against verified answers.

Refreshed to avoid the slow/deadlocking deepseek-v4-pro expert generation:
the "cache content" is a hardcoded VERIFIED solution (answers brute-forced or
computed by hand), and correctness is graded by comparing the cheap model's
stated answer to the known-correct answer — no LLM judge, no pro.

Arms per question:
  DIRECT  = cheap model answers bare (no reference).
  CACHED  = cheap model answers with the [Reference] block (verified solution).

Run: cd ~/cloud/projects/lowcostllm && .venv/bin/python tests/test_hard_reasoning.py
"""
import asyncio
import json
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
    "X-Title": "LowCostLLM-HardAB",
}
CHEAP_MODEL = os.getenv("TEST_CHEAP_MODEL", "qwen/qwen3.7-flash")
OR_REASONING_OFF = {"reasoning": {"enabled": False}}

# Verified ground truth: q, answer (human-readable), reference solution (cache content).
QUESTIONS = [
    {
        "id": "box_coin",
        "answer": "0",
        "q": "Three opaque boxes A, B, C hold three distinct coins: a 50-gram silver, a 100-gram silver, and a 100-gram lead coin (one coin per box). The boxes are labelled 'Silver 50', 'Silver 100', and 'Lead 100', but all three labels are wrong. You draw one coin from Box A and it is silver. What is the probability that Box C holds the 100-gram silver coin? Give only the final answer.",
        "ref": "Enumerate all 6 assignments of {S50,S100,L100} to boxes A,B,C. 'All labels wrong' means A≠S50, B≠S100, C≠L100, which leaves exactly 2 valid: (A=S100,B=L100,C=S50) and (A=L100,B=S50,C=S100). Drawing silver from A rules out A=L100, leaving (A=S100,B=L100,C=S50). So C holds S50, not S100. Probability = 0.",
    },
    {
        "id": "grid_path",
        "answer": "3",
        "q": "A car navigates a 5x5 grid from (0,0) to (4,4), moving only Right (+x) or Up (+y), one cell at a time. But when the car is on an even x-coordinate it cannot move Up, and when it is on an odd y-coordinate it cannot move Right. How many distinct shortest paths exist? Give only the final answer.",
        "ref": "A shortest path is 4 Right + 4 Up = 8 moves. Enumerate recursively with the move rules (Right allowed only when y is even, Up allowed only when x is odd). This prunes most of the 70 unrestricted paths down to exactly 3 valid paths. Answer = 3.",
    },
    {
        "id": "perm_count",
        "answer": "24",
        "q": "Six people A, B, C, D, E, F line up in a single row. How many valid orderings satisfy ALL of: A is before B; C is not first; D is immediately after E; F is before D? Give only the final answer.",
        "ref": "Enumerate all 6! = 720 permutations and keep those satisfying all four constraints. Exactly 24 survive. Answer = 24.",
    },
    {
        "id": "tank_drain",
        "answer": "7.2",
        "q": "A cylindrical tank (diameter 2.0 m, height 3.0 m) is 80% full of liquid (density 850 kg/m^3). A pump drains it at 12 L/s. How many minutes (to one decimal place) does it take to drain the tank down to 25% full? Give only the final answer.",
        "ref": "Volume = pi*(1.0)^2*(3.0) = 9.4248 m3. 80% = 7.5398 m3, 25% = 2.3562 m3. Drain volume = 5.1836 m3. At 12 L/s = 0.012 m3/s, time = 5.1836/0.012 = 431.97 s = 7.2 minutes.",
    },
    {
        "id": "bayes_flip",
        "answer": "50.8%",
        "q": "Three machines make parts: A makes 50% of parts with 1% defective, B makes 30% with 3% defective, C makes 20% with 6% defective. A part is tested and found to be NOT defective. What is the probability (as a percentage, to one decimal place) that it came from machine A? Give only the final answer.",
        "ref": "P(good|A)=0.99, P(good|B)=0.97, P(good|C)=0.94. P(A|good) = (0.99)(0.5) / [(0.99)(0.5) + (0.97)(0.3) + (0.94)(0.2)] = 0.495 / 0.974 = 0.5082 = 50.8%.",
    },
]


async def chat(client, messages, extra_body=None):
    payload = {"model": CHEAP_MODEL, "messages": messages, "max_tokens": 1500, "temperature": 0.7}
    if extra_body:
        payload.update(extra_body)
    for attempt in range(6):
        start = time.time()
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
    print("=" * 78)
    print("HARD-REASONING CACHE A/B — cheap model vs cheap+cache, correctness vs KNOWN answer")
    print(f"  cheap model : {CHEAP_MODEL} (reasoning off)")
    print(f"  judge       : ME (verified answers 0 / 3 / 24 / 7.2 / 50.8%)")
    print("=" * 78)

    async with httpx.AsyncClient(timeout=120) as client:
        for item in QUESTIONS:
            print(f"\n{'─' * 78}")
            print(f"[{item['id']}]  KNOWN ANSWER = {item['answer']}")
            print(f"Q: {item['q'][:110]}...")
            print(f"{'─' * 78}")

            direct = await chat(client, [{"role": "user", "content": item["q"]}], OR_REASONING_OFF)
            cached = await chat(client, [{"role": "user", "content": (
                f"[Reference — background only, may be outdated, do not cite]\n{item['ref']}\n\n{item['q']}"
            )}], OR_REASONING_OFF)

            print(f"\n  DIRECT (no cache):\n  {direct[:500]}")
            print(f"\n  CACHED (with ref):\n  {cached[:500]}")
            print()

    print("=" * 78)
    print("DONE — compare each answer above against its KNOWN ANSWER to grade.")


if __name__ == "__main__":
    asyncio.run(run())
