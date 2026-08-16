"""Real-cache-as-example test: are PRO's RAW answers clean enough to use as
few-shot examples (vs my earlier hand-crafted code snippets)?

Phase 1: ask deepseek-v4-pro each BASE question and store its RAW answer
         (this is exactly what insert_qa stores = call_expensive output).
Phase 2: use that raw answer as the few-shot EXAMPLE for the cheap model
         (qwen3.7-flash, reasoning ON + run_code) on the VARIANT question.
Phase 3: grade vs in-script ground truth.

Run: cd ~/cloud/projects/lowcostllm && .venv/bin/python tests/test_realcache.py
"""
import asyncio
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
load_dotenv(Path.home() / ".hermes" / ".env", override=False)

OR_KEY = os.getenv("OPENROUTER_API_KEY", "")
DS_KEY = os.getenv("EXPENSIVE_API_KEY", "")
OR_BASE = "https://openrouter.ai/api/v1"
DS_BASE = "https://api.deepseek.com/v1"
OR_HEADERS = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json",
              "HTTP-Referer": "http://localhost:8800", "X-Title": "LowCostLLM-RealCache"}
DS_HEADERS = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}

CHEAP_MODEL = os.getenv("TEST_CHEAP_MODEL", "qwen/qwen3.7-flash")
PRO_MODEL = "deepseek-v4-pro"
_PRO_CACHE = Path(__file__).parent / ".pro_base_answers.json"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_code",
        "description": "Execute Python code and return its stdout. Use for any calculation or enumeration.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute. print() the result."}},
            "required": ["code"],
        },
    },
}]

STEER = ("IMPORTANT: You have a `run_code` tool. To answer a computation/enumeration "
         "question, write Python and CALL the run_code tool (do NOT write code as plain "
         "text, do NOT compute in your head).")


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


PAIRS = [
    {"id": "bayes", "answer": solve_bayes(0.60, 0.02, 0.25, 0.04, 0.15, 0.08),
     "base_q": "Three machines make parts: A makes 50% with 1% defective, B 30% with 3%, C 20% with 6%. A part is found NOT defective. What is the probability (percentage, one decimal) it came from machine A? Show your working.",
     "variant_q": "Three machines make parts: A makes 60% with 2% defective, B 25% with 4% defective, C 15% with 8% defective. A part is found NOT defective. What is the probability (percentage, one decimal) it came from machine A?"},
    {"id": "tank", "answer": solve_tank(1.5, 2.5, 90, 30, 8),
     "base_q": "A cylindrical tank (diameter 2.0 m, height 3.0 m) is 80% full of liquid. A pump drains it at 12 L/s. How many minutes (one decimal) to drain down to 25% full? Show your working.",
     "variant_q": "A cylindrical tank (diameter 1.5 m, height 2.5 m) is 90% full of liquid. A pump drains it at 8 L/s. How many minutes (one decimal) to drain down to 30% full?"},
    {"id": "bayes2", "answer": round((0.03 * 0.80) / (0.03 * 0.80 + 0.10 * 0.20) * 100, 1),
     "base_q": "Two machines make parts: A makes 70% with 2% defective, B 30% with 5% defective. A part is found defective. What is the probability (percentage, one decimal) it came from A? Show your working.",
     "variant_q": "Two machines: X makes 80% of parts with 3% defective, Y makes 20% with 10% defective. A part is found defective. What is the probability (percentage, one decimal) it came from X?"},
    {"id": "perm", "answer": solve_perm([
        lambda p: p.index("A") < p.index("B"),
        lambda p: p.index("B") < p.index("C"),
        lambda p: p[0] != "C",
        lambda p: p.index("D") == p.index("E") + 1,
        lambda p: p.index("F") < p.index("D")]),
     "base_q": "Six people A, B, C, D, E, F line up in a row. How many valid orderings satisfy ALL of: A before B; C not first; D immediately after E; F before D? Show your working.",
     "variant_q": "Six people A, B, C, D, E, F line up in a row. How many valid orderings satisfy ALL of: A before B; B before C; C not first; D immediately after E; F before D?"},
    {"id": "grid", "answer": solve_grid(6),
     "base_q": "A car navigates a 5x5 grid from (0,0) to (4,4), moving only Right (+x) or Up (+y). When on an even x-coordinate it cannot move Up, when on an odd y-coordinate it cannot move Right. How many distinct shortest paths exist? Show your working.",
     "variant_q": "A car navigates a 6x6 grid from (0,0) to (5,5), moving only Right (+x) or Up (+y). When on an even x-coordinate it cannot move Up, when on an odd y-coordinate it cannot move Right. How many distinct shortest paths exist?"},
]


def load_pro_cache():
    try:
        return json.loads(_PRO_CACHE.read_text())
    except Exception:
        return {}


def save_pro_cache(d):
    _PRO_CACHE.write_text(json.dumps(d))


async def get_pro_answer(client, item):
    """Ask pro, return raw content (retry on empty — reasoning eats max_tokens)."""
    cache = load_pro_cache()
    if item["id"] in cache and cache[item["id"]].strip():
        return cache[item["id"]]
    for attempt in range(3):
        r = await client.post(f"{DS_BASE}/chat/completions", headers=DS_HEADERS,
                              json={"model": PRO_MODEL,
                                    "messages": [{"role": "user", "content": item["base_q"]}],
                                    "max_tokens": 8192})
        if r.status_code in (429, 500, 502, 503, 504):
            await asyncio.sleep(3)
            continue
        r.raise_for_status()
        data = r.json()
        content = (data["choices"][0]["message"].get("content") or "").strip()
        if content:
            cache[item["id"]] = content
            save_pro_cache(cache)
            return content
        await asyncio.sleep(2)
    return ""


def execute_code(code):
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
        return ((r.stdout or "") + (r.stderr or "")).strip()[:2000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(exec error: {e})"


async def cheap_solve(client, example, variant_q):
    messages = [{"role": "user", "content": (
        f"Here is an EXAMPLE: a similar question and an expert's full solution.\n\n"
        f"EXAMPLE QUESTION:\n{example['base_q']}\n\n"
        f"EXPERT SOLUTION:\n{example['answer']}\n\n"
        f"{STEER}\n\n"
        f"Now solve this new question the same way (its values differ):\n{variant_q}"
    )}]
    n = 0
    for _ in range(8):
        payload = {"model": CHEAP_MODEL, "messages": messages, "max_tokens": 8000,
                   "temperature": 0.7, "tools": TOOLS,
                   "reasoning": {"enabled": True}}
        resp = await client.post(f"{OR_BASE}/chat/completions", headers=OR_HEADERS, json=payload)
        if resp.status_code in (429, 500, 502, 503, 504):
            await asyncio.sleep(3)
            continue
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if tcs:
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                if tc.get("function", {}).get("name") != "run_code":
                    continue
                try:
                    code = json.loads(tc["function"].get("arguments") or "{}").get("code", "")
                except Exception:
                    code = ""
                res = execute_code(code)
                n += 1
                messages.append({"role": "tool", "tool_call_id": tc.get("id", "0"), "content": res})
            continue
        return (msg.get("content") or "").strip(), n
    return "", n


async def run():
    print("=" * 82)
    print("REAL-CACHE-AS-EXAMPLE — do pro's RAW answers work as few-shot examples?")
    print(f"  example source : {PRO_MODEL} raw output (as stored in cache)")
    print(f"  cheap model    : {CHEAP_MODEL} (reasoning ON + run_code)")
    print("=" * 82)

    async with httpx.AsyncClient(timeout=300) as client:
        for item in PAIRS:
            print(f"\n{'─' * 82}")
            print(f"[{item['id']}]  CORRECT = {item['answer']}")
            # Phase 1: pro's raw answer (the "cached" content)
            print("  pro answering base...", end=" ", flush=True)
            t0 = time.time()
            pro_ans = await get_pro_answer(client, item)
            print(f"{len(pro_ans)}c, {time.time()-t0:.0f}s")
            # Phase 2: cheap model + pro's raw answer as example
            ans, n = await cheap_solve(client, {"base_q": item["base_q"], "answer": pro_ans}, item["variant_q"])
            print(f"  tool calls: {n}")
            print(f"  ANSWER: {ans}")
            print()

    print("=" * 82)
    print("DONE — grade ANSWER vs CORRECT.")


if __name__ == "__main__":
    asyncio.run(run())
