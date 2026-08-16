"""Combined A/B: worked example (showing CODE) + run_code tool, reasoning OFF.

The hypothesis: a worked example that shows the exact Python pattern, plus the
run_code tool to execute it, lets the cheap model solve variants WITHOUT paying
for reasoning. Contrast with: reasoning ON + run_code (4/5), example-only (0/5).

Arms:
  EXAMPLE+TOOL = variant + worked example (with code) + run_code tool available,
                 reasoning OFF, with an explicit "call the tool" steer.

Graded against in-script ground truth.

Run: cd ~/cloud/projects/lowcostllm && .venv/bin/python tests/test_combo.py
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
OR_BASE = "https://openrouter.ai/api/v1"
OR_HEADERS = {
    "Authorization": f"Bearer {OR_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8800",
    "X-Title": "LowCostLLM-Combo",
}
CHEAP_MODEL = os.getenv("TEST_CHEAP_MODEL", "qwen/qwen3.7-flash")
REASONING_ON = os.getenv("REASONING", "0") == "1"
_REASONING = {"reasoning": {"enabled": True}} if REASONING_ON else {"reasoning": {"enabled": False}}
_MAX_TOK = 8000 if REASONING_ON else 4000

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_code",
        "description": "Execute Python code and return its stdout. Use for any calculation or enumeration.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute. print() the result."}
            },
            "required": ["code"],
        },
    },
}]

STEER = (
    "IMPORTANT: You have a `run_code` tool. To answer a computation or enumeration "
    "question, write Python and CALL the run_code tool (do NOT write code as plain "
    "text, do NOT compute in your head). Run the code, read the output, then give "
    "the final answer."
)


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
     "example": ("EXAMPLE (solved with Python):\n"
                 "Q: Three machines A 50% (1% def), B 30% (3%), C 20% (6%). NOT defective -> P(A)?\n"
                 "Code:\n"
                 "```python\npa,pb,pc = 0.5,0.3,0.2\nda,db,dc = 0.01,0.03,0.06\np_good = (1-da)*pa + (1-db)*pb + (1-dc)*pc\nprint(round((1-da)*pa/p_good*100,1))\n```\n"
                 "Output: 50.8"),
     "variant_q": "Three machines: A makes 60% with 2% defective, B 25% with 4%, C 15% with 8%. A part is NOT defective. Probability (percentage, one decimal) it came from A?"},
    {"id": "tank", "answer": solve_tank(1.5, 2.5, 90, 30, 8),
     "example": ("EXAMPLE (solved with Python):\n"
                 "Q: Tank d=2.0m h=3.0m, 80% full, drain 12 L/s to 25%. Minutes?\n"
                 "Code:\n"
                 "```python\nimport math\nV = math.pi*(2.0/2)**2*3.0\ndV = V*(80-25)/100\nprint(round(dV/(12/1000)/60,1))\n```\n"
                 "Output: 7.2"),
     "variant_q": "A cylindrical tank (diameter 1.5 m, height 2.5 m) is 90% full. A pump drains it at 8 L/s. How many minutes (one decimal) to drain down to 30% full?"},
    {"id": "bayes2", "answer": round((0.03 * 0.80) / (0.03 * 0.80 + 0.10 * 0.20) * 100, 1),
     "example": ("EXAMPLE (solved with Python):\n"
                 "Q: Machine A 70% (2% def), B 30% (5%). Part defective -> P(A)?\n"
                 "Code:\n"
                 "```python\npa,pb = 0.7,0.3\nda,db = 0.02,0.05\nprint(round(da*pa/(da*pa+db*pb)*100,1))\n```\n"
                 "Output: 48.3"),
     "variant_q": "Two machines: X makes 80% with 3% defective, Y 20% with 10% defective. A part is defective. Probability (percentage, one decimal) it came from X?"},
    {"id": "perm", "answer": solve_perm([
        lambda p: p.index("A") < p.index("B"),
        lambda p: p.index("B") < p.index("C"),
        lambda p: p[0] != "C",
        lambda p: p.index("D") == p.index("E") + 1,
        lambda p: p.index("F") < p.index("D")]),
     "example": ("EXAMPLE (solved with Python):\n"
                 "Q: 6 people A-F. A before B; C not first; D immediately after E; F before D. Count orderings.\n"
                 "Code:\n"
                 "```python\nimport itertools\nn=0\nfor p in itertools.permutations('ABCDEF'):\n    p=list(p)\n    if p.index('A')<p.index('B') and p[0]!='C' and p.index('D')==p.index('E')+1 and p.index('F')<p.index('D'):\n        n+=1\nprint(n)\n```\n"
                 "Output: 24"),
     "variant_q": "Six people A-F line up. How many orderings satisfy: A before B; B before C; C not first; D immediately after E; F before D?"},
    {"id": "grid", "answer": solve_grid(6),
     "example": ("EXAMPLE (solved with Python):\n"
                 "Q: 5x5 grid (0,0)->(4,4), Right/Up; even x no Up, odd y no Right. Count shortest paths.\n"
                 "Code:\n"
                 "```python\nfrom functools import lru_cache\n@lru_cache(None)\ndef go(x,y):\n    if (x,y)==(4,4): return 1\n    if x>4 or y>4: return 0\n    t=0\n    if y%2==0: t+=go(x+1,y)\n    if x%2==1: t+=go(x,y+1)\n    return t\nprint(go(0,0))\n```\n"
                 "Output: 3"),
     "variant_q": "A car navigates a 6x6 grid (0,0) to (5,5), Right/Up only. Even x cannot go Up, odd y cannot go Right. How many distinct shortest paths?"},
]


def execute_code(code: str) -> str:
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
        return ((r.stdout or "") + (r.stderr or "")).strip()[:2000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(exec error: {e})"


async def solve(client, item) -> tuple[str, int]:
    messages = [{
        "role": "user",
        "content": f"{item['example']}\n\n{STEER}\n\nNow solve:\n{item['variant_q']}",
    }]
    n = 0
    for _ in range(8):
        payload = {"model": CHEAP_MODEL, "messages": messages, "max_tokens": _MAX_TOK,
                   "temperature": 0.7, "tools": TOOLS}
        if _REASONING:
            payload.update(_REASONING)
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
    print("COMBINED — worked example (with code) + run_code, reasoning OFF")
    print(f"  model : {CHEAP_MODEL}")
    print("=" * 82)
    async with httpx.AsyncClient(timeout=300) as client:
        for item in PAIRS:
            print(f"\n[{item['id']}]  CORRECT = {item['answer']}")
            t0 = time.time()
            ans, n = await solve(client, item)
            dt = time.time() - t0
            print(f"  tool calls: {n}   time: {dt:.0f}s")
            print(f"  ANSWER: {ans[:200]}")
            print()
    print("=" * 82)
    print("DONE — grade ANSWER vs CORRECT.")


if __name__ == "__main__":
    asyncio.run(run())
