"""Code-interpreter + reasoning A/B: does giving the cheap model a run_code tool
(plus reasoning) close the hard/variant gap?

Implements a manual agentic loop: send the question with a run_code tool, execute
any tool call (local Python subprocess), feed stdout back, repeat until a final
answer. Reasoning ON via extra_body.

Graded against in-script ground truth. Model via TEST_CHEAP_MODEL.

Run: cd ~/cloud/projects/lowcostllm && .venv/bin/python tests/test_code_reasoning.py
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
    "X-Title": "LowCostLLM-CodeReason",
}
CHEAP_MODEL = os.getenv("TEST_CHEAP_MODEL", "qwen/qwen3.7-flash")
REASONING_ON = os.getenv("REASONING", "0") == "1"
_EXTRA = {"reasoning": {"enabled": True}} if REASONING_ON else {"reasoning": {"enabled": False}}
_MAX_TOK = 8000

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_code",
        "description": "Execute Python code and return its stdout. Use for calculations, brute-force enumeration, counting, or any multi-step computation.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute. print() the result."}
            },
            "required": ["code"],
        },
    },
}]


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


QUESTIONS = [
    {"id": "bayes", "answer": solve_bayes(0.60, 0.02, 0.25, 0.04, 0.15, 0.08),
     "q": "Three machines make parts: A makes 60% with 2% defective, B 25% with 4% defective, C 15% with 8% defective. A part is found NOT defective. What is the probability (percentage, one decimal) it came from machine A? Give only the final answer."},
    {"id": "tank", "answer": solve_tank(1.5, 2.5, 90, 30, 8),
     "q": "A cylindrical tank (diameter 1.5 m, height 2.5 m) is 90% full. A pump drains it at 8 L/s. How many minutes (one decimal) to drain down to 30% full? Give only the final answer."},
    {"id": "perm", "answer": solve_perm([
        lambda p: p.index("A") < p.index("B"),
        lambda p: p.index("B") < p.index("C"),
        lambda p: p[0] != "C",
        lambda p: p.index("D") == p.index("E") + 1,
        lambda p: p.index("F") < p.index("D")]),
     "q": "Six people A, B, C, D, E, F line up in a row. How many valid orderings satisfy ALL of: A before B; B before C; C not first; D immediately after E; F before D? Give only the final answer."},
    {"id": "grid", "answer": solve_grid(6),
     "q": "A car navigates a 6x6 grid from (0,0) to (5,5), moving only Right (+x) or Up (+y). When on an even x-coordinate it cannot move Up, when on an odd y-coordinate it cannot move Right. How many distinct shortest paths exist? Give only the final answer."},
    {"id": "bayes2", "answer": round((0.03 * 0.80) / (0.03 * 0.80 + 0.10 * 0.20) * 100, 1),
     "q": "Two machines: X makes 80% of parts with 3% defective, Y makes 20% with 10% defective. A part is found defective. What is the probability (percentage, one decimal) it came from X? Give only the final answer."},
]


def execute_code(code: str) -> str:
    """Run model-authored Python in a subprocess; return stdout+stderr (capped)."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=15,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return out.strip()[:2000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(exec error: {e})"


async def agentic_solve(client, question: str) -> tuple[str, int]:
    """Loop: ask -> execute tool calls -> feed back -> final answer."""
    messages = [{"role": "user", "content": question}]
    n_calls = 0
    for _ in range(8):
        payload = {
            "model": CHEAP_MODEL, "messages": messages,
            "max_tokens": _MAX_TOK, "temperature": 0.7, "tools": TOOLS,
        }
        if _EXTRA:
            payload.update(_EXTRA)
        resp = await client.post(f"{OR_BASE}/chat/completions", headers=OR_HEADERS, json=payload)
        if resp.status_code in (429, 500, 502, 503, 504):
            await asyncio.sleep(3)
            continue
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if tcs:
            # record assistant message with tool_calls
            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tcs,
            })
            for tc in tcs:
                if tc.get("function", {}).get("name") != "run_code":
                    continue
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                    code = args.get("code", "")
                except Exception:
                    code = ""
                result = execute_code(code)
                n_calls += 1
                messages.append({"role": "tool", "tool_call_id": tc.get("id", "0"), "content": result})
            continue
        # final answer
        return (msg.get("content") or "").strip(), n_calls
    return "", n_calls


async def run():
    print("=" * 82)
    print("CODE-INTERPRETER + REASONING — does run_code close the gap?")
    print(f"  model : {CHEAP_MODEL}  reasoning={'ON' if REASONING_ON else 'OFF'}  tool=run_code")
    print("=" * 82)

    async with httpx.AsyncClient(timeout=300) as client:
        for item in QUESTIONS:
            print(f"\n[{item['id']}]  CORRECT = {item['answer']}")
            t0 = time.time()
            answer, n = await agentic_solve(client, item["q"])
            dt = time.time() - t0
            print(f"  tool calls: {n}   time: {dt:.0f}s")
            print(f"  ANSWER: {answer[:250]}")
            print()

    print("=" * 82)
    print("DONE — grade each ANSWER against CORRECT.")


if __name__ == "__main__":
    asyncio.run(run())
