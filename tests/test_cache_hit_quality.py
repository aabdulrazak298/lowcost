"""A/B test: does a cache hit still improve the cheap model's answer?

Refreshed for the current (Aug 2026) production design, where the cached
expert answer is DEMOTED to a "[Reference — background only, may be
outdated, do not cite]" block instead of being used as the primary
knowledge source.

Per question, three things happen:
  1. EXPERT  : deepseek-v4-pro answers (this is what a cache miss stores).
  2. DIRECT  : current cheap model answers the SAME question with no reference.
  3. CACHED  : current cheap model answers with the expert answer injected as a
               [Reference] block (mirrors processor.py cheap path exactly).
  4. JUDGE   : Gemini (blind) scores DIRECT vs CACHED on accuracy/completeness/
               clarity and picks a winner.

The ONLY difference between DIRECT and CACHED is the [Reference] block, so
any score delta is attributable to the cache. Position is randomised so the
judge can't guess by order.

Run:  cd ~/cloud/projects/lowcostllm && venv/bin/python tests/test_cache_hit_quality.py
      (optionally: TEST_CHEAP_MODEL=deepseek/deepseek-v4-flash venv/bin/python ...)
"""
import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load project .env first, then Hermes global .env (for OPENROUTER_API_KEY).
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
load_dotenv(Path.home() / ".hermes" / ".env", override=False)

# Disk cache for expert answers — pro is slow (80-180s reasoning) and the expert
# answer does NOT depend on the cheap model under test, so reuse across runs.
_EXPERT_CACHE = Path(__file__).parent / ".expert_cache.json"


def _load_expert_cache() -> dict:
    try:
        return json.loads(_EXPERT_CACHE.read_text())
    except Exception:
        return {}


def _save_expert_cache(d: dict) -> None:
    _EXPERT_CACHE.write_text(json.dumps(d))

DS_KEY = os.getenv("EXPENSIVE_API_KEY", "")
OR_KEY = os.getenv("OPENROUTER_API_KEY", "")
DS_BASE = "https://api.deepseek.com/v1"
OR_BASE = "https://openrouter.ai/api/v1"

# Expert reference model — the cache is supposed to hold EXPENSIVE (pro) output.
EXPERT_MODEL = "deepseek-v4-pro"

# Cheap model under test: current /model -c override, or TEST_CHEAP_MODEL env.
import sqlite3 as _sqlite

def _current_cheap_override() -> str | None:
    db = Path(__file__).resolve().parents[1] / "cache.db"
    try:
        c = _sqlite.connect(db)
        c.row_factory = _sqlite.Row
        row = c.execute("SELECT cheap_override FROM model_overrides WHERE id=1").fetchone()
        c.close()
        return (row["cheap_override"] if row else None)
    except Exception:
        return None

CHEAP_MODEL = os.getenv("TEST_CHEAP_MODEL", "") or _current_cheap_override() or "deepseek/deepseek-v4-flash"

OR_HEADERS = {
    "Authorization": f"Bearer {OR_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8800",
    "X-Title": "LowCostLLM-CacheAB",
}

# Genuinely hard, VERIFIED reasoning questions (answers brute-forced/computed):
#   box_coin=0, grid_path=3, perm_count=24, tank_drain=7.2, bayes_flip=50.8%
QUESTIONS = [
    {
        "id": "box_coin",
        "q": "Three opaque boxes A, B, C hold three distinct coins: a 50-gram silver, a 100-gram silver, and a 100-gram lead coin (one coin per box). The boxes are labelled 'Silver 50', 'Silver 100', and 'Lead 100', but all three labels are wrong. You draw one coin from Box A and it is silver. What is the probability that Box C holds the 100-gram silver coin?",
    },
    {
        "id": "grid_path",
        "q": "A car navigates a 5x5 grid from (0,0) to (4,4), moving only Right (+x) or Up (+y), one cell at a time. But when the car is on an even x-coordinate it cannot move Up, and when it is on an odd y-coordinate it cannot move Right. How many distinct shortest paths exist?",
    },
    {
        "id": "perm_count",
        "q": "Six people A, B, C, D, E, F line up in a single row. How many valid orderings satisfy ALL of: A is before B; C is not first; D is immediately after E; F is before D?",
    },
    {
        "id": "tank_drain",
        "q": "A cylindrical tank (diameter 2.0 m, height 3.0 m) is 80% full of liquid (density 850 kg/m^3). A pump drains it at 12 L/s. How many minutes (to one decimal place) does it take to drain the tank down to 25% full?",
    },
    {
        "id": "bayes_flip",
        "q": "Three machines make parts: A makes 50% of parts with 1% defective, B makes 30% with 3% defective, C makes 20% with 6% defective. A part is tested and found to be NOT defective. What is the probability (as a percentage, to one decimal place) that it came from machine A?",
    },
]

DATE_CONTEXT = f"Today's date is {datetime.now().strftime('%A, %d %B %Y')}."

SYSTEM_NO_REF = f"You are a helpful assistant. {DATE_CONTEXT}"

# qwen3.7-flash (and friends) FORCE reasoning on OpenRouter; reasoning eats the
# token budget and returns content=None. Disable it for the cheap model — a cheap
# model shouldn't be burning tokens/time on thinking anyway. This is the same
# fix documented in the lowcostllm skill (extra_body={'reasoning':{'enabled':False}}).
OR_REASONING_OFF = {"reasoning": {"enabled": False}}


async def chat(client, base, key, headers, model, messages, max_tok=1500, extra_body=None):
    """Single chat completion.

    Retries on 429/5xx with exponential backoff (OpenRouter rate-limits), and
    once more on empty/None content (some forced-reasoning models return
    content=None when reasoning eats the token budget).
    """
    payload = {"model": model, "messages": messages, "max_tokens": max_tok, "temperature": 0.7}
    if extra_body:
        # OpenRouter raw HTTP: extra params go at TOP LEVEL (the OpenAI SDK's
        # extra_body= kwarg is merged into the body client-side; it is NOT a
        # field named "extra_body").
        payload.update(extra_body)

    def _once():
        resp = client.post(f"{base}/chat/completions", headers=headers, json=payload)
        return resp

    # Outer: retry on 429/5xx with backoff.
    for attempt in range(6):
        start = time.time()
        resp = await _once()
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt
            print(f"    (HTTP {resp.status_code} — backing off {wait}s)", flush=True)
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - start
        text = (data["choices"][0]["message"].get("content") or "").strip()
        if text:
            return {
                "text": text,
                "time": round(elapsed, 1),
                "model": data.get("model", model),
                "finish": data["choices"][0].get("finish_reason"),
            }
        # Empty content: retry once (fresh request) if budget not exhausted.
        if attempt < 5:
            await asyncio.sleep(1)
            continue
        return {
            "text": "",
            "time": round(elapsed, 1),
            "model": data.get("model", model),
            "finish": data["choices"][0].get("finish_reason"),
        }
    raise RuntimeError(f"{model} failed after retries (429/5xx)")


async def run():
    print("=" * 72)
    print("CACHE-HIT QUALITY A/B  —  does [Reference] still improve cheap answer?")
    print(f"  cheap model under test : {CHEAP_MODEL}")
    print(f"  expert (cache) source  : {EXPERT_MODEL}")
    print(f"  judge                  : deepseek-v4-pro (correctness vs reference)")
    print("=" * 72)

    ds_headers = {"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"}

    agg = []  # (id, score_direct, score_cached, winner)

    async with httpx.AsyncClient(timeout=180) as client:
        for idx, item in enumerate(QUESTIONS, 1):
            print(f"\n{'─' * 72}")
            print(f"Q{idx}/{len(QUESTIONS)} [{item['id']}] {item['q'][:64]}...")
            print(f"{'─' * 72}")

            # 1. Expert answer (cache content) — max_tokens must be high: pro is a
            #    reasoning model and reasoning_content counts against the budget.
            ecache = _load_expert_cache()
            if item["id"] in ecache and ecache[item["id"]].strip():
                expert = {"text": ecache[item["id"]], "time": 0, "model": EXPERT_MODEL}
                print(f"  1. expert (disk cache)... {len(expert['text'])}c")
            else:
                print("  1. expert (pro)...", end=" ", flush=True)
                expert = await chat(client, DS_BASE, DS_KEY, ds_headers,
                                    EXPERT_MODEL, [{"role": "user", "content": item["q"]}],
                                    max_tok=8192)
                if not expert["text"].strip():
                    print("EMPTY — retrying")
                    expert = await chat(client, DS_BASE, DS_KEY, ds_headers,
                                        EXPERT_MODEL, [{"role": "user", "content": item["q"]}],
                                        max_tok=8192)
                ecache[item["id"]] = expert["text"]
                _save_expert_cache(ecache)
                print(f"{len(expert['text'])}c, {expert['time']}s")

            # 2 & 3. Direct (no ref) vs Cached (with ref) — randomise order.
            direct_msgs = [
                {"role": "system", "content": SYSTEM_NO_REF},
                {"role": "user", "content": item["q"]},
            ]
            cached_msgs = [
                {"role": "system", "content": SYSTEM_NO_REF},
                {"role": "user", "content": (
                    f"[Reference — background only, may be outdated, do not cite]\n"
                    f"{expert['text']}\n\n"
                    f"{item['q']}"
                )},
            ]
            print("  2. cheap DIRECT (no cache)...", end=" ", flush=True)
            direct = await chat(client, OR_BASE, OR_KEY, OR_HEADERS,
                                CHEAP_MODEL, direct_msgs, extra_body=OR_REASONING_OFF)
            print(f"{len(direct['text'])}c, {direct['time']}s")

            print("  3. cheap CACHED (with ref)...", end=" ", flush=True)
            cached = await chat(client, OR_BASE, OR_KEY, OR_HEADERS,
                                CHEAP_MODEL, cached_msgs, extra_body=OR_REASONING_OFF)
            print(f"{len(cached['text'])}c, {cached['time']}s")

            # Blind judge — shuffle A/B labels.
            if random.random() < 0.5:
                label_a, label_b = "DIRECT", "CACHED"
                ans_a, ans_b = direct["text"], cached["text"]
            else:
                label_a, label_b = "CACHED", "DIRECT"
                ans_a, ans_b = cached["text"], direct["text"]

            judge_prompt = f"""You are grading two candidate answers to a reasoning question. A reference solution is provided; assume it is CORRECT.

QUESTION:
{item['q']}

REFERENCE SOLUTION (assume correct):
{expert['text']}

ANSWER A:
{ans_a}

ANSWER B:
{ans_b}

For EACH answer, grade CORRECTNESS only (does it reach the same correct final
answer/conclusion as the reference — even if worded differently or shorter).
Ignore style, verbosity, and tone. 10 = correct final answer; 0 = wrong final
answer. Output EXACTLY in this format, nothing else:

Score A: <n>/10
Score B: <n>/10
Winner: A or B or TIE
Reason: <one sentence — which answer(s) got the correct final answer>"""

            print("  4. judge (pro)...", end=" ", flush=True)
            jr = await client.post(
                f"{DS_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "max_tokens": 8192,
                },
            )
            jr.raise_for_status()
            verdict = jr.json()["choices"][0]["message"]["content"]
            print("done")

            # Parse scores + winner, mapping back to DIRECT/CACHED.
            import re
            sa = sb = None
            m = re.search(r"Score A:\s*(\d+)", verdict)
            if m:
                sa = int(m.group(1))
            m = re.search(r"Score B:\s*(\d+)", verdict)
            if m:
                sb = int(m.group(1))
            winner_raw = re.search(r"Winner:\s*([ABC]+|TIE)", verdict)
            winner_raw = winner_raw.group(1) if winner_raw else "?"

            score_direct = sa if label_a == "DIRECT" else sb
            score_cached = sb if label_a == "DIRECT" else sa
            if winner_raw == "TIE":
                winner = "TIE"
            elif winner_raw == "A":
                winner = label_a
            elif winner_raw == "B":
                winner = label_b
            else:
                winner = "?"

            agg.append((item["id"], score_direct, score_cached, winner))

            print(f"\n  ╔══ VERDICT [{item['id']}] ══╗")
            for line in verdict.split("\n"):
                print(f"  ║ {line}")
            print(f"  ╚{'═' * 54}╝")
            print(f"  → DIRECT={score_direct}/10  CACHED={score_cached}/10  winner={winner}")

    # Aggregate
    print(f"\n{'=' * 72}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 72}")
    n = len(agg)
    n_win = sum(1 for a in agg if a[3] == "CACHED")
    n_tie = sum(1 for a in agg if a[3] == "TIE")
    n_direct = sum(1 for a in agg if a[3] == "DIRECT")
    scores_d = [a[1] for a in agg if a[1] is not None]
    scores_c = [a[2] for a in agg if a[2] is not None]
    avg_d = sum(scores_d) / len(scores_d) if scores_d else 0
    avg_c = sum(scores_c) / len(scores_c) if scores_c else 0

    for a in agg:
        d = f"{a[1]}/10" if a[1] is not None else "?"
        c = f"{a[2]}/10" if a[2] is not None else "?"
        print(f"  {a[0]:<16} DIRECT={d:>5}  CACHED={c:>5}  winner={a[3]}")

    print(f"\n  CACHED wins  : {n_win}/{n}")
    print(f"  DIRECT wins  : {n_direct}/{n}")
    print(f"  TIE          : {n_tie}/{n}")
    print(f"  avg DIRECT   : {avg_d:.2f}/10")
    print(f"  avg CACHED   : {avg_c:.2f}/10  (Δ {avg_c - avg_d:+.2f})")
    print(f"\n  Bottom line: {'cache STILL improves cheap answer' if (avg_c > avg_d and n_win >= n_direct) else 'cache does NOT clearly improve cheap answer'}")


if __name__ == "__main__":
    asyncio.run(run())
