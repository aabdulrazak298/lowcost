"""Test: Does the cheap model improve when using cached PRO answers?

Test flow for each question:
  A. Cheap model answers directly (no cache) → baseline
  B. PRO answers → cached
  C. Cheap model answers with PRO's cached answer as reference → cached-enhanced
  D. Judge compares A vs C
"""
import asyncio
import httpx
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path.home() / ".hermes" / ".env", override=False)
OR_KEY = os.getenv("OPENROUTER_API_KEY", "")
DS_KEY = os.getenv("EXPENSIVE_API_KEY", "")
OR_BASE = "https://openrouter.ai/api/v1"
DS_BASE = "https://api.deepseek.com/v1"

OR_HEADERS = {
    "Authorization": f"Bearer {OR_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8800",
    "X-Title": "LowCostLLM-Test",
}

QUERY_PAIRS = [
    {
        "id": "4-20ma",
        "base": "Write a Python function to convert a 4-20mA analog reading to engineering units (0-100°C) with proper error handling for open-circuit detection.",
        "similar": "Take that 4-20mA function and add a 5-sample moving average filter, plus support for different analog ranges (0-10V, 0-5V) with user-configurable engineering units.",
    },
    {
        "id": "pid",
        "base": "Explain PID control for industrial temperature regulation with anti-windup and provide the mathematical formulas.",
        "similar": "Extend the PID explanation to include feedforward control, cascade loop configuration, and how to handle actuator saturation limits beyond just anti-windup.",
    },
    {
        "id": "modbus",
        "base": "How do you troubleshoot intermittent Modbus RTU CRC errors in an industrial setting?",
        "similar": "Add to the Modbus troubleshooting guide: how to detect ground loops with proper testing equipment, and when to replace RS-485 transceivers vs just adding termination resistors.",
    },
]


async def query_openrouter(model: str, messages: list, max_tok: int = 1500) -> dict:
    import time
    start = time.time()
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"{OR_BASE}/chat/completions",
            headers=OR_HEADERS,
            json={"model": model, "messages": messages, "max_tokens": max_tok},
        )
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.time() - start
    return {
        "text": data["choices"][0]["message"]["content"],
        "time": round(elapsed, 1),
        "model": data.get("model", model),
    }


async def query_deepseek(model: str, messages: list, max_tok: int = 1500) -> dict:
    import time
    start = time.time()
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"{DS_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {DS_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": max_tok},
        )
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.time() - start
    return {
        "text": data["choices"][0]["message"]["content"],
        "time": round(elapsed, 1),
        "model": data.get("model", model),
    }


async def judge(question_a: str, question_b: str, answer_without: str, answer_with: str) -> str:
    """Judge: is answer_with (cached-enhanced) better than answer_without (direct)?"""
    prompt = f"""You are evaluating whether a cached expert response helps a cheaper model produce better answers.

The user first asked a BASE question. An expert model (DeepSeek V4 Pro) answered it. The answer was cached.

Then the user asked a SIMILAR but EXTENDED question. The cheap model (DeepSeek V4 Flash) answered it in TWO ways:

ANSWER A (no cache): The cheap model answered the extended question directly, without seeing the expert answer.

ANSWER B (with cache): The cheap model was shown the expert answer to the BASE question, then asked to adapt it for the extended question.

ORIGINAL BASE QUESTION: {question_a}

EXTENDED QUESTION: {question_b}

ANSWER A (cheap, no cache):
{answer_without}

ANSWER B (cheap, with cache):
{answer_with}

Please evaluate:
1. Is Answer B better than Answer A? Why or why not?
2. Does Answer B show better depth, accuracy, or structure by building on the cached expert answer?
3. Does Answer B correctly incorporate the NEW requirements from the extended question, or does it just repeat the expert answer?
4. Overall verdict: Does the cache improve quality? (Yes / No / About the same)

Be specific with examples."""

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"{OR_BASE}/chat/completions",
            headers=OR_HEADERS,
            json={"model": "google/gemini-2.5-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 800},
        )
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def run_test():
    CHEAP = "deepseek/deepseek-v4-flash"
    PRO = "deepseek/deepseek-v4-pro"  # via OpenRouter for reliability
    SYS = "You are an expert industrial automation engineer. Answer thoroughly."
    
    # Quick sanity check
    print(f"DS_KEY: {'✓' if DS_KEY and len(DS_KEY)>20 else '✗ EMPTY'}")
    print(f"OR_KEY: {'✓' if OR_KEY and len(OR_KEY)>20 else '✗ EMPTY'}")

    print("=" * 70)
    print("CACHE QUALITY TEST: Does cached PRO answer improve cheap model?")
    print(f"Cheap: {CHEAP} (OpenRouter)")
    print(f"Pro:   {PRO} (native DeepSeek API)")
    print("=" * 70)

    for idx, pair in enumerate(QUERY_PAIRS, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {idx}/{len(QUERY_PAIRS)}: {pair['id']}")
        print(f"{'─' * 70}")

        # Step A: Cheap model answers extended question directly (no cache)
        print(f"  A. Cheap direct (no cache)...", end=" ", flush=True)
        direct = await query_openrouter(CHEAP, [
            {"role": "system", "content": SYS},
            {"role": "user", "content": pair["similar"]},
        ])
        print(f"{len(direct['text'])}c, {direct['time']}s")

        # Step B: PRO answers base question → "cached"
        print(f"  B. PRO answers base...", end=" ", flush=True)
        expert = await query_openrouter(PRO, [
            {"role": "system", "content": SYS},
            {"role": "user", "content": pair["base"]},
        ])
        print(f"{len(expert['text'])}c, {expert['time']}s")

        # Step C: Cheap model with cached PRO answer
        cached_prompt = f"""A similar question was previously answered by an expert AI.
Here is that answer for reference:
---
{expert['text']}
---
Use the expert answer as your knowledge source. Answer the user's question accurately.
If the new question differs from the original, adapt the answer appropriately while
preserving factual accuracy.

User's question: {pair['similar']}"""

        print(f"  C. Cheap with cache...", end=" ", flush=True)
        cached = await query_openrouter(CHEAP, [
            {"role": "system", "content": SYS},
            {"role": "user", "content": cached_prompt},
        ])
        print(f"{len(cached['text'])}c, {cached['time']}s")

        # Step D: Judge
        print(f"  D. Gemini judging...", end=" ", flush=True)
        verdict = await judge(
            pair["base"], pair["similar"],
            direct["text"], cached["text"],
        )
        print("done")

        print(f"\n  ╔══ VERDICT: {pair['id']} ══╗")
        for line in verdict.split("\n"):
            print(f"  ║ {line}")
        print(f"  ╚{'═' * 52}╝")

        print(f"\n  Direct: {len(direct['text'])}c/{direct['time']}s")
        print(f"  Cached: {len(cached['text'])}c/{cached['time']}s")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(run_test())
