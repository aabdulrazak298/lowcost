"""A/B test: DeepSeek V4 Flash (cheap) vs DeepSeek V4 Pro (expensive).

Sends the same question to both models, then asks Gemini to judge
if it can tell which answer came from which model.
"""
import asyncio
import httpx
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DS_KEY = os.getenv("EXPENSIVE_API_KEY")
DS_BASE = "https://api.deepseek.com/v1"

# Load OpenRouter key from Hermes global .env for Gemini judge
_hermes_env = Path.home() / ".hermes" / ".env"
if _hermes_env.exists():
    load_dotenv(_hermes_env, override=False)
GEMINI_KEY = os.getenv("OPENROUTER_API_KEY", "")

QUESTIONS = [
    {
        "id": "plc_basics",
        "category": "Industrial",
        "q": "Explain how a PLC scan cycle works and why it's important for industrial automation. Include the difference between synchronous and asynchronous I/O.",
    },
    {
        "id": "pid_control",
        "category": "Control Systems",
        "q": "Design a PID controller for a temperature regulation system in a rubber vulcanization press. Explain each term (P, I, D) and how anti-windup is implemented.",
    },
    {
        "id": "modbus_debug",
        "category": "Industrial/Comm",
        "q": "A Modbus RTU slave is returning CRC errors intermittently. Walk through the systematic troubleshooting steps, from physical layer to software. What tools would you use?",
    },
    {
        "id": "python_script",
        "category": "Programming",
        "q": "Write a Python script that reads 4-20mA sensor data via Modbus TCP, applies a moving average filter, checks for open-circuit (reading < 3.5mA), and logs anomalies to SQLite.",
    },
    {
        "id": "scada_arch",
        "category": "Architecture",
        "q": "Compare the advantages and disadvantages of on-premise SCADA vs cloud-based SCADA for a rubber factory with 200 I/O points. Consider latency, security, cost, and reliability.",
    },
]


async def query_deepseek(model: str, question: str) -> dict:
    """Query a DeepSeek model, return timing + answer."""
    import time
    start = time.time()
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{DS_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {DS_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert industrial automation engineer. Today is July 15, 2026. Answer thoroughly and precisely."},
                    {"role": "user", "content": question},
                ],
                "max_tokens": 1500,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.time() - start
    content = data["choices"][0]["message"]["content"]
    return {
        "text": content,
        "len": len(content),
        "time": round(elapsed, 1),
        "model": data.get("model", model),
        "tokens": data.get("usage", {}),
    }


async def judge_gemini(question: str, answer_a: str, answer_b: str) -> str:
    """Ask DeepSeek V4 Pro to compare two answers (blind)."""
    prompt = f"""You are a judge evaluating two AI responses to the same question.

QUESTION:
{question}

ANSWER X:
{answer_a}

ANSWER Y:
{answer_b}

Please evaluate:
1. Which answer is better? Why?
2. Can you identify which answer came from a higher-end model vs a lower-cost model? What gives it away?
3. Score both answers on a scale of 1-10 for: accuracy, depth, clarity, and practical usefulness.

Be specific and give examples from the text."""

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"{DS_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {DS_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"]


async def run_test():
    print("=" * 70)
    print("A/B TEST: DeepSeek V4 Pro vs DeepSeek V4 Flash")
    print("Judge: Gemini 2.5 Flash")
    print("=" * 70)

    for idx, item in enumerate(QUESTIONS, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {idx}/{len(QUESTIONS)}: {item['category']} — {item['q'][:60]}...")
        print(f"{'─' * 70}")

        # Randomize order so judge doesn't guess by position
        import random
        models = [
            ("deepseek-v4-pro", "PRO"),
            ("deepseek-v4-flash", "FLASH"),
        ]
        random.shuffle(models)

        results = {}
        for model, label in models:
            print(f"  Querying {label} ({model})...", end=" ", flush=True)
            r = await query_deepseek(model, item["q"])
            results[label] = r
            print(f"{r['len']} chars, {r['time']}s, {r['tokens'].get('total_tokens', '?')} tokens")

        # Unshuffle — map X/Y to actual labels
        mapping = {"X": models[0][1], "Y": models[1][1]}

        print(f"  Judge: X={mapping['X']}, Y={mapping['Y']}...", end=" ", flush=True)
        verdict = await judge_gemini(
            item["q"],
            results[models[0][1]]["text"],
            results[models[1][1]]["text"],
        )
        print("done")

        print(f"\n  ╔══ JUDGE VERDICT ({item['category']}) ══╗")
        for line in verdict.split("\n"):
            print(f"  ║ {line}")
        print(f"  ╚{'═' * 50}╝")

        # Summary stats
        print(f"\n  Stats: FLASH={results['FLASH']['len']}c/{results['FLASH']['time']}s, "
              f"PRO={results['PRO']['len']}c/{results['PRO']['time']}s")

    print(f"\n{'=' * 70}")
    print("ALL TESTS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(run_test())
