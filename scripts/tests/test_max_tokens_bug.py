"""Reproduce the max_tokens-drop bug: force a long generation through the
cheap agent path and measure where the output actually stops.

Run BEFORE the fix (expect ~2048-token cut) and AFTER (expect >2048).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import _run_cheap_agent

PROMPT = (
    "Write a very long, detailed short story (aim for 3000+ words) about a "
    "Malaysian instrumentation engineer in Gebeng who finds a mysterious "
    "4-20mA transmitter that predicts factory downtime. Include full dialogue, "
    "scene descriptions, and a complete ending. Do not stop early — the story "
    "must reach a proper conclusion."
)


async def main() -> None:
    messages = [
        {"role": "system", "content": "You are a creative fiction writer. Always finish your story completely."},
        {"role": "user", "content": PROMPT},
    ]
    print("Calling _run_cheap_agent(max_tokens=8192, reasoning=True) ...", flush=True)
    out = await _run_cheap_agent(
        "qwen/qwen3.7-flash",
        messages,
        temperature=0.3,
        max_tokens=8192,
        tools=[],
        reasoning=True,
    )
    words = len(out.split())
    chars = len(out)
    print(f"OUTPUT length: {chars} chars, {words} words")
    print(f"Last 200 chars: ...{out[-200:]!r}")
    # Rough token estimate: ~4 chars/token for English
    est_tokens = chars / 4
    print(f"Estimated tokens: ~{est_tokens:.0f}")
    if words < 400:
        print("VERDICT: SUSPICIOUSLY SHORT — looks truncated")
    else:
        print("VERDICT: full-length output")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
