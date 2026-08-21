"""Inspect what max_tokens the Agents SDK actually sends for the cheap agent path."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm
from llm import _run_cheap_agent

PROMPT = (
    "Write a very long, detailed short story (aim for 3000+ words) about a "
    "Malaysian instrumentation engineer in Gebeng who finds a mysterious "
    "4-20mA transmitter that predicts factory downtime. Do not stop early — "
    "the story must reach a proper conclusion."
)


async def main() -> None:
    orig = llm._client_for_model
    seen = []

    def spy_client(model_id):
        client = orig(model_id)
        orig_create = client.chat.completions.create

        async def wrapped(*args, **kwargs):
            seen.append(kwargs)
            return await orig_create(*args, **kwargs)

        client.chat.completions.create = wrapped
        return client

    llm._client_for_model = spy_client

    messages = [
        {"role": "system", "content": "You are a creative fiction writer. Always finish your story completely."},
        {"role": "user", "content": PROMPT},
    ]
    out = await _run_cheap_agent(
        "qwen/qwen3.7-flash", messages,
        temperature=0.3, max_tokens=8192, tools=[], reasoning=True,
    )
    print("turns:", len(seen))
    for i, kw in enumerate(seen):
        print(f"turn {i}: max_tokens={kw.get('max_tokens')!r} "
              f"extra_body={kw.get('extra_body')!r} n_msgs={len(kw.get('messages', []))}")
    print("out chars:", len(out), "est tokens:", len(out) / 4)
    print("last 120:", repr(out[-120:]))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
