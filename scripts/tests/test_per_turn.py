"""Capture per-turn finish_reason + usage in the real cache-hit pipeline to
see if the final answer turn is being cut by an output-length cap."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm
from processor import process_query


async def main() -> None:
    orig = llm._client_for_model
    turns = []

    def spy_client(model_id):
        client = orig(model_id)
        orig_create = client.chat.completions.create

        async def wrapped(*args, **kwargs):
            resp = await orig_create(*args, **kwargs)
            if hasattr(resp, "choices") and resp.choices:
                ch = resp.choices[0]
                turns.append({
                    "max_tokens_sent": kwargs.get("max_tokens"),
                    "finish_reason": getattr(ch, "finish_reason", None),
                    "completion_tokens": getattr(resp, "usage", None) and resp.usage.completion_tokens,
                    "tool_calls": bool(getattr(ch.message, "tool_calls", None)),
                    "msg_chars": len(getattr(ch.message, "content", "") or ""),
                })
            return resp

        client.chat.completions.create = wrapped
        return client

    llm._client_for_model = spy_client

    print("Running pipeline ...", flush=True)
    answer, model_used, _imgs, usage = await process_query(
        user_query="Latest worldwide news reports August 2026",
        chat_history="",
    )
    print("model_used:", model_used)
    print("total usage:", usage)
    print("answer chars:", len(answer))
    print("LAST 250 chars:", repr(answer[-250:]))
    print("\nper-turn:")
    for i, t in enumerate(turns):
        print(f"  turn {i}: max_tokens_sent={t['max_tokens_sent']} "
              f"finish={t['finish_reason']} completion_tok={t['completion_tokens']} "
              f"tool_calls={t['tool_calls']} msg_chars={t['msg_chars']}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
