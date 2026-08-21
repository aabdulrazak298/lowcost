"""Test deepseek-v4-flash (via OpenRouter) as the cheap model on a cache-hit
news query: does it run web_search/web_fetch to refresh stale cached answers?

DB-write functions are no-oped so the test never contends with the live
service's SQLite WAL (database is locked otherwise). In-memory override only.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import llm
import processor

# In-memory cheap override — no DB write
config._cheap_override = "deepseek/deepseek-v4-flash"

# No-op DB writes so we don't collide with the live service's WAL
processor.increment_hit_count = lambda *a, **k: None
processor.record_request = lambda *a, **k: None
processor.cache_store = lambda *a, **k: None
processor.record_irrelevant_escalation = lambda *a, **k: None
processor.run_curator = lambda *a, **k: None


async def main() -> None:
    print("cheap model now:", llm.get_cheap_model(), flush=True)

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
                    "model": model_id,
                    "finish_reason": getattr(ch, "finish_reason", None),
                    "completion_tokens": getattr(resp, "usage", None) and resp.usage.completion_tokens,
                    "tool_calls": bool(getattr(ch.message, "tool_calls", None)),
                    "msg_chars": len(getattr(ch.message, "content", "") or ""),
                })
            return resp

        client.chat.completions.create = wrapped
        return client

    llm._client_for_model = spy_client

    print("Running pipeline (cache-hit news query) ...", flush=True)
    answer = model_used = usage = None
    try:
        answer, model_used, _imgs, usage = await processor.process_query(
            user_query="Latest worldwide news reports August 2026",
            chat_history="",
        )
    except Exception as e:
        print(f"(pipeline raised {type(e).__name__}: {e})", flush=True)

    print("\nper-turn:")
    for i, t in enumerate(turns):
        print(f"  turn {i}: model={t['model']} finish={t['finish_reason']} "
              f"completion_tok={t['completion_tokens']} tool_calls={t['tool_calls']} "
              f"msg_chars={t['msg_chars']}")
    searched = any(t["tool_calls"] for t in turns)
    print("\nVERDICT: web tools used =", searched)

    if answer:
        print("model_used:", model_used)
        print("usage:", usage)
        print("answer chars:", len(answer))
        print("LAST 200 chars:", repr(answer[-200:]))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
