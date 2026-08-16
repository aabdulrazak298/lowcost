"""Tests for the cache-key pollution fix (store under the current query only).

Root cause: `match_query = " ".join(user_lines[-3:])` joined the last 3 user
messages into the cache key, so a fresh answer got stored under a polluted key
(real incident: id=60 mixed an "indonesia earthquake" question with a stale
X-Men URL, and the answer was cached under the joined string).

Fix under test: the EXPENSIVE path stores under `user_query` (the current
message), never the joined `match_query`. Lookup keeps using `match_query` so
follow-ups ("tell me more") still resolve against prior context.
"""
import asyncio
import os
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CURRENT_QUERY = "summarise https://example.com/x-men-article"
PRIOR_HISTORY = (
    "User: tell me more about indonesia earthquake\n"
    "Assistant: (previous answer)\n"
    "User: more details on the earthquake\n"
    "Assistant: (previous answer)\n"
)


def test_store_key_is_current_query_only():
    import processor

    captured = {}

    async def fake_cache_lookup(q, purpose="chat"):
        return None  # force a miss → expensive path

    async def fake_expensive(messages):
        return "fresh answer", "deepseek-v4-pro"

    def fake_insert_qa(query, answer, model, purpose="chat"):
        captured["query"] = query
        captured["answer"] = answer
        captured["model"] = model
        return 999

    def fake_record(**kw):
        pass

    async def _run():
        return await processor.process_query(
            user_query=CURRENT_QUERY,
            chat_history=PRIOR_HISTORY,
        )

    with mock.patch.object(processor, "cache_lookup", fake_cache_lookup), \
         mock.patch.object(processor, "call_expensive", fake_expensive), \
         mock.patch.object(processor, "insert_qa", fake_insert_qa), \
         mock.patch.object(processor, "record_request", fake_record), \
         mock.patch.object(processor, "_clear_generated_images", lambda: None), \
         mock.patch.object(processor, "_get_generated_images", lambda: []):
        asyncio.run(_run())

    assert captured["query"] == CURRENT_QUERY, (
        f"stored key must be the current query only, got {captured['query']!r}"
    )
    assert captured["answer"] == "fresh answer"
    assert captured["model"] == "deepseek-v4-pro"


def test_store_key_is_current_query_only_stream():
    import processor

    captured = {}

    async def fake_cache_lookup(q, purpose="chat"):
        return None

    async def fake_expensive_stream(messages, callback):
        return "fresh answer", "deepseek-v4-pro"

    def fake_insert_qa(query, answer, model, purpose="chat"):
        captured["query"] = query
        return 999

    def fake_record(**kw):
        pass

    async def _run():
        return await processor.process_query_stream(
            user_query=CURRENT_QUERY,
            callback=lambda chunk: None,
            chat_history=PRIOR_HISTORY,
        )

    with mock.patch.object(processor, "cache_lookup", fake_cache_lookup), \
         mock.patch.object(processor, "call_expensive_stream", fake_expensive_stream), \
         mock.patch.object(processor, "insert_qa", fake_insert_qa), \
         mock.patch.object(processor, "record_request", fake_record), \
         mock.patch.object(processor, "_clear_generated_images", lambda: None), \
         mock.patch.object(processor, "_get_generated_images", lambda: []):
        asyncio.run(_run())

    assert captured["query"] == CURRENT_QUERY, (
        f"stream stored key must be the current query only, got {captured['query']!r}"
    )


def main():
    tests = [
        test_store_key_is_current_query_only,
        test_store_key_is_current_query_only_stream,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
