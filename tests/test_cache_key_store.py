"""Tests for the anchored-rewrite-as-cache-key design.

The stored cache key is the cheap-model rewrite (generate_search_query) with
ABSOLUTE dates ("this month" → "August 2026") and URLs kept, so:
- G0 exact video-ID reuse still works (URLs are in the stored text)
- September "this month" does not match an August-anchored entry
- referential follow-ups ("number 5") are NEVER cached (context-dependent)
- key generation is temperature=0 (deterministic → upsert dedupe stays effective)

Run standalone:
    .venv/bin/python tests/test_cache_key_store.py
"""
import asyncio
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import processor  # noqa: E402

HISTORY = (
    "User: tell me about the dune universe\n"
    "Assistant: (previous answer)\n"
    "User: number 5\n"
    "Assistant: (previous answer)\n"
)


# ── Key generation: date injection + determinism ────────────────────

def test_search_key_prompt_injects_today_and_temp0():
    captured = {}

    async def fake_call_cheap(messages, tools=None, temperature=0.7):
        captured["system"] = messages[0]["content"]
        captured["temperature"] = temperature
        return "latest news August 2026"

    async def _run():
        return await processor.generate_search_query("latest news this month", [])

    with mock.patch.object(processor, "call_cheap", fake_call_cheap):
        key = asyncio.run(_run())

    assert key == "latest news August 2026"
    assert "Today's date is" in captured["system"], "prompt must carry today's date"
    assert "absolute" in captured["system"].lower(), "prompt must demand absolute dates"
    assert "KEEP any URLs" in captured["system"], "prompt must keep URLs (G0 needs them)"
    assert captured["temperature"] == 0, "key generation must be deterministic"


def test_search_key_fallback_keeps_urls():
    """The prompt must not instruct dropping URLs (previous prompt did)."""
    assert "NOT the raw URL" not in processor._SEARCH_QUERY_PROMPT


# ── Store path: anchored rewrite becomes the key ─────────────────────

def test_store_uses_anchored_rewrite():
    captured = {}

    async def fake_cache_lookup(q, purpose="chat"):
        return None

    async def fake_generate_search_query(q, turns):
        return "number of global disasters recorded August 2026"

    async def fake_expensive(messages):
        return "fresh answer " * 40, "deepseek-v4-pro"

    def fake_upsert_qa(query, answer, model, purpose="chat"):
        captured["query"] = query
        return 999

    def fake_record(**kw):
        pass

    async def _run():
        return await processor.process_query(
            user_query="how many disasters happened around the globe this month alone?",
            chat_history="",
        )

    with mock.patch.object(processor, "cache_lookup", fake_cache_lookup), \
         mock.patch.object(processor, "generate_search_query", fake_generate_search_query), \
         mock.patch.object(processor, "call_expensive", fake_expensive), \
         mock.patch.object(processor, "upsert_qa", fake_upsert_qa), \
         mock.patch.object(processor, "record_request", fake_record), \
         mock.patch.object(processor, "_clear_generated_images", lambda: None), \
         mock.patch.object(processor, "_get_generated_images", lambda: []):
        asyncio.run(_run())

    assert captured["query"] == "number of global disasters recorded August 2026", (
        f"stored key must be the anchored rewrite, got {captured['query']!r}"
    )


def test_referential_query_never_stored():
    captured = {}

    async def fake_cache_lookup(q, purpose="chat"):
        raise AssertionError("referential queries must bypass the cache entirely")

    async def fake_expensive(messages):
        return "the fifth point is... " * 10, "deepseek-v4-pro"

    def fake_upsert_qa(query, answer, model, purpose="chat"):
        captured["query"] = query  # must never fire
        return 999

    def fake_record(**kw):
        pass

    async def _run():
        return await processor.process_query(
            user_query="number 5",
            chat_history=HISTORY,
        )

    with mock.patch.object(processor, "cache_lookup", fake_cache_lookup), \
         mock.patch.object(processor, "call_expensive", fake_expensive), \
         mock.patch.object(processor, "upsert_qa", fake_upsert_qa), \
         mock.patch.object(processor, "record_request", fake_record), \
         mock.patch.object(processor, "_clear_generated_images", lambda: None), \
         mock.patch.object(processor, "_get_generated_images", lambda: []):
        asyncio.run(_run())

    assert "query" not in captured, (
        f"referential answer must not be cached, got {captured.get('query')!r}"
    )


def test_store_fallback_to_raw_when_key_empty():
    """Unresolvable-link → generate_search_query returns '' → store raw query."""
    captured = {}

    async def fake_cache_lookup(q, purpose="chat"):
        return None

    async def fake_generate_search_query(q, turns):
        return ""  # unresolvable link

    async def fake_expensive(messages):
        return "the link could not be resolved " * 10, "deepseek-v4-pro"

    def fake_upsert_qa(query, answer, model, purpose="chat"):
        captured["query"] = query
        return 999

    def fake_record(**kw):
        pass

    async def _run():
        return await processor.process_query(
            user_query="summarise https://example.com/blocked-page",
            chat_history="",
        )

    with mock.patch.object(processor, "cache_lookup", fake_cache_lookup), \
         mock.patch.object(processor, "generate_search_query", fake_generate_search_query), \
         mock.patch.object(processor, "call_expensive", fake_expensive), \
         mock.patch.object(processor, "upsert_qa", fake_upsert_qa), \
         mock.patch.object(processor, "record_request", fake_record), \
         mock.patch.object(processor, "_clear_generated_images", lambda: None), \
         mock.patch.object(processor, "_get_generated_images", lambda: []):
        asyncio.run(_run())

    assert captured["query"] == "summarise https://example.com/blocked-page", (
        f"empty key must fall back to raw query, got {captured['query']!r}"
    )


# ── Runner ───────────────────────────────────────────────────────────

def main():
    tests = [
        test_search_key_prompt_injects_today_and_temp0,
        test_search_key_fallback_keeps_urls,
        test_store_uses_anchored_rewrite,
        test_referential_query_never_stored,
        test_store_fallback_to_raw_when_key_empty,
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
