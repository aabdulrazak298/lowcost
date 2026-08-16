"""Standalone test for agentic cache search + exact video-ID reuse (G0).

Proves:
  1. Same video re-ask → exact video-ID hit (#68 WWII) WITHOUT any LLM/VPS call.
  2. Different video (Rick Astley) → G0 miss → agentic search → title key → miss.
  3. Unresolvable (garbage ID) → G0 miss → resolve fail → "" → skip cache.
Runs against the live cheap model + VPS. Does NOT touch the running service.
"""
import asyncio
import sys

sys.path.insert(0, ".")

from processor import _lookup_exact_video, _extract_video_ids, generate_search_query, _resolve_references  # noqa: E402
from db import cache_lookup  # noqa: E402


async def main():
    print("=== G0: exact video-ID lookup (free, no LLM) ===")
    for url, expect in [
        ("summarise https://www.youtube.com/watch?v=41dD-N5-3no", "#68 (WWII, cached)"),
        ("summarise https://www.youtube.com/watch?v=dQw4w9WgXcQ", "None (not cached)"),
        ("summarise https://www.youtube.com/watch?v=AAAAAAAAAAA", "None (not cached)"),
        ("tell me about https://youtu.be/41dD-N5-3no please", "#68 via youtu.be short link"),
    ]:
        ids = _extract_video_ids(url)
        hit = _lookup_exact_video(url)
        print(f"  ids={ids!r:40} -> {('HIT #' + str(hit['id'])) if hit else 'None'}  (expect {expect})")

    print()
    print("=== Full flow ===")
    cases = [
        ("same video re-ask (WWII)", [], "summarise https://www.youtube.com/watch?v=41dD-N5-3no"),
        ("different video (Rick Astley)", [], "summarise https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
        ("unresolvable (garbage id)", [], "summarise https://www.youtube.com/watch?v=AAAAAAAAAAA"),
    ]
    for desc, hist, q in cases:
        print(f"[{desc}]  raw: {q}")
        hit = _lookup_exact_video(q)
        if hit:
            print(f"  -> G0 HIT: #{hit['id']} (reuse, no LLM/VPS)")
            print()
            continue
        refs, ok = await asyncio.to_thread(_resolve_references, q)
        key = await generate_search_query(q, hist)
        if key == "":
            print(f"  resolve ok={ok} -> EMPTY key -> skip cache, final LLM reports")
            print()
            continue
        print(f"  key : {key}")
        m = await cache_lookup(key)
        print(f"  -> cache {'HIT #' + str(m['id']) if m else 'MISS'}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
