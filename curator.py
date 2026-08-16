"""Cache curator — expensive-model verdict to detect and evict poisoned entries.

Layer 1 of the cache-poisoning fix. When the cheap model rejects a cache match
(IRRELEVANT), the system already holds the rejected candidate. Instead of
letting it linger in the DB forever (TTL is 365 days and there is no eviction
path), the expensive model — the only model trusted to judge facts — issues a
verdict:

    EVICT  → the entry is poisoned (answer doesn't match its own question,
             or is factually wrong/hallucinated). Delete the row.
    KEEP   → the entry is valid for its own question; it merely didn't match
             this user (a matcher false-positive, not a cache defect).

Unclear/malformed verdicts default to KEEP — a false-negative only leaves a
stale row (harmless-ish), while a false-positive would delete a good answer.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("lowcostllm.curator")

CURATOR_SYSTEM_PROMPT = (
    "You are auditing a Q&A cache entry for poisoning. "
    "The entry below was auto-matched to a user question, then rejected as "
    "irrelevant during serving. Decide whether the cache entry ITSELF is "
    "poisoned and should be permanently deleted, or whether it is a valid "
    "entry that merely did not match this particular user.\n"
    "\n"
    "A cache entry is POISONED (EVICT) if:\n"
    "- its answer does not actually answer its own cached question (topic mismatch), or\n"
    "- its answer is factually wrong or hallucinated, or\n"
    "- its cached question is corrupted/garbled (mixes unrelated content).\n"
    "\n"
    "If the entry is valid — the answer correctly answers its own cached "
    "question — reply KEEP: the mismatch was the matcher's fault, not the entry's.\n"
    "\n"
    "Reply with exactly one word: EVICT or KEEP."
)


def build_curator_messages(cached_question: str, cached_answer: str) -> list[dict]:
    """Build the (system, user) messages for the curator verdict call."""
    user = (
        f"Cached question:\n{cached_question}\n\n"
        f"Cached answer:\n{cached_answer}"
    )
    return [
        {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_curator_verdict(text: str | None) -> str:
    """Extract EVICT/KEEP from the model output.

    Strict single-token parse: the prompt demands exactly one word, so we look
    at the first token only. Anything that is not an explicit leading EVICT
    defaults to KEEP (never delete on an ambiguous signal).
    """
    if not text:
        return "KEEP"
    stripped = text.strip()
    if not stripped:
        return "KEEP"
    first = stripped.split()[0].strip().rstrip(".,!?;:").upper()
    return "EVICT" if first == "EVICT" else "KEEP"


async def run_curator(match: dict[str, Any], verdict_fn=None) -> str:
    """Ask the expensive model to judge a rejected cache entry, and evict if poisoned.

    Args:
        match: the rejected cache candidate (dict with id/query/answer).
        verdict_fn: optional injectable `async (cached_q, cached_a) -> str`
            returning the raw model text. Defaults to llm.call_curator_verdict.

    Returns:
        "EVICT" if the row was deleted, "KEEP" otherwise.
    """
    if verdict_fn is None:
        from llm import call_curator_verdict

        verdict_fn = call_curator_verdict

    try:
        raw = await verdict_fn(match.get("query", ""), match.get("answer", ""))
    except Exception:
        logger.exception("Curator verdict call failed — keeping entry (default KEEP)")
        return "KEEP"

    verdict = parse_curator_verdict(raw)

    if verdict == "EVICT":
        cache_id = match.get("id")
        if cache_id is None:
            logger.warning("Curator wanted EVICT but candidate has no id — skipping")
            return "KEEP"
        try:
            from db import delete_cache_entry

            deleted = delete_cache_entry(cache_id)
            logger.info(
                "Curator evicted poisoned cache entry id=%s (rows deleted=%s)",
                cache_id, deleted,
            )
        except Exception:
            logger.exception("Curator eviction failed for id=%s", cache_id)
            return "KEEP"

    return verdict
