"""Shared query processor — cache-check → route → respond.

Both the Flask Chat webhook and Telegram bot call this same function.
"""
import datetime as _dt
import asyncio
import logging
import re as _re
from config import get_cheap_model, AGENTIC_CACHE
from db import cache_lookup, insert_qa, increment_hit_count
from llm import call_cheap, call_expensive, call_expensive_stream, _clear_generated_images, _get_generated_images, _wait_for_images
from stats import record_request

logger = logging.getLogger(__name__)

_TODAY = _dt.datetime.now().strftime("%A, %d %B %Y")
_DATE_CONTEXT = f"Today's date is {_TODAY}. Use this for any time-sensitive context."

CHEAP_MODEL_CONTEXT_PROMPT = """A similar question was previously answered by an expert AI.
Here is that answer as a FOUNDATION to build on:

---
{expert_answer}
---

IMPORTANT — RELEVANCE CHECK (do this FIRST):
Read the user's question. Decide whether the expert answer above is about the
SAME topic.

- If it is about a DIFFERENT topic (e.g. the expert answer is about an anime
  and the user is asking about a novel), STOP immediately. Do NOT explain the
  mismatch, do NOT apologize, do NOT answer the question anyway. Reply with
  EXACTLY one word and nothing else:

IRRELEVANT

- Only if it is genuinely about the SAME topic, proceed.

If you proceed, use the expert answer as a FOUNDATION — not a constraint. You have
access to tools (web search) to add fresh information, verify facts, or expand
on what's cached. The expert answer may be outdated or incomplete:
- Augment it with new details from web search
- Verify any claims that seem questionable
- Add context the expert answer missed
- Replace stale facts with current ones

Do NOT say "based on", "according to", or cite the expert answer in any way.
Answer the user's question directly, accurately and thoroughly.

User's question: {user_query}"""


_REJECTION_PHRASES = (
    # Metadata leak — model cites the cached "expert" material instead of answering.
    "expert answer", "expert material", "expert says", "expert reference",
    "expert ai", "cached answer", "cached expert", "cached response",
    "source material", "provided information", "provided reference",
    "provided by an expert",
    # Topic mismatch — model noticed the cached answer is about a different subject.
    "unrelated", "different topic", "completely different", "not related",
    "off-topic", "off topic", "wrong topic", "mismatch", "nothing to do with",
    "no relation", "different subject", "another topic", "not the same topic",
    "fresh answer", "from scratch",
)


def _is_rejection(answer: str) -> bool:
    """Return True if the cheap model rejected the cached answer.

    Signals (scoped to the first 300 chars to avoid false positives in a
    legitimate answer body):
      1. The literal IRRELEVANT keyword.
      2. Metadata-leak phrases ("expert answer", "reference", ...).
      3. Topic-mismatch phrases ("unrelated", "different topic", ...).

    This errs on the side of escalating: a false positive only costs one
    expensive call, while a false negative serves unrelated content as a
    "cached" hit.
    """
    if not answer:
        return True
    clean = _re.sub(r"</?think\w*>", "", answer, flags=_re.IGNORECASE).strip()
    head = clean[:300].lower()
    if "irrelevant" in head:
        return True
    return any(p in head for p in _REJECTION_PHRASES)


def _is_escalate(answer: str) -> bool:
    """True if the cheap model asked to escalate (no usable cache match)."""
    if not answer:
        return True
    return answer.strip().upper().startswith("ESCALATE")


AGENTIC_CACHE_PROMPT = """You answer the user using a cache of previous expert answers as your knowledge base.

You have a `search_cache` tool. Follow these steps exactly:
1. Rewrite the user's question into a concrete, specific question. Resolve any vague or implicit wording (like "the first one" or "that thing") using the conversation context.
2. Call search_cache with that question.
3. If the tool returns a RELATED cached question+answer, use it as your FOUNDATION — not a constraint — and expand or adapt it to fully answer the user. Augment with anything you know. Do NOT mention the cache, "expert", or say "based on".
4. If the tool returns NO MATCH or an answer about a DIFFERENT topic, do NOT answer the question — reply with exactly one word and nothing else:

ESCALATE"""


async def _agentic_cache_flow(user_query: str, chat_history: str) -> tuple[bool, str]:
    """Cheap model orchestrates cache retrieval via the search_cache tool.

    Returns (is_escalate, answer). When is_escalate is True, answer is ''.
    """
    from llm import search_cache

    messages = [
        {"role": "system", "content": f"{AGENTIC_CACHE_PROMPT}\n\n{_DATE_CONTEXT}"},
    ]
    if chat_history:
        messages.append({
            "role": "system",
            "content": f"Previous conversation:\n{chat_history[-2000:]}",
        })
    messages.append({"role": "user", "content": user_query})
    answer = await call_cheap(messages, tools=[search_cache])
    if _is_escalate(answer):
        return True, ""
    return False, answer


async def process_query(
    user_query: str,
    chat_history: str = "",
    system_prompt: str | None = None,
) -> tuple[str, str, list[str]]:
    """Process a user query through the two-tier cache → cheap → expensive pipeline.

    Args:
        user_query: The user's current message.
        chat_history: Previous conversation text (for multi-turn context).
        system_prompt: Optional override system prompt for expensive path.

    Returns:
        (answer_text, model_used_label, generated_image_paths)
    """
    # Clear any images from a previous request
    _clear_generated_images()

    # Build match query from recent USER messages only (not assistant answers).
    # chat_history is "User: ...\nAssistant: ..." — extract just the user lines
    # to avoid polluting the cache key with answer fragments.
    user_lines = []
    if chat_history:
        for line in chat_history.split("\n"):
            if line.startswith("User: "):
                user_lines.append(line[6:])  # strip "User: " prefix
    user_lines.append(user_query)
    # Use last 3 user messages for matching; if only 1, use just the current query
    match_query = " ".join(user_lines[-3:])

    if AGENTIC_CACHE:
        # New: cheap model orchestrates cache retrieval via the search_cache tool.
        is_escalate, answer = await _agentic_cache_flow(user_query, chat_history)
        if not is_escalate:
            model_used = f"{get_cheap_model()} (agentic-cached)"
            record_request(hit=True, model=model_used)
            return answer, model_used, _get_generated_images()
    else:
        match = await cache_lookup(match_query)

        if match:
            # --- TRY CHEAP PATH ---
            context_prompt = CHEAP_MODEL_CONTEXT_PROMPT.format(
                expert_answer=match["answer"],
                user_query=user_query,
            )
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. {_DATE_CONTEXT} Answer accurately using the provided expert reference.",
                },
            ]
            if chat_history:
                messages.append({
                    "role": "system",
                    "content": f"Previous conversation:\n{chat_history[-2000:]}",
                })
            messages.append({"role": "user", "content": context_prompt})
            answer = await call_cheap(messages, tools=None)  # tools enabled, max 10 rounds

            # Self-check: did the cheap model reject the cached answer?
            if _is_rejection(answer):
                record_request(hit=False, model="irrelevant-escalated")
            else:
                increment_hit_count(match["id"])
                model_used = f"{get_cheap_model()} (cached)"
                record_request(hit=True, model=model_used)
                return answer, model_used, _get_generated_images()

    # --- EXPENSIVE PATH ---
    messages = [
        {
            "role": "system",
            "content": _DATE_CONTEXT,
        },
    ]
    if chat_history:
        messages.append({
            "role": "system",
            "content": f"Previous conversation:\n{chat_history[-3000:]}",
        })
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})
    answer, model_used = await call_expensive(messages)
    insert_qa(match_query, answer, model_used)
    record_request(hit=False, model=model_used)

    return answer, model_used, _get_generated_images()


async def process_query_stream(
    user_query: str,
    callback,
    chat_history: str = "",
    system_prompt: str | None = None,
) -> tuple[str, list[str]]:
    """Same as process_query but streams expensive-model response via callback.

    callback(chunk: str) is called for each text chunk as it arrives.
    Returns (model_used, generated_image_paths).
    """
    _clear_generated_images()

    # Build match query
    user_lines = []
    if chat_history:
        for line in chat_history.split("\n"):
            if line.startswith("User: "):
                user_lines.append(line[6:])
    user_lines.append(user_query)
    match_query = " ".join(user_lines[-3:])

    if AGENTIC_CACHE:
        is_escalate, answer = await _agentic_cache_flow(user_query, chat_history)
        if not is_escalate:
            model_used = f"{get_cheap_model()} (agentic-cached)"
            record_request(hit=True, model=model_used)
            if asyncio.iscoroutinefunction(callback):
                await callback(answer)
            else:
                callback(answer)
            return model_used, _get_generated_images()
    else:
        match = await cache_lookup(match_query)

        if match:
            context_prompt = CHEAP_MODEL_CONTEXT_PROMPT.format(
                expert_answer=match["answer"],
                user_query=user_query,
            )
            messages = [{"role": "system", "content": f"You are a helpful assistant. {_DATE_CONTEXT} Answer accurately using the provided expert reference."}]
            if chat_history:
                messages.append({"role": "system", "content": f"Previous conversation:\n{chat_history[-2000:]}"})
            messages.append({"role": "user", "content": context_prompt})
            answer = await call_cheap(messages, tools=None)  # tools enabled, max 10 rounds

            if _is_rejection(answer):
                record_request(hit=False, model="irrelevant-escalated")
            else:
                increment_hit_count(match["id"])
                model_used = f"{get_cheap_model()} (cached)"
                record_request(hit=True, model=model_used)
                if asyncio.iscoroutinefunction(callback):
                    await callback(answer)
                else:
                    callback(answer)
                return model_used, _get_generated_images()

    # Expensive path with streaming
    messages = [{"role": "system", "content": _DATE_CONTEXT}]
    if chat_history:
        messages.append({"role": "system", "content": f"Previous conversation:\n{chat_history[-3000:]}"})
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})

    logger.info(f"Starting streaming for query: {user_query[:80]}")
    answer, model_used = await call_expensive_stream(messages, callback)
    logger.info(f"Streaming complete: {len(answer)} chars, model={model_used}")
    insert_qa(match_query, answer, model_used)
    record_request(hit=False, model=model_used)

    return model_used, _get_generated_images()
