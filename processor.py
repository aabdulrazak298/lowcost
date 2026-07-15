"""Shared query processor — cache-check → route → respond.

Both the Flask Chat webhook and Telegram bot call this same function.
"""
import datetime as _dt
import asyncio
import logging
from config import CHEAP_MODEL
from db import cache_lookup, insert_qa, increment_hit_count
from llm import call_cheap, call_expensive, call_expensive_stream, _clear_generated_images, _get_generated_images, _wait_for_images
from stats import record_request

logger = logging.getLogger(__name__)

_TODAY = _dt.datetime.now().strftime("%A, %d %B %Y")
_DATE_CONTEXT = f"Today's date is {_TODAY}. Use this for any time-sensitive context."

CHEAP_MODEL_CONTEXT_PROMPT = """A similar question was previously answered by an expert AI.
Here is that answer for reference:

---
{expert_answer}
---

IMPORTANT — RELEVANCE CHECK:
If the expert answer above is about a COMPLETELY DIFFERENT topic than the
user's question, do NOT try to adapt it. Reply with EXACTLY the single word
"IRRELEVANT" and nothing else.

Otherwise, use the expert answer as your knowledge source. Answer the user's
question accurately. If the new question differs from the original, adapt the
answer appropriately while preserving factual accuracy.

User's question: {user_query}"""


async def process_query(
    user_query: str,
    chat_history: str = "",
    system_prompt: str | None = None,
) -> tuple[str, str]:
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
        # Strip XML thinking tags first. Models may explain WHY it's irrelevant
        # before saying the word — scan first 300 chars for IRRELEVANT.
        import re as _re
        clean = _re.sub(r'</?think\w*>', '', answer, flags=_re.IGNORECASE).strip()
        if "IRRELEVANT" in clean[:300].upper():
            record_request(hit=False, model="irrelevant-escalated")
            match = None  # force expensive path below
        else:
            increment_hit_count(match["id"])
            model_used = f"{CHEAP_MODEL} (cached)"
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

        import re as _re
        clean = _re.sub(r'</?think\w*>', '', answer, flags=_re.IGNORECASE).strip()
        if "IRRELEVANT" in clean[:300].upper():
            record_request(hit=False, model="irrelevant-escalated")
            match = None
        else:
            increment_hit_count(match["id"])
            model_used = f"{CHEAP_MODEL} (cached)"
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
