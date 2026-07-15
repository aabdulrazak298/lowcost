"""Core orchestrator: match -> route -> cache -> respond."""
import time
from fastapi import Request

from config import CHEAP_MODEL
from db import get_all_queries, insert_qa, get_cache_stats
from matcher import find_best_match
from llm import call_cheap, call_expensive
from stats import record_request


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


async def handle_chat_completion(request: Request) -> dict:
    """Process a /v1/chat/completions request."""
    body = await request.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 2048)
    # DeepSeek V4 Pro is a reasoning model — needs headroom for thinking.
    # Ensure at least 512 tokens or reasoning may consume the entire budget.
    expensive_max_tokens = max(max_tokens, 512)

    # Build a context-aware query from recent user messages.
    # Using only the last message loses conversation context:
    #   "What is a VFD?" → "How to commission it?" should match VFD cache.
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_msgs:
        user_msgs = [m.get("content", "") for m in messages]

    # Last user message is the primary query; prefix with recent context
    user_query = user_msgs[-1]
    match_query = " ".join(user_msgs[-3:])

    # Check cache
    cached = get_all_queries()
    match = find_best_match(match_query, cached)

    if match:
        # --- TRY CHEAP PATH ---
        context_prompt = CHEAP_MODEL_CONTEXT_PROMPT.format(
            expert_answer=match["answer"],
            user_query=user_query,
        )
        cheap_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer accurately using "
                    "the provided expert reference."
                ),
            },
            {"role": "user", "content": context_prompt},
        ]
        answer = await call_cheap(cheap_messages, temperature, max_tokens)

        # Self-check: did the cheap model reject the cached answer?
        if answer.strip().upper().startswith("IRRELEVANT"):
            # Cached answer was about a different topic — escalate
            record_request(hit=False, model="irrelevant-escalated")
            match = None  # force expensive path below
        else:
            model_used = f"{CHEAP_MODEL} (cached)"
            record_request(hit=True, model=model_used)

    if not match:
        # --- EXPENSIVE PATH: DeepSeek V4 Pro ---
        # Add cache-aware system prompt so the model generates thorough,
        # self-contained answers suitable for reuse.
        cache_aware = [{
            "role": "system",
            "content": (
                "Your answer will be cached and reused as a reference for "
                "similar future queries. Be thorough, self-contained, and "
                "include all relevant details so the cached version stands "
                "alone as a complete reference."
            ),
        }]
        answer, model_used = await call_expensive(
            cache_aware + messages, temperature, expensive_max_tokens
        )
        insert_qa(match_query, answer, model_used)
        record_request(hit=False, model=model_used)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
