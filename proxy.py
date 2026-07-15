"""Core orchestrator: match -> route -> cache -> respond."""
import json
import time

from config import CHEAP_MODEL
from db import cache_lookup, insert_qa
from matcher import find_best_match
from llm import (
    call_cheap,
    call_expensive,
    call_cheap_full,
    call_expensive_full,
    stream_expensive_full,
)
from stats import record_request


def _record(hit: bool, model: str, usage: dict | None = None, tool_calls: int = 0):
    pt = (usage or {}).get("prompt_tokens", 0)
    ct = (usage or {}).get("completion_tokens", 0)
    record_request(hit=hit, model=model, tool_calls=tool_calls,
                   prompt_tokens=pt, completion_tokens=ct)


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


def _extract_query_info(body: dict) -> tuple[str, str, list, float, int, list | None, int]:
    """Parse common fields from request body. Returns (user_query, match_query,
    messages, temperature, max_tokens, tools, expensive_max_tokens)."""
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 2048)
    tools = body.get("tools")
    expensive_max_tokens = max(max_tokens, 512)

    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    t = item.get("text") or item.get("content") or ""
                    parts.append(str(t))
            return " ".join(parts)
        return str(content) if content else ""

    user_msgs = [_extract_text(m.get("content", "")) for m in messages if m.get("role") == "user"]
    if not user_msgs:
        user_msgs = [_extract_text(m.get("content", "")) for m in messages]

    user_query = user_msgs[-1] if user_msgs else ""
    match_query = " ".join(u for u in user_msgs[-3:] if u)

    return user_query, match_query, messages, temperature, max_tokens, tools, expensive_max_tokens


async def handle_chat_completion(body: dict) -> dict:
    """Process a /v1/chat/completions request (non-streaming)."""
    user_query, match_query, messages, temperature, max_tokens, tools, expensive_max_tokens = (
        _extract_query_info(body)
    )

    match = await cache_lookup(match_query)
    response_content = None
    response_tool_calls = None
    model_used = CHEAP_MODEL
    usage = None
    finish_reason = "stop"

    if match:
        context_prompt = CHEAP_MODEL_CONTEXT_PROMPT.format(
            expert_answer=match["answer"],
            user_query=user_query,
        )
        cheap_messages = list(messages)
        cheap_messages.append({"role": "user", "content": context_prompt})

        if tools:
            result = await call_cheap_full(
                cheap_messages, temperature, max_tokens, tools=tools
            )
        else:
            text = await call_cheap(cheap_messages, temperature, max_tokens)
            result = {
                "content": text,
                "tool_calls": None,
                "model": f"{CHEAP_MODEL} (cached)",
                "usage": None,
                "finish_reason": "stop",
            }

        if (
            not result.get("tool_calls")
            and result.get("content", "").strip().upper().startswith("IRRELEVANT")
        ):
            _record(hit=False, model="irrelevant-escalated")
            match = None
        else:
            response_content = result["content"]
            response_tool_calls = result.get("tool_calls")
            model_used = result.get("model", f"{CHEAP_MODEL} (cached)")
            usage = result.get("usage")
            finish_reason = result.get("finish_reason", "stop")
            _record(hit=True, model=model_used, usage=usage)

    if not match:
        cache_aware = [
            {
                "role": "system",
                "content": (
                    "Your answer will be cached and reused as a reference for "
                    "similar future queries. Be thorough, self-contained, and "
                    "include all relevant details so the cached version stands "
                    "alone as a complete reference."
                ),
            }
        ]

        if tools:
            result = await call_expensive_full(
                cache_aware + messages, temperature, expensive_max_tokens, tools=tools
            )
        else:
            text, model = await call_expensive(
                cache_aware + messages, temperature, expensive_max_tokens
            )
            result = {
                "content": text,
                "tool_calls": None,
                "model": model,
                "usage": None,
                "finish_reason": "stop",
            }

        response_content = result["content"]
        response_tool_calls = result.get("tool_calls")
        model_used = result.get("model", "deepseek-v4-pro")
        usage = result.get("usage")
        finish_reason = result.get("finish_reason", "stop")

        insert_qa(match_query, response_content, model_used)
        _record(hit=False, model=model_used, usage=usage)

    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    if response_tool_calls:
        response["choices"][0]["message"]["tool_calls"] = response_tool_calls

    return response


def _format_sse(chat_id: str, created: int, model: str, chunk: dict) -> str:
    """Wrap a delta chunk into an SSE data line."""
    data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": chunk.get("delta", {}),
                "finish_reason": chunk.get("finish_reason"),
            }
        ],
    }
    if chunk.get("usage"):
        data["usage"] = chunk["usage"]
    return f"data: {json.dumps(data)}\n\n"


async def stream_chat_completion(body: dict):
    """Process a /v1/chat/completions request with stream=True.

    Yields SSE-formatted strings for a FastAPI StreamingResponse.
    """
    user_query, match_query, messages, temperature, max_tokens, tools, expensive_max_tokens = (
        _extract_query_info(body)
    )

    chat_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    match = await cache_lookup(match_query)

    if match:
        # Cache hit — buffer cheap response, verify IRRELEVANT, then stream
        context_prompt = CHEAP_MODEL_CONTEXT_PROMPT.format(
            expert_answer=match["answer"],
            user_query=user_query,
        )
        cheap_messages = list(messages)
        cheap_messages.append({"role": "user", "content": context_prompt})

        result = await call_cheap_full(
            cheap_messages, temperature, max_tokens, tools=tools
        )
        content = result["content"]
        tool_calls = result.get("tool_calls")
        model_used = result.get("model", f"{CHEAP_MODEL} (cached)")
        usage = result.get("usage")
        finish_reason = result.get("finish_reason", "stop")

        if (
            not tool_calls
            and content.strip().upper().startswith("IRRELEVANT")
        ):
            _record(hit=False, model="irrelevant-escalated")
            match = None
        else:
            # Stream buffered response
            if tool_calls:
                for tc in tool_calls:
                    yield _format_sse(chat_id, created, model_used, {
                        "delta": {"tool_calls": [tc]},
                        "finish_reason": None,
                    })
                yield _format_sse(chat_id, created, model_used, {
                    "delta": {},
                    "finish_reason": "tool_calls",
                })
            else:
                chunk_size = 16
                for i in range(0, len(content), chunk_size):
                    yield _format_sse(chat_id, created, model_used, {
                        "delta": {"content": content[i : i + chunk_size]},
                        "finish_reason": None,
                    })

                yield _format_sse(chat_id, created, model_used, {
                    "delta": {},
                    "finish_reason": finish_reason,
                    "usage": usage,
                })

            yield "data: [DONE]\n\n"
            _record(hit=True, model=model_used, usage=usage)
            return

    # Cache miss (or IRRELEVANT escalated) — stream from expensive model
    cache_aware = [
        {
            "role": "system",
            "content": (
                "Your answer will be cached and reused as a reference for "
                "similar future queries. Be thorough, self-contained, and "
                "include all relevant details so the cached version stands "
                "alone as a complete reference."
            ),
        }
    ]

    full_text = ""
    model_used = "deepseek-v4-pro"

    try:
        async for chunk in stream_expensive_full(
            cache_aware + messages, temperature, expensive_max_tokens, tools=tools
        ):
            delta = chunk.get("delta", {}) or {}
            full_text += delta.get("content") or ""
            model_used = chunk.get("model") or model_used
            yield _format_sse(chat_id, created, model_used, chunk)

        yield "data: [DONE]\n\n"

        if full_text:
            insert_qa(match_query, full_text, model_used)
        _record(hit=False, model=model_used, usage=usage)

    except Exception as e:
        error_chunk = {
            "error": {"message": str(e), "type": "server_error"},
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"