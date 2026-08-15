"""Code-completion endpoint with JUDGE-based cache routing.

Separate from the general-chat path (proxy.py). General chat keeps its
context-style + IRRELEVANT design; this module serves the NEW code endpoints
(/v1/code/chat/completions) and uses the validated 3-way transferability judge:

    cache match -> JUDGE (cheap) -> TRIVIAL / RESTRUCTURE / UNRELATED
        TRIVIAL               -> cheap WRITER adapts the cached example
        RESTRUCTURE/UNRELATED -> EXPENSIVE model answers fresh
    no match -> EXPENSIVE model

Design decisions (see ~/scratch/lcl-injection-test/DESIGN.md):
  * Conservative routing: over-routing to expensive is preferred over garbage.
  * Expensive path has NO effective token cap (CODE_EXPENSIVE_MAX_TOKENS) so a
    reasoner (deepseek-v4-pro) can finish its hidden deliberation + answer.
  * Same qa_cache as general chat for v1 (judge/IRRELEVANT gates keep it safe;
    a `purpose` column is a possible follow-up).
  * v1 streams text only; non-streaming returns tool_calls if the model emits
    them. Full agentic tool-call streaming is a follow-up.
"""

import logging
import time

from openai import AsyncOpenAI

from db import cache_lookup, insert_qa
from llm import (
    _build_cheap_client,
    _build_expensive_client,
    get_cheap_model,
    get_expensive_model,
)
from proxy import _extract_query_info, _format_sse
from stats import record_request

logger = logging.getLogger("lowcostllm.code")

# ── Tunables ──────────────────────────────────────────────────────
CODE_EXPENSIVE_MAX_TOKENS = 8192   # effectively "no cap" for the reasoner
CODE_WRITER_MAX_TOKENS = 1500
JUDGE_MAX_TOKENS = 2000   # headroom for forced-reasoning cheap models (qwen3.7-flash)
JUDGE_TEMPERATURE = 0.0
WRITER_TEMPERATURE = 0.2

# OpenRouter `reasoning.enabled=false` makes forced-reasoning models (e.g.
# qwen3.7-flash) emit directly without hidden deliberation. Measured ~19x faster
# judge + ~10x faster writer at a fraction of the tokens. Toggle off if a future
# cheap model rejects the flag.
DISABLE_REASONING = True


def _no_reasoning() -> dict:
    return {"extra_body": {"reasoning": {"enabled": False}}} if DISABLE_REASONING else {}

JUDGE_PROMPT = """You are a routing classifier for a code-assistance cache.

You are given:
1. A CACHED question and its CACHED answer (produced by an expert model).
2. A NEW question the user is asking right now.

Classify how much the cached answer must change to correctly solve the new question:

- TRIVIAL: the new task is the SAME operation as the cached one, needing only
  cosmetic edits — rename the function/variables, change a constant, change a
  condition, or change input/output types. A weak model can adapt the cached
  answer by editing a few tokens; it is essentially a ready template.

- RESTRUCTURE: the cached answer is RELEVANT (same domain or concept) but cannot
  be adapted by simple edits. It must be significantly rewritten, composed into
  a larger program, or used repeatedly as a sub-step inside new logic (e.g.
  wrapping it in a loop, calling it many times, combining it with other logic).
  A weak model would likely fail to do this correctly.

- UNRELATED: the cached answer is about a DIFFERENT topic or task and offers no
  usable building block for the new question.

Respond with EXACTLY one word on the first line: TRIVIAL, RESTRUCTURE, or
UNRELATED. You may add a short reason on the second line.

CACHED question:
{cached_q}

CACHED answer:
{cached_a}

NEW question:
{new_q}"""

WRITER_PROMPT = """Here is an example of a similar, already-solved task:

Task: {cached_q}

Solution:
{cached_a}

Now write the solution for this related task. Output ONLY the function code —
no explanation, no tests, no markdown fences.

Task: {new_q}"""


def _parse_class(content: str) -> str:
    first = content.strip().splitlines()[0].strip().upper() if content.strip() else ""
    for c in ("TRIVIAL", "RESTRUCTURE", "UNRELATED"):
        if first.startswith(c):
            return c
    # Conservative default: unparseable judgement -> expensive.
    return "UNRELATED"


def _expensive_client_and_model() -> tuple[AsyncOpenAI, str]:
    """Route like llm.call_expensive: OpenRouter-format IDs go to OpenRouter,
    native IDs go to the direct client."""
    model = get_expensive_model()
    if "/" in model:
        return _build_cheap_client(), model
    return _build_expensive_client(), model


def _usage_dict(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }


def _message_dict(resp) -> dict:
    msg = resp.choices[0].message
    out = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
            for tc in msg.tool_calls
        ]
    return out


async def _judge(cached_q: str, cached_a: str, new_q: str) -> str:
    client = _build_cheap_client()
    model = get_cheap_model()
    prompt = JUDGE_PROMPT.format(cached_q=cached_q, cached_a=cached_a, new_q=new_q)
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        **_no_reasoning(),
    )
    content = resp.choices[0].message.content or ""
    cls = _parse_class(content)
    logger.info("code judge -> %s (raw=%r)", cls, content[:80])
    return cls


async def _write_code(cached_q, cached_a, new_q, tools=None) -> dict:
    client = _build_cheap_client()
    model = get_cheap_model()
    prompt = WRITER_PROMPT.format(cached_q=cached_q, cached_a=cached_a, new_q=new_q)
    kw = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=WRITER_TEMPERATURE,
        max_tokens=CODE_WRITER_MAX_TOKENS,
        **_no_reasoning(),
    )
    if tools:
        kw["tools"] = tools
    resp = await client.chat.completions.create(**kw)
    return {
        "message": _message_dict(resp),
        "model": model,
        "usage": _usage_dict(resp),
        "finish_reason": resp.choices[0].finish_reason or "stop",
    }


async def _answer_expensive(new_q, tools=None) -> dict:
    client, model = _expensive_client_and_model()
    kw = dict(
        model=model,
        messages=[{"role": "user", "content": new_q}],
        temperature=WRITER_TEMPERATURE,
        max_tokens=CODE_EXPENSIVE_MAX_TOKENS,
    )
    if tools:
        kw["tools"] = tools
    resp = await client.chat.completions.create(**kw)
    return {
        "message": _message_dict(resp),
        "model": model,
        "usage": _usage_dict(resp),
        "finish_reason": resp.choices[0].finish_reason or "stop",
    }


# ── Routing (shared by streaming + non-streaming) ─────────────────


async def _route_and_respond(user_query: str, match_query: str, tools) -> tuple[dict, str]:
    """Judge -> route -> answer. Returns (result_dict, model_label).

    Caches fresh expensive answers (miss / RESTRUCTURE / UNRELATED); TRIVIAL
    answers are a cache adaptation and are not re-inserted.
    """
    match = await cache_lookup(match_query, purpose="code")
    is_trivial_hit = False
    logger.info("code route: match=%s query=%r", bool(match), user_query[:60])

    if match:
        cls = await _judge(match["query"], match["answer"], user_query)
        if cls == "TRIVIAL":
            is_trivial_hit = True
            result = await _write_code(match["query"], match["answer"], user_query, tools)
            model_used = f"{result['model']} (code-cached)"
            record_request(hit=True, model=model_used,
                           prompt_tokens=result["usage"]["prompt_tokens"],
                           completion_tokens=result["usage"]["completion_tokens"])
        else:
            result = await _answer_expensive(user_query, tools)
            model_used = result["model"]
            record_request(hit=False, model=model_used,
                           prompt_tokens=result["usage"]["prompt_tokens"],
                           completion_tokens=result["usage"]["completion_tokens"])
    else:
        result = await _answer_expensive(user_query, tools)
        model_used = result["model"]
        record_request(hit=False, model=model_used,
                       prompt_tokens=result["usage"]["prompt_tokens"],
                       completion_tokens=result["usage"]["completion_tokens"])

    if not is_trivial_hit:
        content = result["message"]["content"]
        if content.strip():
            insert_qa(match_query, content, model_used, purpose="code")

    return result, model_used


# ── Non-streaming handler ─────────────────────────────────────────


async def handle_code_completion(body: dict) -> dict:
    user_query, match_query, _messages, _temp, _max_tokens, tools, _ = _extract_query_info(body)
    result, model_used = await _route_and_respond(user_query, match_query, tools)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": result["message"],
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": result["usage"],
    }


# ── Streaming handler (text deltas; judge runs first, non-streamed) ──


async def stream_code_completion(body: dict):
    user_query, match_query, _messages, _temp, _max_tokens, tools, _ = _extract_query_info(body)
    chat_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    result, model_used = await _route_and_respond(user_query, match_query, tools)

    content = result["message"]["content"] or ""
    chunk_size = 16
    for i in range(0, len(content), chunk_size):
        yield _format_sse(chat_id, created, model_used, {
            "delta": {"content": content[i : i + chunk_size]},
            "finish_reason": None,
        })
    yield _format_sse(chat_id, created, model_used, {
        "delta": {},
        "finish_reason": result["finish_reason"],
        "usage": result["usage"],
    })
    yield "data: [DONE]\n\n"
