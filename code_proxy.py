"""Code-completion endpoint with JUDGE-based cache routing.

Separate from the general-chat path (proxy.py). General chat keeps its
context-style + IRRELEVANT design; this module serves the NEW code endpoints
(/v1/code/chat/completions) and uses a JSON-verdict transferability judge:

    cache match -> JUDGE (cheap) -> p_solve + capability_boundary
        >= threshold -> cheap WRITER adapts the cached example
        <  threshold  -> EXPENSIVE model answers fresh
    no match -> EXPENSIVE model

Design decisions (see ~/scratch/lcl-injection-test/DESIGN.md):
  * Conservative routing: over-routing to expensive is preferred over garbage.
  * Expensive path has NO effective token cap (CODE_EXPENSIVE_MAX_TOKENS) so a
    reasoner (deepseek-v4-pro) can finish its hidden deliberation + answer.
  * Same qa_cache as general chat for v1 (judge gates keep it safe).
  * v1 streams text + tool_calls; full agentic streaming is a follow-up.
"""

import json as _json
import logging
import time

from openai import AsyncOpenAI

from llm import (
    _build_cheap_client,
    _build_expensive_client,
    get_cheap_model,
    get_expensive_model,
)
from proxy import _extract_query_info, _format_sse
from session import session_id_from

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

Assess whether the cheap model can correctly adapt the cached answer to the new
question, and return a single JSON object with these fields:

- p_solve: float (0.0-1.0) — probability the cheap model can correctly adapt the
  cached answer to the new question. Think about: does this require only cosmetic
  edits, or does it need deep restructuring / new algorithms?

- capability_boundary: one of "supported" | "uncertain" | "unsupported"
  - "supported": same operation + cosmetic edits — rename, change a constant,
    tweak a condition, swap types
  - "uncertain": same domain but needs real rewrite, composition, or multiple
    adaptation steps
  - "unsupported": different topic / no usable building block at all

- crux: the single hardest requirement for whole-task success (short string).

- primary_rule: which rule decided the boundary (short string).

Return ONLY the JSON object, nothing else. Do not use markdown fences.

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


def _parse_verdict(content: str) -> dict | None:
    """Parse the JSON verdict from judge output. Returns dict or None on failure."""
    if not content or not content.strip():
        return None
    text = content.strip()
    # Strip optional markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        end = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "```":
                end = i
                break
        if end is not None and end > 0:
            text = "\n".join(lines[1:end]).strip()
        else:
            text = "\n".join(lines[1:]).strip()
    try:
        verdict = _json.loads(text)
    except _json.JSONDecodeError:
        return None
    if not isinstance(verdict, dict):
        return None
    p_solve = verdict.get("p_solve")
    boundary = verdict.get("capability_boundary")
    if p_solve is None or boundary is None:
        return None
    try:
        p_solve = float(p_solve)
    except (ValueError, TypeError):
        return None
    if not (0.0 <= p_solve <= 1.0):
        return None
    if boundary not in ("supported", "uncertain", "unsupported"):
        return None
    return {
        "p_solve": p_solve,
        "capability_boundary": boundary,
        "crux": str(verdict.get("crux", "")),
        "primary_rule": str(verdict.get("primary_rule", "")),
    }


# _parse_class is superseded by _parse_verdict — kept as doc reference only.
# Original semantics: TRIVIAL → supported (high p_solve), RESTRUCTURE → uncertain
# (mid p_solve), UNRELATED → unsupported (low p_solve).


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


async def _judge(cached_q: str, cached_a: str, new_q: str) -> dict | None:
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
    verdict = _parse_verdict(content)
    if verdict is None:
        logger.info("code judge -> parse failure (raw=%r)", content[:120])
    else:
        logger.info(
            "code judge -> p_solve=%.2f boundary=%s crux=%r rule=%r",
            verdict["p_solve"],
            verdict["capability_boundary"],
            verdict.get("crux", ""),
            verdict.get("primary_rule", ""),
        )
    return verdict


async def _write_code(cached_q, cached_a, messages: list[dict], tools=None) -> dict:
    client = _build_cheap_client()
    model = get_cheap_model()
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "") or ""
            break
    prompt = WRITER_PROMPT.format(cached_q=cached_q, cached_a=cached_a, new_q=last_user_msg)
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


async def _answer_expensive(messages: list[dict], tools=None) -> dict:
    client, model = _expensive_client_and_model()
    kw = dict(
        model=model,
        messages=messages,
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


# ── Router (dispatches to classifier / stage / passthrough) ──────
#
# The routing logic lives in `router.py` (route_and_answer). `code_proxy.py`
# keeps only the low-level judge / writer / expensive callers, which `router.py`
# imports lazily to avoid a circular import at module load time.


# ── Non-streaming handler ─────────────────────────────────────────


async def handle_code_completion(body: dict) -> dict:
    from router import route_and_answer

    user_query, match_query, messages, _temp, _max_tokens, tools, _ = _extract_query_info(body)
    session_id = session_id_from(messages, body.get("x_session_id"))
    result, routing_meta = await route_and_answer(
        messages, match_query, user_query, tools, session_id, body,
    )
    logger.info("routing decision: source=%s rationale=%s",
                routing_meta["decision_source"], routing_meta["rationale"])

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": routing_meta["selected_model"],
        "choices": [
            {
                "index": 0,
                "message": result["message"],
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": result["usage"],
        "_routing_meta": routing_meta,
    }


# ── Streaming handler (text deltas; judge runs first, non-streamed) ──


async def stream_code_completion(body: dict):
    from router import route_and_answer

    user_query, match_query, messages, _temp, _max_tokens, tools, _ = _extract_query_info(body)
    session_id = session_id_from(messages, body.get("x_session_id"))
    chat_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    result, routing_meta = await route_and_answer(
        messages, match_query, user_query, tools, session_id, body,
    )
    logger.info("routing decision: source=%s rationale=%s",
                routing_meta["decision_source"], routing_meta["rationale"])

    model_used = routing_meta["selected_model"]
    msg = result["message"]
    tool_calls = msg.get("tool_calls")

    if tool_calls:
        for tc in tool_calls:
            yield _format_sse(chat_id, created, model_used, {
                "delta": {"tool_calls": [tc]},
                "finish_reason": None,
            })
        yield _format_sse(chat_id, created, model_used, {
            "delta": {},
            "finish_reason": "tool_calls",
            "usage": result["usage"],
        })
    else:
        content = msg.get("content") or ""
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
