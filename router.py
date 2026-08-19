"""Route-type abstraction: dispatches code completions to classifier / stage /
passthrough routing and returns a result plus routing metadata.

Replaces the direct routing calls inside `code_proxy.py`'s handlers.
"""

import logging

from config import (
    CODE_ROUTE_MODE,
    JUDGE_BASE_THRESHOLD,
    JUDGE_THRESHOLD_STEP,
    SESSION_AFFINITY,
    SESSION_TTL_SECONDS,
    STAGE_CONFIDENCE_THRESHOLD,
    STAGE_RECENT_TURN_WINDOW,
)
from db import cache_lookup, upsert_qa
from session import get_affinity, set_affinity, evict_stale
from stage_router import detect_tool_loop, score_signals
from stats import record_request

logger = logging.getLogger("lowcostllm.code")


async def _classifier_route(
    user_query: str, match_query: str, messages: list[dict], tools,
    session_id: str,
) -> tuple[dict, dict]:
    """Classifier path: cache lookup -> judge -> session affinity."""
    from code_proxy import _answer_expensive, _judge, _write_code

    aff = None
    if SESSION_AFFINITY:
        aff = await get_affinity(session_id, SESSION_TTL_SECONDS)
        await evict_stale(SESSION_TTL_SECONDS)

    if aff:
        logger.info("code route: session affinity -> %s", aff)
        if aff == "cheap":
            match = await cache_lookup(match_query, purpose="code")
            if match:
                result = await _write_code(match["query"], match["answer"], messages, tools)
                model_used = f"{result['model']} (code-cached)"
                record_request(hit=True, model=model_used,
                               prompt_tokens=result["usage"]["prompt_tokens"],
                               completion_tokens=result["usage"]["completion_tokens"], purpose="code")
                return result, {
                    "selected_model": model_used,
                    "rationale": "session-affinity cheap",
                    "decision_source": "session-affinity",
                }
        result = await _answer_expensive(messages, tools)
        model_used = result["model"]
        record_request(hit=False, model=model_used,
                       prompt_tokens=result["usage"]["prompt_tokens"],
                       completion_tokens=result["usage"]["completion_tokens"], purpose="code")
        _cache_fresh(result, match_query, model_used)
        return result, {
            "selected_model": model_used,
            "rationale": "session-affinity expensive",
            "decision_source": "session-affinity",
        }

    match = await cache_lookup(match_query, purpose="code")
    logger.info("code route: match=%s query=%r", bool(match), user_query[:60])

    if match:
        verdict = await _judge(match["query"], match["answer"], user_query)
        if verdict is None:
            result = await _answer_expensive(messages, tools)
            model_used = result["model"]
            record_request(hit=False, model=model_used,
                           prompt_tokens=result["usage"]["prompt_tokens"],
                           completion_tokens=result["usage"]["completion_tokens"], purpose="code")
            _cache_fresh(result, match_query, model_used)
            return result, {
                "selected_model": model_used,
                "rationale": "judge parse failure — fail-open",
                "decision_source": "fall_open",
            }

        threshold = JUDGE_BASE_THRESHOLD
        boundary = verdict["capability_boundary"]
        if boundary == "uncertain":
            threshold += JUDGE_THRESHOLD_STEP
        if boundary == "unsupported":
            threshold = min(1.0, threshold + 2 * JUDGE_THRESHOLD_STEP)
        target = "cheap" if verdict["p_solve"] >= threshold else "expensive"

        if SESSION_AFFINITY:
            await set_affinity(session_id, target)

        rationale = (
            f"p_solve={verdict['p_solve']:.2f} vs {threshold:.2f} "
            f"boundary={boundary} crux={verdict.get('crux', '')} "
            f"rule={verdict.get('primary_rule', '')}"
        )

        if target == "cheap":
            result = await _write_code(match["query"], match["answer"], messages, tools)
            model_used = f"{result['model']} (code-cached)"
            record_request(hit=True, model=model_used,
                           prompt_tokens=result["usage"]["prompt_tokens"],
                           completion_tokens=result["usage"]["completion_tokens"], purpose="code")
            return result, {
                "selected_model": model_used,
                "rationale": rationale,
                "decision_source": "llm-classifier",
            }

        result = await _answer_expensive(messages, tools)
        model_used = result["model"]
        record_request(hit=False, model=model_used,
                       prompt_tokens=result["usage"]["prompt_tokens"],
                       completion_tokens=result["usage"]["completion_tokens"], purpose="code")
        _cache_fresh(result, match_query, model_used)
        return result, {
            "selected_model": model_used,
            "rationale": rationale,
            "decision_source": "llm-classifier",
        }

    result = await _answer_expensive(messages, tools)
    model_used = result["model"]
    record_request(hit=False, model=model_used,
                   prompt_tokens=result["usage"]["prompt_tokens"],
                   completion_tokens=result["usage"]["completion_tokens"], purpose="code")
    _cache_fresh(result, match_query, model_used)
    return result, {
        "selected_model": model_used,
        "rationale": "no cache match",
        "decision_source": "fall_open",
    }


async def _stage_route(messages: list[dict], tools) -> tuple[dict, dict]:
    """Stage path: signal-driven routing from tool-result history."""
    from code_proxy import _answer_expensive, _write_code

    signals = score_signals(messages, STAGE_RECENT_TURN_WINDOW)
    confidence = signals["confidence"]
    source = signals["source"]
    target = signals["target"]

    if confidence < STAGE_CONFIDENCE_THRESHOLD:
        result = await _answer_expensive(messages, tools)
        model_used = result["model"]
        record_request(hit=False, model=model_used,
                       prompt_tokens=result["usage"]["prompt_tokens"],
                       completion_tokens=result["usage"]["completion_tokens"], purpose="code")
        return result, {
            "selected_model": model_used,
            "rationale": (
                f"low confidence ({confidence:.2f}<{STAGE_CONFIDENCE_THRESHOLD}): "
                f"{signals.get('rationale', '')}"
            ),
            "decision_source": "fall_open",
        }

    rationales = [signals.get("rationale", ""), f"conf={confidence:.2f}"]

    if target == "cheap":
        # Try cache lookup; if match, adapt with cheap writer.
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "") or ""
                break
        match = await cache_lookup(last_user, purpose="code") if last_user else None
        if match:
            result = await _write_code(match["query"], match["answer"], messages, tools)
            model_used = f"{result['model']} (code-cached)"
            record_request(hit=True, model=model_used,
                           prompt_tokens=result["usage"]["prompt_tokens"],
                           completion_tokens=result["usage"]["completion_tokens"], purpose="code")
            return result, {
                "selected_model": model_used,
                "rationale": "stage cheap: " + ", ".join(rationales),
                "decision_source": source,
            }
        # No cache match — fall to expensive.
        result = await _answer_expensive(messages, tools)
        model_used = result["model"]
        record_request(hit=False, model=model_used,
                       prompt_tokens=result["usage"]["prompt_tokens"],
                       completion_tokens=result["usage"]["completion_tokens"], purpose="code")
        return result, {
            "selected_model": model_used,
            "rationale": "stage cheap (no cache match): " + ", ".join(rationales),
            "decision_source": source,
        }

    result = await _answer_expensive(messages, tools)
    model_used = result["model"]
    record_request(hit=False, model=model_used,
                   prompt_tokens=result["usage"]["prompt_tokens"],
                   completion_tokens=result["usage"]["completion_tokens"], purpose="code")
    return result, {
        "selected_model": model_used,
        "rationale": "stage expensive: " + ", ".join(rationales),
        "decision_source": source,
    }


def _cache_fresh(result: dict, match_query: str, model_used: str) -> None:
    content = result["message"]["content"]
    if content.strip():
        upsert_qa(match_query, content, model_used, purpose="code")


async def route_and_answer(
    messages: list[dict],
    match_query: str,
    user_query: str,
    tools,
    session_id: str,
    body: dict,
) -> tuple[dict, dict]:
    """Dispatch to the configured routing mode.

    Returns (result_dict, routing_meta):
      result_dict: {"message", "model", "usage", "finish_reason"}
      routing_meta: {"selected_model", "rationale", "decision_source"}
    """
    from code_proxy import _answer_expensive

    mode = CODE_ROUTE_MODE
    if mode == "auto":
        mode = "stage" if detect_tool_loop(messages) else "classifier"

    if mode == "passthrough":
        result = await _answer_expensive(messages, tools)
        model_used = result["model"]
        record_request(hit=False, model=model_used,
                       prompt_tokens=result["usage"]["prompt_tokens"],
                       completion_tokens=result["usage"]["completion_tokens"], purpose="code")
        _cache_fresh(result, match_query, model_used)
        return result, {
            "selected_model": model_used,
            "rationale": "passthrough mode",
            "decision_source": "passthrough",
        }

    if mode == "stage":
        return await _stage_route(messages, tools)

    if mode == "classifier":
        return await _classifier_route(user_query, match_query, messages, tools, session_id)

    # Unknown mode — conservative passthrough.
    result = await _answer_expensive(messages, tools)
    model_used = result["model"]
    record_request(hit=False, model=model_used,
                   prompt_tokens=result["usage"]["prompt_tokens"],
                   completion_tokens=result["usage"]["completion_tokens"], purpose="code")
    _cache_fresh(result, match_query, model_used)
    return result, {
        "selected_model": model_used,
        "rationale": f"unknown mode={mode} - conservative passthrough",
        "decision_source": "passthrough",
    }
