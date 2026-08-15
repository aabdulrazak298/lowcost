"""Signal-driven routing from tool-result history (no LLM call).

This is a v1 simplification of Switchyard's tanh-squashed corroborative scorer:
instead of a smooth corroboration score, we count discrete signal categories
(severity, spinning, exploring, production_intensity) over the recent tool
activity window and combine them with a simple weighted threshold.

Signals (looked up in `tool` role messages' `content` and assistant
`tool_calls`):
  * severity (wrong): tool results containing error markers — a traceback /
    exception is critical and hard-overrides to expensive.
  * spinning (wrong): repeated/identical tool_calls with no successful write.
  * exploring (wrong, weak): read-only tool calls (read/grep/ls/find/cat) with
    no writes yet.
  * production_intensity (progress): successful write/edit tool results.

Corroborative score:
    wrong = count of severity/spinning signals
    progress = count of production_intensity signals
    wrong & not progress -> expensive, confidence = min(1, 0.5 + 0.25*wrong)
    progress & not wrong -> cheap,    confidence = min(1, 0.5 + 0.25*progress)
    else (ambiguous)      -> expensive, confidence = 0.0, source="fall_open"
"""

import logging

logger = logging.getLogger("lowcostllm.code")

_SEVERITY_MARKERS = (
    "error",
    "exception",
    "traceback",
    "failed",
    "eacces",
    "enoent",
    "not found",
    "non-zero",
)

_CRITICAL_MARKERS = ("traceback", "exception")

_READ_TOOLS = ("read", "grep", "ls", "find", "cat", "glob", "search")
_WRITE_TOOLS = (
    "write",
    "edit",
    "apply",
    "patch",
    "create",
    "replace",
    "insert",
    "delete",
    "mkdir",
    "move",
    "rename",
)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content) if content else ""


def detect_tool_loop(messages: list[dict]) -> bool:
    """True if any message has role == "tool" OR any assistant message has
    non-empty `tool_calls`."""
    for m in messages:
        if m.get("role") == "tool":
            return True
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return True
    return False


def score_signals(messages: list[dict], recent_window: int = 3) -> dict:
    """Inspect the last `recent_window` turns of tool activity.

    Returns {"target", "confidence", "source", "rationale"}.
    """
    if not detect_tool_loop(messages):
        return {
            "target": "expensive",
            "confidence": 0.0,
            "source": "fall_open",
            "rationale": "no tool activity detected",
        }

    # Build an id -> name map from assistant tool_calls.
    name_by_id: dict[str, str] = {}
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = (fn.get("name") or "").lower()
                if tc.get("id") and name:
                    name_by_id[tc["id"]] = name

    # Collect tool messages in order, then take the last window.
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    recent = tool_msgs[-recent_window:] if tool_msgs else []

    wrong = 0
    progress = 0
    critical = False
    rationale_parts = []

    recent_names: list[str] = []

    for m in recent:
        text = _extract_text(m.get("content", "")).lower()
        name = name_by_id.get(m.get("tool_call_id"), "")
        if name:
            recent_names.append(name)

        # severity
        for marker in _SEVERITY_MARKERS:
            if marker in text:
                wrong += 1
                rationale_parts.append(f"severity:{marker}")
                if marker in _CRITICAL_MARKERS:
                    critical = True
                break
            # else no severity for this marker

        # production_intensity: successful write/edit tool result (no errors)
        has_error = any(mk in text for mk in _SEVERITY_MARKERS)
        if name and not has_error and any(w in name for w in _WRITE_TOOLS):
            progress += 1
            rationale_parts.append(f"write:{name}")

    # spinning: repeated/identical tool_calls in window, no successful write.
    if recent_names and progress == 0:
        seen = set()
        for n in recent_names:
            if n in seen:
                wrong += 1
                rationale_parts.append(f"spinning:{n}")
            seen.add(n)

    # exploring: read-only tool calls in window with no writes yet (weak signal).
    if recent_names and progress == 0 and wrong == 0:
        if all(any(r in n for r in _READ_TOOLS) for n in recent_names):
            rationale_parts.append("exploring")

    if critical:
        return {
            "target": "expensive",
            "confidence": 1.0,
            "source": "override",
            "rationale": "critical traceback/exception: " + ", ".join(rationale_parts),
        }

    if wrong and not progress:
        confidence = min(1.0, 0.5 + 0.25 * wrong)
        return {
            "target": "expensive",
            "confidence": confidence,
            "source": "dimensions",
            "rationale": ", ".join(rationale_parts) or "severity/spinning signals",
        }

    if progress and not wrong:
        confidence = min(1.0, 0.5 + 0.25 * progress)
        return {
            "target": "cheap",
            "confidence": confidence,
            "source": "dimensions",
            "rationale": ", ".join(rationale_parts) or "production progress",
        }

    return {
        "target": "expensive",
        "confidence": 0.0,
        "source": "fall_open",
        "rationale": "ambiguous signals - conservative default",
    }
