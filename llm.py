"""LLM callers using OpenAI Agents SDK — provider-agnostic tool calling.

Migrated from raw httpx to the Agents SDK. Handles retries, fallbacks,
tool-calling loops, and provider routing automatically.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
import time

import httpx
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, set_tracing_disabled, ModelSettings

from config import (
    CHEAP_API_KEY,
    CHEAP_BASE_URL,
    CHEAP_FALLBACK_MODEL,
    CHEAP_FALLBACK_RETRIES,
    LLM_DEADLINE_SECONDS,
    PROVIDER_COOLDOWN_SECONDS,
    PROVIDER_FAIL_THRESHOLD,
    EXPENSIVE_API_KEY,
    EXPENSIVE_BASE_URL,
    ENGY_API_KEY,
    ENGY_BASE_URL,
    ENGY_MODELS,
    ENGY_FALLBACK_MODEL,
    get_cheap_model,
    get_cheap_fallback_model,
    get_expensive_model,
)
from stats import record_request

# Disable SDK tracing to stop 401 spam on non-OpenAI providers
set_tracing_disabled(True)

logger = logging.getLogger("lowcostllm.llm")

# ── Global usage tracker for real cost in footer ──────────────────

_last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "model": ""}


def get_last_usage() -> dict:
    return dict(_last_usage)


def _reset_usage() -> None:
    global _last_usage
    _last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "model": ""}


def _extract_usage(result, model_id: str) -> dict:
    """Sum token usage across a run's raw responses (SDK 0.20.0).

    RunResult no longer exposes a top-level .usage; usage lives on each
    ModelResponse in result.raw_responses (one per turn, so a multi-turn
    tool loop sums to the full run cost).
    """
    prompt = 0
    completion = 0
    for resp in getattr(result, "raw_responses", None) or []:
        u = getattr(resp, "usage", None)
        if u is not None:
            prompt += getattr(u, "input_tokens", 0) or 0
            completion += getattr(u, "output_tokens", 0) or 0
    return {"prompt_tokens": prompt, "completion_tokens": completion, "model": model_id}


# ── Image delivery context ────────────────────────────────────────
#
# The image tools are generic — they don't know which platform/chat asked.
# Each entry point binds a delivery target here before running a query so
# generate_image / edit_image can self-deliver instead of returning a URL
# that the LLM never relays:
#   * telegram → photo is sent to the chat via sendPhoto (before the reply)
#   * web/api  → URL is recorded in _generated_images and embedded inline

import contextvars

_delivery_ctx: contextvars.ContextVar = contextvars.ContextVar("delivery_ctx", default=None)


def set_delivery_context(platform: str, chat_id: int | None = None, token: str | None = None) -> None:
    """Bind the delivery target for the current request. platform: 'telegram' or 'web'."""
    _delivery_ctx.set({"platform": platform, "chat_id": chat_id, "token": token})


def clear_delivery_context() -> None:
    _delivery_ctx.set(None)


def _send_telegram_photo(chat_id: int, token: str, image_url: str) -> tuple[bool, str]:
    """Send a photo by URL to a Telegram chat. Returns (ok, detail)."""
    try:
        payload = urllib.parse.urlencode({
            "chat_id": chat_id,
            "photo": image_url,
        })
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data=payload.encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
        if resp.get("ok"):
            return True, ""
        return False, str(resp)[:300]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _deliver_generated_image(result: dict) -> str:
    """Record a generated image and self-deliver it to the bound target."""
    url = result["download_url"]
    model = result.get("model", "unknown")
    _generated_images.append(url)
    ctx = _delivery_ctx.get()
    if ctx and ctx.get("platform") == "telegram" and ctx.get("chat_id"):
        ok, detail = _send_telegram_photo(ctx["chat_id"], ctx["token"], url)
        if ok:
            return f"Image generated and sent to the chat (model: {model})."
        return f"Image generated but Telegram delivery failed ({detail}). URL: {url}"
    return f"Image: {url}\nModel: {model}"


# ── File delivery (send_file tool) ────────────────────────────────
#
# Same pattern as images: send_file self-delivers to the bound target.
#   * telegram → document sent to the chat via sendDocument (before the reply)
#   * web/api  → uploaded for a download URL; the model relays the URL in
#                its text answer
#   * no ctx   → saved to a local temp file (shouldn't happen in prod paths)

ALLOWED_FILE_EXTS = {"txt", "md", "csv", "json", "log", "yaml", "yml", "py", "ini", "cfg"}
MAX_FILE_BYTES = 300_000


def _sanitize_filename(filename: str) -> str:
    """Force a safe bare filename with an allowed extension (defaults to .md)."""
    name = os.path.basename(filename.strip().replace("\\", "/")) or "file"
    name = name.strip()
    if name.lower().endswith(tuple(f".{e}" for e in ALLOWED_FILE_EXTS)):
        return name
    return f"{os.path.splitext(name)[0] or 'file'}.md"


def _send_telegram_document(chat_id: int, token: str, filename: str, content: str) -> tuple[bool, str]:
    """Send a text file to a Telegram chat via sendDocument (multipart). Returns (ok, detail)."""
    try:
        data_bytes = content.encode("utf-8")
        with httpx.Client(timeout=60) as c:
            r = c.post(
                f"https://api.telegram.org/bot{token}/sendDocument",
                data={"chat_id": str(chat_id)},
                files={"document": (filename, data_bytes, "text/plain; charset=utf-8")},
            )
        resp = r.json()
        if r.status_code == 200 and resp.get("ok"):
            return True, ""
        return False, f"{r.status_code}: {str(resp)[:300]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _upload_file_for_url(filename: str, content: str) -> str:
    """Upload a text file to the local upload service, returning its download URL ('' on failure)."""
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(filename)[1] or ".txt", delete=False
        ) as f:
            f.write(content.encode("utf-8"))
            tmp_path = f.name
        p = subprocess.run(
            ["curl", "-s", "-m", "20", "http://localhost:8000/upload/file",
             "-H", "Authorization: Bearer 987654321",
             "-F", f"file=@{tmp_path};filename={filename}"],
            capture_output=True, text=True, timeout=25,
        )
        up = json.loads(p.stdout)
        return up.get("download_url") or ""
    except Exception:
        return ""
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _send_file_impl(filename: str, content: str) -> str:
    """Send a text file to the user (e.g. .txt, .md, .csv, .json) so they can store or copy it.

Use this whenever the user asks to send, save, export, or download content as a file
(e.g. \"send this as a .md file\", \"save this as notes.txt\"). Pass the COMPLETE file
content and a suitable filename — never truncate the content.
    """
    filename = _sanitize_filename(filename)
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        return (
            f"File too large ({len(content)} chars, max {MAX_FILE_BYTES}). "
            f"Split it into smaller parts or shorten the content."
        )
    ctx = _delivery_ctx.get()
    if ctx and ctx.get("platform") == "telegram" and ctx.get("chat_id"):
        ok, detail = _send_telegram_document(ctx["chat_id"], ctx["token"], filename, content)
        if ok:
            return f"File sent to the chat: {filename}"
        return f"File ready but Telegram delivery failed ({detail})."
    if ctx and ctx.get("platform") == "web":
        url = _upload_file_for_url(filename, content)
        if url:
            return f"File saved. Download: {url}"
        return "File ready but upload failed — paste the content directly in your reply."
    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(filename)[1] or ".txt", delete=False
    ) as f:
        f.write(content.encode("utf-8"))
        return f"File saved to {f.name} (no delivery target)."


send_file = function_tool(_send_file_impl, name_override="send_file")

# Appended to every agent's instructions so models reach for send_file when the
# user asks for a file (models otherwise default to printing content as text).
_FILE_TOOL_HINT = (
    "\n\nIf the user asks you to send, save, export, or download content as a file "
    "(.txt, .md, .csv, .json, etc.), call the send_file tool with the COMPLETE file "
    "content and a suitable filename — do not just print the content as text."
)


# ── Tool implementations ──────────────────────────────────────────

SEARXNG_URL = "http://127.0.0.1:8080/search?format=json"


@function_tool
def web_search(query: str) -> str:
    """Search the web using SearXNG. Returns title, snippet, and URL for each result."""
    try:
        url = f"{SEARXNG_URL}&q={urllib.parse.quote(query)}"
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "LowCostLLM/0.5"}),
            timeout=10,
        ) as r:
            data = json.loads(r.read())
    except Exception as e:
        return f"Search unavailable: {e}"

    results = data.get("results", [])
    if not results:
        return "No results found."

    return "\n\n".join(
        f"{i+1}. {r.get('title', '?')}\n   {r.get('content', '')[:300]}\n   {r.get('url', '')}"
        for i, r in enumerate(results[:8])
    )


@function_tool
def web_fetch(url: str) -> str:
    """Fetch and extract text content from a web page URL."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 LowCostLLM/0.5"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Fetch failed: {e}"

    # Strip scripts, styles, HTML tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:8000]


ALL_TOOLS = [web_search, web_fetch]


@function_tool
def youtube_transcript(video_url: str) -> str:
    """Get the transcript of a YouTube video. Use when user asks about YouTube content."""
    try:
        req = urllib.request.Request(
            "http://141.11.17.227:8000/api/youtube/script",
            data=json.dumps({"video_url_or_id": video_url}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": "987654321",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
    except Exception as e:
        return f"YouTube transcript unavailable: {e}"

    if not data.get("transcript_available"):
        title = data.get("metadata", {}).get("title", "Unknown")
        return f"No transcript available for: {title}"

    meta = data.get("metadata", {})
    transcript = data.get("transcript", [])

    lines = [
        f"Title: {meta.get('title', 'Unknown')}",
        f"Duration: {int(meta.get('duration', 0)) // 60} min",
        f"Video ID: {data.get('video_id', '?')}",
        "=" * 60,
    ]

    # Character budget (~400k chars ≈ 100k tokens). The old 200-segment cap
    # truncated ~15-25 min videos; qwen3.7-flash has a 1M-token context, so
    # ~8h of video fits in a single pass. Report truncation honestly so the
    # model doesn't pretend it saw the whole thing.
    _MAX_CHARS = 400_000
    total = 0
    shown = 0
    for seg in transcript:
        s = int(seg["start"])
        ts = f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        line = f"[{ts}] {seg['text']}"
        if total + len(line) + 1 > _MAX_CHARS:
            break
        lines.append(line)
        total += len(line) + 1
        shown += 1

    if shown < len(transcript):
        shown_min = int(meta.get("duration", 0) * shown / max(len(transcript), 1)) // 60
        total_min = int(meta.get("duration", 0)) // 60
        lines.append(
            f"\n[Transcript truncated: {shown} of {len(transcript)} segments "
            f"(~{shown_min} of {total_min} min shown).]"
        )

    return "\n".join(lines)


# ── Additional tools (matching ThinkLLM executor) ─────────────────


@function_tool
def run_code(code: str) -> str:
    """Execute Python code in a sandbox. Use for calculations, data processing, or logic."""
    try:
        req = urllib.request.Request(
            "http://localhost:8000/code/execute",
            data=json.dumps({"code": code, "timeout": 30}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer 987654321"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=35) as r:
            result = json.loads(r.read())
        if result.get("error"):
            return f"Error: {result['error']}"
        return str(result.get("stdout") or result.get("output") or result.get("result") or "(no output)")[:4000]
    except Exception as e:
        return f"Code execution failed: {e}"


@function_tool
def generate_graph(x: list, y: list, chart_type: str = "line", title: str = "") -> str:
    """Generate a chart/graph from data.

    chart_type: line, bar, scatter, histogram, pie.
    x can be numeric values OR string category labels (e.g. ["Jan","Feb","Mar"]).
    y must be numeric values. For pie, x = slice labels and y = slice values.
    """
    try:
        import base64
        payload = {"data": {"x": x, "y": y}, "graph_type": chart_type}
        if title:
            payload["data"]["title"] = title
        req = urllib.request.Request(
            "http://localhost:8000/code/generate_graph",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer 987654321"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            result = json.loads(r.read())
        if result.get("image"):
            b64_data = result["image"].split(",", 1)[-1] if "," in result["image"] else result["image"]
            img_data = base64.b64decode(b64_data)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(img_data)
                tmp_path = f.name
            p = subprocess.run(
                ["curl", "-s", "http://localhost:8000/upload/file",
                 "-H", "Authorization: Bearer 987654321",
                 "-F", f"file=@{tmp_path}"],
                capture_output=True, text=True, timeout=20,
            )
            Path(tmp_path).unlink(missing_ok=True)
            try:
                up_result = json.loads(p.stdout)
                if up_result.get("download_url"):
                    return f"Graph: {up_result['download_url']}"
            except json.JSONDecodeError:
                pass
            return f"Upload failed: {p.stdout[:200]}"
        return f"Graph generation failed: {str(result)[:500]}"
    except Exception as e:
        return f"Graph failed: {e}"


@function_tool
def generate_image(prompt: str) -> str:
    """Generate an AI image from a text prompt. Returns a download URL."""
    try:
        api_key = os.environ.get("CHEAP_API_KEY", "")
        req = urllib.request.Request(
            "http://127.0.0.1:8000/image/generate",
            data=json.dumps({
                "api_key": api_key,
                "model": "qwen/qwen-image-3",
                "content": prompt,
            }).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer 987654321"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
        if result.get("download_url"):
            return _deliver_generated_image(result)
        return f"Image generation failed: {result}"
    except Exception as e:
        return f"Image failed: {e}"


@function_tool
def edit_image(image_url: str, prompt: str) -> str:
    """Edit an existing image using AI. Provide the image URL and edit instructions."""
    try:
        api_key = os.environ.get("CHEAP_API_KEY", "")
        req = urllib.request.Request(
            "http://127.0.0.1:8000/image/edit",
            data=json.dumps({
                "api_key": api_key,
                "model": "qwen/qwen-image-3",
                "prompt": prompt,
                "image_url": image_url,
            }).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer 987654321"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
        if result.get("download_url"):
            return _deliver_generated_image(result)
        return f"Image edit failed: {result}"
    except Exception as e:
        return f"Image edit failed: {e}"


ALL_TOOLS = [web_search, web_fetch, youtube_transcript, run_code, generate_graph, generate_image, edit_image, send_file]


@function_tool
async def search_cache(query: str) -> str:
    """Search the cache of previously-answered questions for a similar one.

    Returns a JSON object with the cached question and answer, or 'NO MATCH' if
    nothing similar exists. Use this to check whether a past expert answer can
    be reused as a foundation for the current question."""
    import json as _json
    try:
        from matcher import smart_cache_lookup
        match = await smart_cache_lookup(query, purpose="chat")
    except Exception as e:
        return f"NO MATCH (error: {type(e).__name__})"
    if not match:
        return "NO MATCH"
    try:
        from db import increment_hit_count
        increment_hit_count(match["id"])
    except Exception:
        pass
    return _json.dumps({"question": match["query"], "answer": match["answer"][:1500]})


# ── Client factories ──────────────────────────────────────────────


def _wrap_error_detection(client: AsyncOpenAI) -> AsyncOpenAI:
    """Raise when OpenRouter signals a mid-stream provider failure.

    OpenRouter returns HTTP 200 with finish_reason="error" and PARTIAL content
    when the upstream provider (e.g. Alibaba serving qwen) fails mid-generation.
    The OpenAI SDK accepts the truncated text as a complete answer — observed
    2026-08-20: long qwen3.7-flash answers delivered cut mid-sentence with the
    footer intact. Raising here sends the failure into the existing retry /
    fallback chain (call_cheap attempts -> cheap fallback -> processor
    transient-failure path) instead of delivering a partial answer.

    Streaming responses are left untouched (their error path is chunk-based).
    """
    orig_create = client.chat.completions.create

    async def wrapped(*args, **kwargs):
        if kwargs.get("stream"):
            return await orig_create(*args, **kwargs)
        resp = await orig_create(*args, **kwargs)
        if getattr(resp, "choices", None):
            fr = getattr(resp.choices[0], "finish_reason", None)
            if fr == "error":
                partial = getattr(resp.choices[0].message, "content", "") or ""
                raise RuntimeError(
                    f"Provider returned finish_reason='error' (mid-stream "
                    f"failure) — partial response discarded: {partial[:120]!r}"
                )
        return resp

    client.chat.completions.create = wrapped  # type: ignore[method-assign]
    return client


def _build_cheap_client() -> AsyncOpenAI:
    """OpenRouter client for cheap model."""
    return _wrap_error_detection(AsyncOpenAI(
        base_url=CHEAP_BASE_URL,
        api_key=CHEAP_API_KEY,
        max_retries=1, timeout=httpx.Timeout(connect=5.0, read=180.0, write=120.0, pool=5.0),
        default_headers={
            "HTTP-Referer": "http://localhost:8800",
            "X-Title": "LowCostLLM",
        },
    ))


def _build_expensive_client() -> AsyncOpenAI:
    """Direct API client for expensive model (with OpenRouter fallback)."""
    return _wrap_error_detection(AsyncOpenAI(
        base_url=EXPENSIVE_BASE_URL,
        api_key=EXPENSIVE_API_KEY,
        max_retries=1, timeout=httpx.Timeout(connect=5.0, read=180.0, write=120.0, pool=5.0),
    ))


def _build_engy_client() -> AsyncOpenAI:
    """Engy client — decentralized verified inference (Bittensor SN53).

    max_retries=0: engy is a cheap provider, failures are EXPECTED. The SDK's
    auto-retry doubles every slow 503 wait (~30s each when engy is down);
    call_cheap's own retry/fallback loop already covers retries.
    """
    return _wrap_error_detection(AsyncOpenAI(
        base_url=ENGY_BASE_URL,
        api_key=ENGY_API_KEY,
        max_retries=0, timeout=httpx.Timeout(connect=5.0, read=180.0, write=120.0, pool=5.0),
    ))


def _client_for_model(model_id: str) -> AsyncOpenAI:
    """Resolve the correct upstream client for a model id.

    Engy-served ids -> Engy client; OpenRouter-style ids ("org/model") -> cheap
    (OpenRouter) client; native DeepSeek ids -> direct DeepSeek client.
    """
    if model_id in ENGY_MODELS:
        return _build_engy_client()
    if "/" in model_id:
        return _build_cheap_client()
    return _build_expensive_client()


def _reasoning_settings(model_id: str, force_reasoning: bool = False) -> ModelSettings | None:
    """Per-provider thinking settings (returns None = rely on provider default).

    - Engy: thinking is OFF by default, so always enable it via reasoning_effort
      to match DeepSeek direct's default-thinking flash. (Engy ignores the
      OpenRouter reasoning.enabled flag.)
    - OpenRouter: only when force_reasoning — qwen3.7-flash needs reasoning.enabled
      to emit structured tool calls, and only the answer path forces it.
    - DeepSeek direct: never needed — thinking already ON by default.
    """
    if model_id in ENGY_MODELS:
        return ModelSettings(extra_body={"reasoning_effort": "high"})
    if "/" in model_id and force_reasoning:
        return ModelSettings(extra_body={"reasoning": {"enabled": True}})
    return None


# ── Provider circuit breaker ────────────────────────────────────────
#
# Cheap providers (engy/Bittensor) are expected to fail. Without a breaker,
# every request burns the full deadline on a known-dead upstream before
# failing over. After PROVIDER_FAIL_THRESHOLD failures of the PRIMARY cheap
# model within PROVIDER_COOLDOWN_SECONDS, the provider is skipped entirely
# and call_cheap goes straight to the fallback model. Circuit auto-closes
# after the cooldown window (next request retries the provider).

_PROVIDER_FAILURES: dict[str, list[float]] = {}


class CircuitOpenError(Exception):
    """Raised when a provider's circuit is open — skip it, use fallback."""


def _record_provider_failure(model_id: str) -> None:
    now = time.time()
    fails = [t for t in _PROVIDER_FAILURES.get(model_id, []) if now - t < PROVIDER_COOLDOWN_SECONDS]
    fails.append(now)
    _PROVIDER_FAILURES[model_id] = fails
    if len(fails) >= PROVIDER_FAIL_THRESHOLD:
        logger.warning(
            "Provider %s circuit OPEN for %.0fs (%d failures in window)",
            model_id, PROVIDER_COOLDOWN_SECONDS, len(fails),
        )


def _provider_circuit_open(model_id: str) -> bool:
    now = time.time()
    fails = [t for t in _PROVIDER_FAILURES.get(model_id, []) if now - t < PROVIDER_COOLDOWN_SECONDS]
    _PROVIDER_FAILURES[model_id] = fails
    return len(fails) >= PROVIDER_FAIL_THRESHOLD


# ── Agent wrappers ────────────────────────────────────────────────


async def _run_cheap_agent(
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    tools: list | None,
    reasoning: bool,
) -> str:
    """Run the cheap Agent for ONE specific model. Raises on failure — the
    caller (call_cheap) owns retry and fallback policy."""
    if _provider_circuit_open(model_id):
        raise CircuitOpenError(f"circuit open for {model_id}")
    client = _client_for_model(model_id)

    # Build the SDK model
    sdk_model = OpenAIChatCompletionsModel(
        model=model_id,
        openai_client=client,
    )

    use_tools = ALL_TOOLS if tools is None else (tools or [])

    # Extract system + user messages for the Agent
    # If no explicit system message, use the first user message as instructions
    # (critical for cache reformulation: the context prompt needs to guide behavior)
    system_prompt = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt += m["content"] + "\n"
        else:
            user_messages.append(m)

    # When the only "user" message is actually a context prompt (cache reformulation),
    # use it as instructions and construct a clean user input
    if not system_prompt and len(user_messages) == 1:
        content = user_messages[0]["content"]
        if "IMPORTANT — RELEVANCE CHECK" in content or "FOUNDATION to build on" in content:
            # Split: instructions from the context prompt, user query from the last line
            lines = content.split("\n")
            instructions = content
            # Extract User's question line for cleaner input
            user_query_line = ""
            for i, line in enumerate(lines):
                if line.startswith("User's question:"):
                    user_query_line = line.replace("User's question:", "").strip()
                    break
            system_prompt = instructions
            user_messages = [{"role": "user", "content": user_query_line or user_messages[0]["content"]}]

    agent_kwargs = dict(
        name="LowCostLLM-Cheap",
        instructions=(
            system_prompt.strip()
            or "You are a helpful assistant. Use web_search to find current information whenever needed."
        )
        + _FILE_TOOL_HINT,
        tools=use_tools,
        model=sdk_model,
    )
    ms = _reasoning_settings(model_id, force_reasoning=reasoning)
    if ms is None:
        ms = ModelSettings()
    # The agent SDK sends max_tokens ONLY from ModelSettings; without this the
    # request goes out with max_tokens omitted and the provider's default
    # applies (OpenRouter defaults vary by model). Enforce the caller's cap.
    ms.max_tokens = max_tokens
    agent_kwargs["model_settings"] = ms
    agent = Agent(**agent_kwargs)

    # Convert messages to input string
    input_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in user_messages[-5:]
    )

    # Wall-clock deadline: a stuck upstream that trickles bytes never trips
    # the read idle timeout — without this the whole request can hang forever.
    # Deadline exceeded raises asyncio.TimeoutError, which call_cheap treats
    # as "stuck" → straight to the fallback model (no pointless retries).
    result = await asyncio.wait_for(
        Runner.run(agent, input=input_text, max_turns=30),
        timeout=LLM_DEADLINE_SECONDS,
    )

    # Track usage (SDK 0.20.0: usage on raw ModelResponses, not RunResult)
    if result:
        global _last_usage
        _last_usage = _extract_usage(result, model_id)

    return result.final_output if result else "(no response)"


async def call_cheap(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: list | None = None,
    reasoning: bool = False,
) -> str:
    """Call cheap model with retries and a configurable fallback model.

    Retries the primary cheap model up to CHEAP_FALLBACK_RETRIES attempts (the
    OpenAI client additionally retries 429/5xx once per attempt). If every
    attempt raises — rate limit, 5xx, timeout — switches to the fallback model
    (set via /model -cb or CHEAP_FALLBACK_MODEL env) and returns ITS answer.
    Only if the fallback also fails does it return the usual "(error: ...)".

    reasoning=True enables model thinking (OpenRouter `reasoning.enabled`), which
    qwen3.7-flash needs to emit structured tool calls (otherwise it writes code
    as plain text). Only set it on the answer path, NOT the search-key path.
    """
    primary = get_cheap_model()
    attempts = max(1, CHEAP_FALLBACK_RETRIES)
    last_err: Exception | None = None

    for i in range(attempts):
        try:
            return await _run_cheap_agent(primary, messages, temperature, max_tokens, tools, reasoning)
        except CircuitOpenError as e:
            # Provider circuit open — no point trying it again right now.
            last_err = e
            logger.warning("Cheap model %s circuit OPEN — going straight to fallback", primary)
            break
        except asyncio.TimeoutError as e:
            # Wall-clock deadline exceeded = upstream STUCK, not a transient
            # blip. Retrying a dead upstream just burns minutes — go straight
            # to the fallback model.
            last_err = e
            _record_provider_failure(primary)
            logger.warning(
                "Cheap model %s STUCK (deadline %.0fs) — skipping retries, falling back",
                primary, LLM_DEADLINE_SECONDS,
            )
            break
        except Exception as e:
            last_err = e
            _record_provider_failure(primary)
            logger.warning("Cheap model %s attempt %d/%d failed: %s", primary, i + 1, attempts, e)
            if i < attempts - 1:
                await asyncio.sleep(0.5 * (i + 1))

    fallback = get_cheap_fallback_model()
    if fallback and fallback != primary:
        logger.warning(
            "Cheap model %s failed after %d attempts — falling back to %s",
            primary, attempts, fallback,
        )
        try:
            return await _run_cheap_agent(fallback, messages, temperature, max_tokens, tools, reasoning)
        except Exception as e:
            last_err = e
            logger.warning("Cheap fallback %s also failed: %s", fallback, e)

    return f"(error: {last_err})"


async def call_cheap_raw(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    **client_kwargs,
) -> tuple[Any, str]:
    """Raw chat.completions call on the cheap model, with the same retry +
    fallback policy as call_cheap.

    For callers that need the raw response object (code path judge/writer)
    instead of the Agent SDK. Returns (response, model_used). Raises if both
    the primary and the fallback fail.
    """
    primary = get_cheap_model()
    attempts = max(1, CHEAP_FALLBACK_RETRIES)
    last_err: Exception | None = None

    for i in range(attempts):
        try:
            client = _client_for_model(primary)
            if _provider_circuit_open(primary):
                raise CircuitOpenError(f"circuit open for {primary}")
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=primary,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **client_kwargs,
                ),
                timeout=LLM_DEADLINE_SECONDS,
            )
            return resp, primary
        except CircuitOpenError as e:
            # Circuit open — skip remaining primary retries, go to fallback.
            last_err = e
            logger.warning("Cheap raw %s circuit OPEN — going straight to fallback", primary)
            break
        except asyncio.TimeoutError as e:
            # Stuck upstream — skip remaining primary retries, go to fallback.
            last_err = e
            _record_provider_failure(primary)
            logger.warning(
                "Cheap raw %s STUCK (deadline %.0fs) — skipping retries, falling back",
                primary, LLM_DEADLINE_SECONDS,
            )
            break
        except Exception as e:
            last_err = e
            _record_provider_failure(primary)
            logger.warning("Cheap raw %s attempt %d/%d failed: %s", primary, i + 1, attempts, e)
            if i < attempts - 1:
                await asyncio.sleep(0.5 * (i + 1))

    fallback = get_cheap_fallback_model()
    if fallback and fallback != primary:
        logger.warning(
            "Cheap raw %s failed after %d attempts — falling back to %s",
            primary, attempts, fallback,
        )
        try:
            client = _client_for_model(fallback)
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=fallback,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **client_kwargs,
                ),
                timeout=LLM_DEADLINE_SECONDS,
            )
            return resp, fallback
        except Exception as e:
            last_err = e
            logger.warning("Cheap raw fallback %s also failed: %s", fallback, e)

    raise last_err  # type: ignore[misc]


async def call_expensive(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> tuple[str, str]:
    """Call expensive model with tools. Falls back to OpenRouter on failure.

    Returns (response_text, model_used).
    """
    model_id = get_expensive_model()

    # Route by model id: Engy ids -> Engy, OpenRouter ("org/model") -> OpenRouter,
    # native DeepSeek ids -> DeepSeek direct. Non-OpenRouter failures fall back
    # to OpenRouter.
    client = _client_for_model(model_id)
    try:
        text = await _call_expensive_with_client(
            messages, temperature, max_tokens, client, model_id
        )
        return text, model_id
    except Exception as e:
        logger.warning(f"Expensive {model_id} failed ({e}), falling back to OpenRouter")

    # Fallback: OpenRouter. Engy's native id (deepseek-v4-flash-0731) isn't
    # served by OpenRouter, so swap in the OpenRouter-equivalent model.
    fallback_id = model_id
    if model_id in ENGY_MODELS:
        fallback_id = ENGY_FALLBACK_MODEL
    text = await _call_expensive_with_client(
        messages, temperature, max_tokens, _build_cheap_client(), fallback_id
    )
    return text, f"{model_id} (fallback)"


async def _call_expensive_with_client(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    client: AsyncOpenAI,
    model_id: str,
) -> str:
    """Internal: call expensive model through the given client."""
    sdk_model = OpenAIChatCompletionsModel(
        model=model_id,
        openai_client=client,
    )

    system_prompt = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt += m["content"] + "\n"
        else:
            user_messages.append(m)

    agent_kwargs = dict(
        name="LowCostLLM-Expensive",
        instructions=(system_prompt.strip() or "You are a helpful assistant with web search and browsing tools.")
        + _FILE_TOOL_HINT,
        tools=ALL_TOOLS,
        model=sdk_model,
    )
    ms = _reasoning_settings(model_id)
    if ms is None:
        ms = ModelSettings()
    # Same max_tokens wiring as _run_cheap_agent — see comment there.
    ms.max_tokens = max_tokens
    agent_kwargs["model_settings"] = ms
    agent = Agent(**agent_kwargs)

    input_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in user_messages[-10:]
    )

    # Wall-clock deadline — same stuck-upstream protection as the cheap agent.
    result = await asyncio.wait_for(
        Runner.run(agent, input=input_text, max_turns=30),
        timeout=LLM_DEADLINE_SECONDS,
    )

    # Track usage (SDK 0.20.0: usage on raw ModelResponses, not RunResult)
    if result:
        global _last_usage
        _last_usage = _extract_usage(result, model_id)

    return result.final_output if result else "(no response)"


async def call_curator_verdict(cached_question: str, cached_answer: str) -> str:
    """Ask the expensive model to judge a rejected cache entry (EVICT/KEEP).

    No tools, single turn, plain text — a cheap verdict, not a full answer.
    Routes like call_expensive: OpenRouter IDs ("org/model") go via the OR
    client, native DeepSeek IDs via the direct client.
    """
    from curator import build_curator_messages

    model_id = get_expensive_model()
    client = _client_for_model(model_id)

    msgs = build_curator_messages(cached_question, cached_answer)
    sdk_model = OpenAIChatCompletionsModel(model=model_id, openai_client=client)
    agent_kwargs = dict(
        name="CacheCurator",
        instructions=msgs[0]["content"],
        tools=[],
        model=sdk_model,
    )
    ms = _reasoning_settings(model_id)
    if ms is not None:
        agent_kwargs["model_settings"] = ms
    agent = Agent(**agent_kwargs)
    try:
        result = await asyncio.wait_for(
            Runner.run(agent, input=msgs[1]["content"], max_turns=1),
            timeout=LLM_DEADLINE_SECONDS,
        )
        return result.final_output if result else ""
    except Exception as e:
        logger.warning(f"Curator verdict call failed: {e}")
        return ""


async def call_expensive_stream(
    messages: list[dict],
    callback,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> tuple[str, str]:
    """Streaming variant — falls back to non-streaming for now."""
    text, model = await call_expensive(messages, temperature, max_tokens)
    if asyncio.iscoroutinefunction(callback):
        await callback(text)
    else:
        callback(text)
    return text, model


# ── OpenCode-compatible wrappers ──────────────────────────────────
#
# Tool-calling contract (Option B): when the CLIENT supplies tools, the proxy
# must NOT run them through the Agent SDK loop (it would try to execute the
# client's raw schema dicts). Instead it does a raw chat.completions call with
# those tools and returns the model's tool_calls to the client, which executes
# them and loops back. Without client tools, the agentic path (proxy's own
# ALL_TOOLS) is unchanged.


def _raw_response_dict(resp, model_used: str) -> dict:
    """Convert a raw chat.completions response into the proxy dict format."""
    choice = resp.choices[0]
    msg = choice.message
    tool_calls = None
    if getattr(msg, "tool_calls", None):
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
    }
    return {
        "content": msg.content or "",
        "tool_calls": tool_calls,
        "model": model_used,
        "usage": usage,
        "finish_reason": choice.finish_reason or "stop",
    }


def _tool_reasoning_kwargs() -> dict:
    """Reasoning settings for raw tool-calling requests.

    OpenRouter cheap models (qwen3.7-flash) need reasoning ENABLED to emit
    structured tool_calls instead of code-as-text. Engy / native DeepSeek ids
    default to thinking on — no override needed.
    """
    if "/" in get_cheap_model():
        return {"extra_body": {"reasoning": {"enabled": True}}}
    return {}


async def _call_expensive_raw(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    tools: list,
) -> tuple[Any, str]:
    """Raw chat.completions call on the expensive model with client tools.

    Routes like call_expensive (Engy / OpenRouter / DeepSeek direct) and falls
    back to OpenRouter on failure. Returns (response, model_used).
    """
    model_id = get_expensive_model()
    client = _client_for_model(model_id)
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        return resp, model_id
    except Exception as e:
        logger.warning(f"Expensive raw {model_id} failed ({e}), falling back to OpenRouter")

    fallback_id = ENGY_FALLBACK_MODEL if model_id in ENGY_MODELS else model_id
    client = _build_cheap_client()
    resp = await client.chat.completions.create(
        model=fallback_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
    )
    return resp, f"{model_id} (fallback)"


async def call_expensive_full(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: list | None = None,
) -> dict:
    """Call expensive model — returns full response dict for proxy compatibility.

    With client tools: raw chat.completions, tool_calls returned to the client.
    Without tools: agentic path (proxy's own ALL_TOOLS run inside the agent).
    """
    if tools:
        resp, model = await _call_expensive_raw(messages, temperature, max_tokens, tools)
        return _raw_response_dict(resp, model)
    text, model = await call_expensive(messages, temperature, max_tokens)
    return {
        "content": text,
        "tool_calls": None,
        "model": model,
        "usage": get_last_usage(),
        "finish_reason": "stop",
    }


async def stream_expensive_full(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: list | None = None,
):
    """Streaming wrapper — async generator yielding SSE-style chunk dicts.

    With client tools: true streaming via raw chat.completions (tool_calls
    deltas included). Without tools: agentic path, buffered then re-chunked so
    the proxy sees the same chunk shape either way.
    """
    if tools:
        model_id = get_expensive_model()
        client = _client_for_model(model_id)
        stream = None
        try:
            stream = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as e:
            logger.warning(
                f"Expensive stream {model_id} with stream_options failed ({e}), "
                "retrying without usage tracking"
            )
            try:
                stream = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    stream=True,
                )
            except Exception as e2:
                logger.warning(
                    f"Expensive stream {model_id} failed ({e2}), falling back to OpenRouter"
                )
                model_id = ENGY_FALLBACK_MODEL if model_id in ENGY_MODELS else model_id
                client = _build_cheap_client()
                stream = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    stream=True,
                )

        async for chunk in stream:
            if not chunk.choices:
                # usage-only chunk (stream_options include_usage final chunk)
                if getattr(chunk, "usage", None):
                    yield {
                        "delta": {},
                        "model": model_id,
                        "finish_reason": None,
                        "usage": {
                            "prompt_tokens": chunk.usage.prompt_tokens or 0,
                            "completion_tokens": chunk.usage.completion_tokens or 0,
                            "total_tokens": chunk.usage.total_tokens or 0,
                        },
                    }
                continue
            choice = chunk.choices[0]
            delta = {}
            if choice.delta.content:
                delta["content"] = choice.delta.content
            if getattr(choice.delta, "tool_calls", None):
                delta["tool_calls"] = [
                    {
                        "index": tc.index,
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.delta.tool_calls
                ]
            yield {
                "delta": delta,
                "model": model_id,
                "finish_reason": choice.finish_reason,
            }
        return

    # No client tools — agentic path (proxy's own tools run inside the agent),
    # buffered and re-chunked to the same SSE shape.
    text, model = await call_expensive(messages, temperature, max_tokens)
    usage = get_last_usage()
    chunk_size = 16
    for i in range(0, len(text), chunk_size):
        yield {
            "delta": {"content": text[i : i + chunk_size]},
            "model": model,
            "finish_reason": None,
        }
    yield {
        "delta": {},
        "model": model,
        "finish_reason": "stop",
        "usage": usage,
    }


async def call_cheap_full(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: list | None = None,
) -> dict:
    """Call cheap model — returns full response dict for proxy compatibility.

    With client tools: raw chat.completions via call_cheap_raw (same retry +
    fallback policy), tool_calls returned to the client. Without tools:
    agentic path (proxy's own ALL_TOOLS run inside the agent).
    """
    if tools:
        resp, model = await call_cheap_raw(
            messages,
            temperature,
            max_tokens,
            tools=tools,
            **_tool_reasoning_kwargs(),
        )
        return _raw_response_dict(resp, model)
    text = await call_cheap(messages, temperature, max_tokens)
    # Report the model that ACTUALLY answered (fallback may have fired), so the
    # calling card is honest. get_last_usage()["model"] is set by _run_cheap_agent.
    used = get_last_usage().get("model") or get_cheap_model()
    return {
        "content": text,
        "tool_calls": None,
        "model": used,
        "usage": get_last_usage(),
        "finish_reason": "stop",
    }


# ── Image generation stubs (not implemented in SDK path yet) ─────

_generated_images: list[str] = []


def _clear_generated_images() -> None:
    global _generated_images
    _generated_images = []


def _get_generated_images() -> list[str]:
    return list(_generated_images)


async def _wait_for_images(timeout: float = 30.0) -> list[str]:
    return list(_generated_images)
