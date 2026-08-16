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

import httpx
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, set_tracing_disabled, ModelSettings

from config import (
    CHEAP_API_KEY,
    CHEAP_BASE_URL,
    EXPENSIVE_API_KEY,
    EXPENSIVE_BASE_URL,
    get_cheap_model,
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

    for seg in transcript[:200]:  # cap at 200 segments to avoid token overflow
        s = int(seg["start"])
        ts = f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        lines.append(f"[{ts}] {seg['text']}")

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
    """Generate a chart/graph from data. chart_type: line, bar, scatter, pie."""
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


ALL_TOOLS = [web_search, web_fetch, youtube_transcript, run_code, generate_graph, generate_image, edit_image]


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


def _build_cheap_client() -> AsyncOpenAI:
    """OpenRouter client for cheap model."""
    return AsyncOpenAI(
        base_url=CHEAP_BASE_URL,
        api_key=CHEAP_API_KEY,
        max_retries=1, timeout=httpx.Timeout(connect=5.0, read=180.0, write=120.0, pool=5.0),
        default_headers={
            "HTTP-Referer": "http://localhost:8800",
            "X-Title": "LowCostLLM",
        },
    )


def _build_expensive_client() -> AsyncOpenAI:
    """Direct API client for expensive model (with OpenRouter fallback)."""
    return AsyncOpenAI(
        base_url=EXPENSIVE_BASE_URL,
        api_key=EXPENSIVE_API_KEY,
        max_retries=1, timeout=httpx.Timeout(connect=5.0, read=180.0, write=120.0, pool=5.0),
    )


# ── Agent wrappers ────────────────────────────────────────────────


async def call_cheap(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: list | None = None,
    reasoning: bool = False,
) -> str:
    """Call cheap model via OpenRouter with optional tool calling.

    Uses OpenAI Agents SDK — provider routing handled automatically.
    reasoning=True enables model thinking (OpenRouter `reasoning.enabled`), which
    qwen3.7-flash needs to emit structured tool calls (otherwise it writes code
    as plain text). Only set it on the answer path, NOT the search-key path.
    """
    client = _build_cheap_client()
    model_id = get_cheap_model()

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
        instructions=system_prompt.strip() or "You are a helpful assistant. Use web_search to find current information whenever needed.",
        tools=use_tools,
        model=sdk_model,
    )
    if reasoning:
        # OpenRouter reasoning control via SDK ModelSettings.extra_body (merged to
        # top-level "reasoning" in the HTTP body — NOT an "extra_body" field).
        agent_kwargs["model_settings"] = ModelSettings(extra_body={"reasoning": {"enabled": True}})
    agent = Agent(**agent_kwargs)

    try:
        # Convert messages to input string
        input_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in user_messages[-5:]
        )

        result = await Runner.run(agent, input=input_text, max_turns=30)

        # Track usage
        if result and hasattr(result, "usage"):
            global _last_usage
            _last_usage = {
                "prompt_tokens": getattr(result.usage, "input_tokens", 0),
                "completion_tokens": getattr(result.usage, "output_tokens", 0),
                "model": model_id,
            }

        return result.final_output if result else "(no response)"

    except Exception as e:
        logger.warning(f"Cheap model failed: {e}")
        return f"(error: {e})"


async def call_expensive(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> tuple[str, str]:
    """Call expensive model with tools. Falls back to OpenRouter on failure.

    Returns (response_text, model_used).
    """
    model_id = get_expensive_model()

    # OpenRouter-format IDs (org/model) route straight to OpenRouter — the
    # DeepSeek direct API only recognizes native IDs (deepseek-v4-pro, etc.)
    # and rejects "upstage/solar-pro4" with a wasted round-trip. (Restores the
    # auto-detect from before the SDK migration — see skill pitfall 50.)
    is_or = "/" in model_id

    if is_or:
        text = await _call_expensive_with_client(
            messages, temperature, max_tokens, _build_cheap_client(), model_id
        )
        return text, model_id

    # Native DeepSeek ID: try direct first, fall back to OpenRouter on failure
    try:
        text = await _call_expensive_with_client(
            messages, temperature, max_tokens, _build_expensive_client(), model_id
        )
        return text, model_id
    except Exception as e:
        logger.warning(f"Expensive direct failed ({e}), falling back to OpenRouter")

    # Fallback: OpenRouter
    fallback_client = _build_cheap_client()
    text = await _call_expensive_with_client(
        messages, temperature, max_tokens, fallback_client, model_id
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

    agent = Agent(
        name="LowCostLLM-Expensive",
        instructions=system_prompt.strip() or "You are a helpful assistant with web search and browsing tools.",
        tools=ALL_TOOLS,
        model=sdk_model,
    )

    input_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in user_messages[-10:]
    )

    result = await Runner.run(agent, input=input_text, max_turns=30)

    # Track usage
    if result and hasattr(result, "usage"):
        global _last_usage
        _last_usage = {
            "prompt_tokens": getattr(result.usage, "input_tokens", 0),
            "completion_tokens": getattr(result.usage, "output_tokens", 0),
            "model": model_id,
        }

    return result.final_output if result else "(no response)"


async def call_curator_verdict(cached_question: str, cached_answer: str) -> str:
    """Ask the expensive model to judge a rejected cache entry (EVICT/KEEP).

    No tools, single turn, plain text — a cheap verdict, not a full answer.
    Routes like call_expensive: OpenRouter IDs ("org/model") go via the OR
    client, native DeepSeek IDs via the direct client.
    """
    from curator import build_curator_messages

    model_id = get_expensive_model()
    is_or = "/" in model_id
    client = _build_cheap_client() if is_or else _build_expensive_client()

    msgs = build_curator_messages(cached_question, cached_answer)
    sdk_model = OpenAIChatCompletionsModel(model=model_id, openai_client=client)
    agent = Agent(
        name="CacheCurator",
        instructions=msgs[0]["content"],
        tools=[],
        model=sdk_model,
    )
    try:
        result = await Runner.run(agent, input=msgs[1]["content"], max_turns=1)
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


async def call_expensive_full(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> dict:
    """Call expensive model — returns full response dict for proxy compatibility."""
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
) -> dict:
    """Streaming wrapper — returns same dict format for proxy compatibility."""
    text, model = await call_expensive(messages, temperature, max_tokens)
    return {
        "content": text,
        "tool_calls": None,
        "model": model,
        "usage": get_last_usage(),
        "finish_reason": "stop",
    }


async def call_cheap_full(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: list | None = None,
) -> dict:
    """Call cheap model — returns full response dict for proxy compatibility."""
    text = await call_cheap(messages, temperature, max_tokens, tools)
    return {
        "content": text,
        "tool_calls": None,
        "model": get_cheap_model(),
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
