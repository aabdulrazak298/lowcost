"""Async LLM callers — cheap model (Qwen via OR) and expensive (DeepSeek V4 Pro direct + tools)."""
import asyncio
import json
import httpx
from config import (
    CHEAP_API_KEY,
    CHEAP_BASE_URL,
    CHEAP_MODEL,
    EXPENSIVE_API_KEY,
    EXPENSIVE_BASE_URL,
    EXPENSIVE_MODEL,
    FALLBACK_MODEL,
    FALLBACK_API_KEY,
    FALLBACK_BASE_URL,
    UPSTREAM_TIMEOUT,
    UPSTREAM_MAX_RETRIES,
)

CHEAP_CHAT_URL = f"{CHEAP_BASE_URL}/chat/completions"
EXPENSIVE_CHAT_URL = f"{EXPENSIVE_BASE_URL}/chat/completions"
SEARXNG_URL = "http://127.0.0.1:8080/search?format=json"

# ── Tool definitions ──────────────────────────────────────────────

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for current, up-to-date information. "
            "Use when you need facts beyond your knowledge cutoff, "
            "recent news, live data, or current events."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — be specific.",
                }
            },
            "required": ["query"],
        },
    },
}

TOOLS = [WEB_SEARCH_TOOL]

# ── YouTube transcript tool ───────────────────────────────────────

YOUTUBE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_youtube_transcript",
        "description": (
            "Fetch the full transcript of a YouTube video. "
            "Use this when the user asks about a YouTube video's content, "
            "wants a summary, or references a specific video. "
            "Returns the transcript with timestamps and video metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "video_url": {
                    "type": "string",
                    "description": (
                        "YouTube video URL or ID. Supports: "
                        "youtube.com/watch?v=ID, youtu.be/ID, youtube.com/shorts/ID, "
                        "or bare video ID."
                    ),
                }
            },
            "required": ["video_url"],
        },
    },
}

CODE_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "run_code",
        "description": (
            "Execute Python code in a secure sandbox. Use for calculations, "
            "data analysis, unit conversions, or quick math. "
            "The output of the last expression is returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."}
            },
            "required": ["code"],
        },
    },
}

IMAGE_GEN_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an AI image from a text description.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Image description."}
            },
            "required": ["prompt"],
        },
    },
}

PLOT_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_plot",
        "description": (
            "Generate a chart or graph using matplotlib. Provide Python code that "
            "creates a plot using matplotlib — the output image will be sent to the user. "
            "Use `import matplotlib.pyplot as plt` and call `plt.savefig('/tmp/plot.png')` "
            "then print 'DONE'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code using matplotlib to generate a chart."}
            },
            "required": ["code"],
        },
    },
}

IMAGE_ANALYZE_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_image",
        "description": "Analyze or describe an image from a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {"type": "string", "description": "Image URL."},
                "question": {
                    "type": "string",
                    "description": "What to ask about the image.",
                },
            },
            "required": ["image_url"],
        },
    },
}

FETCH_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_webpage",
        "description": (
            "Fetch and read the full text content of a webpage. "
            "Use when the user asks you to read, summarize, or extract "
            "information from a specific URL or website."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL of the webpage to fetch (e.g. https://example.com/page).",
                }
            },
            "required": ["url"],
        },
    },
}

ALL_TOOLS = [
    WEB_SEARCH_TOOL,
    YOUTUBE_TOOL,
    CODE_EXEC_TOOL,
    IMAGE_GEN_TOOL,
    PLOT_TOOL,
    IMAGE_ANALYZE_TOOL,
    FETCH_PAGE_TOOL,
]


# ── n8n API config ────────────────────────────────────────────────

N8N_BASE = "http://127.0.0.1:8000"
N8N_KEY = "987654321"


YOUTUBE_VPS_URL = "http://141.11.17.227:8000/api/youtube/script"
YOUTUBE_VPS_KEY = "987654321"


def _is_openrouter(url: str) -> bool:
    return "openrouter.ai" in url


# ── Web search ────────────────────────────────────────────────────


async def _search_web(query: str, max_results: int = 5) -> str:
    """Execute web search via local SearXNG."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                SEARXNG_URL, params={"q": query, "format": "json"}
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])[:max_results]
        if not results:
            return "No results for: " + query

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("content", "")[:300]
            lines.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")

        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


async def _execute_tool(tool_call: dict) -> dict:
    """Execute a tool call, return result message dict."""
    fn_name = tool_call["function"]["name"]
    fn_args = json.loads(tool_call["function"]["arguments"])

    if fn_name == "search_web":
        content = await _search_web(fn_args.get("query", ""))
    elif fn_name == "get_youtube_transcript":
        content = await _fetch_youtube_transcript(fn_args.get("video_url", ""))
    elif fn_name == "run_code":
        content = await _run_code(fn_args.get("code", ""))
    elif fn_name == "generate_image":
        content = await _generate_image(fn_args.get("prompt", ""))
    elif fn_name == "generate_plot":
        content = await _generate_plot(fn_args.get("code", ""))
    elif fn_name == "analyze_image":
        content = await _analyze_image(
            fn_args.get("image_url", ""),
            fn_args.get("question", "Describe this image"),
        )
    elif fn_name == "fetch_webpage":
        content = await _fetch_webpage(fn_args.get("url", ""))
    else:
        content = f"Unknown tool: {fn_name}"

    return {"role": "tool", "tool_call_id": tool_call["id"], "content": content}


# ── n8n API tool implementations ──────────────────────────────────


async def _n8n_post(endpoint: str, body: dict) -> dict:
    """Call n8n API endpoint with auth."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{N8N_BASE}{endpoint}",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {N8N_KEY}",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def _run_code(code: str) -> str:
    """Execute Python code via n8n sandbox."""
    try:
        result = await _n8n_post("/code/execute", {
            "code": code,
            "language": "python",
        })
        if not result.get("success", True):
            err = result.get("error_message") or result.get("stderr", "")
            return f"Code error: {err}"

        stdout = result.get("stdout", "").strip()
        value = result.get("result")
        if value is not None:
            return str(value)
        if stdout:
            return stdout.split("\n")[-1] if "\n" in stdout else stdout
        return "(code executed, no output)"
    except Exception as e:
        return f"Code execution failed: {e}"


async def _generate_plot(code: str) -> str:
    """Execute matplotlib code and capture the plot output."""
    import subprocess, tempfile, uuid
    from pathlib import Path

    # Ensure matplotlib is available
    wrapped = (
        "import matplotlib\nmatplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        + code +
        "\nif plt.get_fignums():\n    plt.savefig('/tmp/plot.png')\n"
    )
    try:
        result = subprocess.run(
            ["python3", "-c", wrapped],
            capture_output=True, text=True, timeout=30,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        plot_path = Path("/tmp/plot.png")
        if plot_path.exists():
            img_dir = Path(__file__).parent / "generated"
            img_dir.mkdir(exist_ok=True)
            fname = f"plot_{uuid.uuid4().hex[:8]}.png"
            fpath = img_dir / fname
            import shutil
            shutil.move(str(plot_path), str(fpath))
            _generated_images.append(str(fpath))
            return f"Plot generated: {fname}" + (f"\nOutput: {stdout}" if stdout else "")
        else:
            return f"No plot produced.\nOutput: {stdout}\nErrors: {stderr}" if stdout or stderr else "No plot produced."
    except Exception as e:
        return f"Plot generation failed: {e}"


# Track images generated during this request so Telegram can send them
_generated_images: list[str] = []
_image_tasks: list = []  # pending async generation tasks


def _clear_generated_images() -> None:
    _generated_images.clear()
    _image_tasks.clear()


def _get_generated_images() -> list[str]:
    return list(_generated_images)


async def _wait_for_images() -> list[str]:
    """Wait for any pending image generation tasks to complete."""
    if _image_tasks:
        await asyncio.gather(*_image_tasks)
    return list(_generated_images)


async def _generate_image(prompt: str) -> str:
    """Generate image via OpenRouter (fire-and-forget, adds to _generated_images)."""
    import base64, uuid
    from pathlib import Path

    async def _do_generate():
        try:
            headers = {
                "Authorization": f"Bearer {CHEAP_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8800",
                "X-Title": "LowCostLLM",
            }
            payload = {"model": "google/gemini-2.5-flash-image", "prompt": prompt, "n": 1}
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/images/generations",
                    json=payload, headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            item = data.get("data", [{}])[0]
            b64 = item.get("b64_json", "")
            if b64:
                img_dir = Path(__file__).parent / "generated"
                img_dir.mkdir(exist_ok=True)
                fname = f"{uuid.uuid4().hex[:8]}.png"
                fpath = img_dir / fname
                fpath.write_bytes(base64.b64decode(b64))
                _generated_images.append(str(fpath))
        except Exception:
            pass  # failure is silent — LLM already moved on

    # Fire and forget
    task = asyncio.create_task(_do_generate())
    _image_tasks.append(task)

    return f"Image generation started for: {prompt[:80]}"


async def _analyze_image(image_url: str, question: str) -> str:
    """Analyze image via n8n vision model."""
    try:
        result = await _n8n_post("/image/analyze", {
            "image_url": image_url,
            "question": question,
        })
        return result.get("analysis", result.get("response", str(result)))
    except Exception as e:
        return f"Image analysis failed: {e}"


# ── Webpage fetch via n8n ─────────────────────────────────────────


async def _fetch_webpage(url: str) -> str:
    """Fetch webpage text content via n8n text/load API."""
    try:
        # Step 1: Load the URL
        load_result = await _n8n_post("/text/load", {"url": url})
        file_id = load_result.get("file_id")
        if not file_id:
            return f"Failed to load page: {load_result}"

        # Step 2: Get chunk 0 (first chunk, up to ~50K chars)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{N8N_BASE}/text/{file_id}/chunk/0",
                headers={"Authorization": f"Bearer {N8N_KEY}"},
            )
            resp.raise_for_status()
            chunk = resp.json()

        text = chunk.get("content", chunk.get("text", str(chunk)))
        # Truncate to reasonable size for the model
        if len(text) > 8000:
            text = text[:8000] + "\n\n... (truncated, page is longer)"
        return text

    except Exception as e:
        return f"Failed to fetch webpage: {e}"


# ── YouTube transcript fetch ──────────────────────────────────────


async def _fetch_youtube_transcript(video_url: str) -> str:
    """Fetch transcript from YouTube via the second VPS."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                YOUTUBE_VPS_URL,
                json={"video_url_or_id": video_url},
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": YOUTUBE_VPS_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if not data.get("transcript_available"):
            return (
                f"No transcript available for this video.\n"
                f"Title: {data.get('metadata', {}).get('title', 'Unknown')}"
            )

        meta = data.get("metadata", {})
        segments = data.get("transcript", [])

        lines = [
            f"Title: {meta.get('title', 'Unknown')}",
            f"Duration: {int(meta.get('duration', 0)) // 60} min",
            f"Segments: {len(segments)}",
            "",
        ]

        for seg in segments:
            s = int(seg["start"])
            ts = f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
            lines.append(f"[{ts}] {seg['text']}")

        return "\n".join(lines)

    except Exception as e:
        return f"YouTube transcript fetch error: {e}"


# ── LLM callers ───────────────────────────────────────────────────


async def call_cheap(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: list | None = None,
) -> str:
    """Call the cheap model with optional tool calling. Retries on known errors.

    Supports web search tool so cached answers can be augmented with
    fresh information when the similar query needs it.
    """
    import logging
    _log = logging.getLogger(__name__)

    headers = {
        "Authorization": f"Bearer {CHEAP_API_KEY}",
        "Content-Type": "application/json",
    }
    if _is_openrouter(CHEAP_BASE_URL):
        headers["HTTP-Referer"] = "http://localhost:8800"
        headers["X-Title"] = "LowCostLLM"

    # None = ALL_TOOLS, [] = no tools
    use_tools = ALL_TOOLS if tools is None else tools

    last_error = None

    for attempt in range(UPSTREAM_MAX_RETRIES):
        if attempt > 0:
            _log.info(f"Retry {attempt + 1}/3 after: {last_error}")
            temp = temperature + (attempt * 0.15)  # bump temperature each retry
        else:
            temp = temperature

        try:
            result = await _call_cheap_once(
                messages, temp, max_tokens, use_tools, headers
            )
            # If result is a string (error signal), it means we should retry
            if result == "__LOOP__":
                last_error = "tool loop"
                use_tools = []  # disable tools on retry after loop
                continue
            if result == "__MAX_ROUNDS__":
                last_error = "max rounds"
                continue  # retry with higher temperature
            return result
        except Exception as e:
            last_error = str(e)[:80]
            _log.warning(f"Attempt {attempt + 1} failed: {last_error}")
            if attempt < 2:
                continue
            raise  # all 3 attempts failed

    return "(all retries exhausted — giving up)"


async def _call_cheap_once(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    use_tools: list | None,
    headers: dict,
) -> str:
    """Single call to cheap model — returns str or error signal."""
    conversation = list(messages)
    last_tool_calls = []

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        for _ in range(50):
            payload = {
                "model": CHEAP_MODEL,
                "messages": conversation,
                "temperature": temperature,
                "max_tokens": max_tokens or 8192,
            }
            if use_tools:
                payload["tools"] = use_tools
                payload["tool_choice"] = "auto"

            resp = await client.post(CHEAP_CHAT_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason", "")

            if msg.get("tool_calls") and finish in ("tool_calls", "length"):
                tc_sigs = [(tc["function"]["name"], tc["function"]["arguments"]) for tc in msg["tool_calls"]]
                if tc_sigs == last_tool_calls:
                    return "__LOOP__"  # signal to retry
                last_tool_calls = tc_sigs
                conversation.append({
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": msg["tool_calls"],
                })
                for tc in msg["tool_calls"]:
                    result = await _execute_tool(tc)
                    conversation.append(result)
                continue

            return msg.get("content", "") or msg.get("reasoning_content", "") or "(no response)"

        return "__MAX_ROUNDS__"


async def call_expensive(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> tuple[str, str]:
    """Call DeepSeek V4 Pro with native tool calling. Falls back to Qwen Flash
    via OpenRouter on API failure (circuit breaker).

    Returns (response_text, model_used).
    """
    try:
        return await _call_expensive_primary(messages, temperature, max_tokens)
    except Exception:
        # Circuit breaker — fall back to model via OpenRouter
        return await _call_expensive_fallback(messages, temperature, max_tokens)


async def _call_expensive_primary(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> tuple[str, str]:
    """Primary path: DeepSeek V4 Pro direct API."""
    headers = {
        "Authorization": f"Bearer {EXPENSIVE_API_KEY}",
        "Content-Type": "application/json",
    }

    conversation = list(messages)

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        for _ in range(50):  # max 50 LLM calls — enough for deep research
            payload: dict = {
                "model": EXPENSIVE_MODEL,
                "messages": conversation,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": ALL_TOOLS,
                "tool_choice": "auto",
            }

            resp = await client.post(
                EXPENSIVE_CHAT_URL, json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason", "")

            if msg.get("tool_calls") and finish in ("tool_calls", "length"):
                conversation.append({
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": msg["tool_calls"],
                })
                for tc in msg["tool_calls"]:
                    result = await _execute_tool(tc)
                    conversation.append(result)
                continue

            text = msg.get("content", "") or msg.get("reasoning_content", "") or "(no response)"
            model = data.get("model", EXPENSIVE_MODEL)
            return text, model

        return "(tool calling exceeded max rounds)", EXPENSIVE_MODEL


async def call_expensive_stream(
    messages: list[dict],
    callback,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> tuple[str, str]:
    """Stream DeepSeek V4 Pro response, calling callback(chunk_text) for each chunk.

    Returns (full_text, model_used) when complete.
    Falls back to non-streaming if streaming fails.
    """
    import logging
    _logger = logging.getLogger(__name__)
    try:
        return await _call_expensive_stream_primary(messages, callback, temperature, max_tokens)
    except Exception as e:
        _logger.exception(f"Stream failed, falling back: {e}")
        # Fall back to non-streaming
        text, model = await _call_expensive_primary(messages, temperature, max_tokens)
        if asyncio.iscoroutinefunction(callback):
            await callback(text)
        else:
            callback(text)
        return text, model


async def _call_expensive_stream_primary(
    messages: list[dict],
    callback,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> tuple[str, str]:
    """Stream from DeepSeek V4 Pro with tool calling support."""
    headers = {
        "Authorization": f"Bearer {EXPENSIVE_API_KEY}",
        "Content-Type": "application/json",
    }

    conversation = list(messages)
    full_text = ""

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        for _ in range(20):
            payload: dict = {
                "model": EXPENSIVE_MODEL,
                "messages": conversation,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": ALL_TOOLS,
                "tool_choice": "auto",
                "stream": True,
            }

            tool_calls = []
            current_tool = None

            async with client.stream("POST", EXPENSIVE_CHAT_URL, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except Exception:
                        continue
                    delta = data["choices"][0].get("delta", {})
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            while len(tool_calls) <= idx:
                                tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                            if "id" in tc:
                                tool_calls[idx]["id"] = tc["id"]
                            if "function" in tc:
                                if "name" in tc["function"]:
                                    tool_calls[idx]["function"]["name"] = tc["function"]["name"]
                                if "arguments" in tc["function"]:
                                    tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
                    content = delta.get("content", "") or ""
                    if content:
                        full_text += content
                        if asyncio.iscoroutinefunction(callback):
                            await callback(content)
                        else:
                            callback(content)

            if tool_calls:
                conversation.append({
                    "role": "assistant",
                    "content": full_text or None,
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    result = await _execute_tool(tc)
                    conversation.append(result)
                full_text = ""
                continue

            return full_text, EXPENSIVE_MODEL

        return "(tool calling exceeded max rounds)", EXPENSIVE_MODEL


async def _call_expensive_fallback(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> tuple[str, str]:
    """Fallback path: FALLBACK_MODEL via OpenRouter when primary is down."""
    headers = {
        "Authorization": f"Bearer {FALLBACK_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8800",
        "X-Title": "LowCostLLM",
    }

    payload = {
        "model": FALLBACK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": ALL_TOOLS,
        "tool_choice": "auto",
    }

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        resp = await client.post(
            f"{FALLBACK_BASE_URL}/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        model = f"{FALLBACK_MODEL} (fallback)"
        return text, model


# === OpenCode-compatible full-response functions ===


async def call_cheap_full(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: list | None = None,
) -> dict:
    """Call cheap model — returns full response dict with tool_calls and usage.

    When `tools` is provided, they are passed through to the model and
    returned to the caller (no local tool execution).
    """
    headers = {
        "Authorization": f"Bearer {CHEAP_API_KEY}",
        "Content-Type": "application/json",
    }
    if _is_openrouter(CHEAP_BASE_URL):
        headers["HTTP-Referer"] = "http://localhost:8800"
        headers["X-Title"] = "LowCostLLM"

    payload: dict = {
        "model": CHEAP_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or 8192,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        resp = await client.post(CHEAP_CHAT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]

        return {
            "content": msg.get("content", "") or "",
            "tool_calls": msg.get("tool_calls"),
            "model": data.get("model", CHEAP_MODEL),
            "usage": data.get("usage"),
            "finish_reason": choice.get("finish_reason", "stop"),
        }


async def call_expensive_full(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: list | None = None,
) -> dict:
    """Call expensive model — returns full response dict with tool_calls and usage.

    When `tools` is provided, they are passed through to the model and
    returned to the caller (no local tool execution).
    Falls back to FALLBACK_MODEL on API failure.
    """
    import logging
    _logger = logging.getLogger(__name__)

    headers = {
        "Authorization": f"Bearer {EXPENSIVE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict = {
        "model": EXPENSIVE_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            resp = await client.post(
                EXPENSIVE_CHAT_URL, json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]

            return {
                "content": msg.get("content", "") or "",
                "tool_calls": msg.get("tool_calls"),
                "model": data.get("model", EXPENSIVE_MODEL),
                "usage": data.get("usage"),
                "finish_reason": choice.get("finish_reason", "stop"),
            }
    except Exception:
        _logger.exception("Expensive model failed, falling back")
        fallback_headers = {
            "Authorization": f"Bearer {FALLBACK_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8800",
            "X-Title": "LowCostLLM",
        }
        fb_payload: dict = {
            "model": FALLBACK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            fb_payload["tools"] = tools
            fb_payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            resp = await client.post(
                f"{FALLBACK_BASE_URL}/chat/completions", json=fb_payload, headers=fallback_headers
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]

            return {
                "content": msg.get("content", "") or "",
                "tool_calls": msg.get("tool_calls"),
                "model": f"{FALLBACK_MODEL} (fallback)",
                "usage": data.get("usage"),
                "finish_reason": choice.get("finish_reason", "stop"),
            }


async def stream_expensive_full(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    tools: list | None = None,
):
    """Stream from expensive model — yields {delta, finish_reason, model, usage} dicts.

    When `tools` is provided, they are passed through and tool_call deltas
    are relayed to the caller (no local tool execution).
    """
    headers = {
        "Authorization": f"Bearer {EXPENSIVE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict = {
        "model": EXPENSIVE_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        async with client.stream(
            "POST", EXPENSIVE_CHAT_URL, json=payload, headers=headers
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue

                choice = data["choices"][0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")
                model = data.get("model", EXPENSIVE_MODEL)
                usage = data.get("usage")

                yield {
                    "delta": delta,
                    "finish_reason": finish_reason,
                    "model": model,
                    "usage": usage,
                }
