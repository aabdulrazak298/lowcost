"""n8n-compatible webhook endpoint for Flask Chat integration.

Flask Chat sends POST with {"chatinput": "message", "conversation_id": 123}
and expects SSE-style JSON lines: begin → item(s) → end.

Supports slash commands (parity with the Telegram bot):
  /model            — show current models
  /model list       — list all available
  /model -c <name>  — swap cheap model
  /model -e <name>  — swap expensive model
  /model -cb <name> — set cheap fallback (or "off" to disable)
  /help             — command summary
"""
import time
import shlex
import json as _json
from fastapi import Request
from fastapi.responses import StreamingResponse

from processor import process_query, get_last_cache_id
from config import (
    build_calling_card, AVAILABLE_MODELS,
    get_cheap_model, get_expensive_model, get_cheap_fallback_model,
    set_cheap_model, set_expensive_model, set_cheap_fallback_model,
    WEBHOOK_DEADLINE_SECONDS,
)
from llm import set_delivery_context, clear_delivery_context


async def _stream_json_line(data: dict):
    """Yield one JSON line for the SSE stream."""
    return _json.dumps(data, ensure_ascii=False) + "\n"


def _handle_command(text: str) -> str | None:
    """Handle slash commands sent through the webhook (FlaskChat parity).

    Returns the reply text (markdown) or None if this is not a command.
    Model overrides are GLOBAL (same model_overrides row the Telegram
    bot writes), so a switch here applies to both platforms.
    """
    parts = shlex.split(text)
    parts = [p for p in parts if p]
    if not parts:
        return None

    # Strip possible bot mention (/model@SarahPetAibot)
    cmd = parts[0].split("@")[0].lower()
    rest = parts[1:]

    if cmd == "/help":
        return (
            "**LowCostLLM commands:**\n"
            f"• `/model` — show current models\n"
            f"• `/model list` — show all available\n"
            f"• `/model -c <name>` — swap cheap model\n"
            f"• `/model -e <name>` — swap expensive model\n"
            f"• `/model -cb <name>` — cheap fallback (or `off`)\n\n"
            f"Current — Cheap: `{get_cheap_model()}` · Expensive: `{get_expensive_model()}` · "
            f"Fallback: `{get_cheap_fallback_model() or 'OFF'}`"
        )

    if cmd != "/model":
        return None

    if not rest:
        return (
            "**Current models:**\n"
            f"• Cheap: `{get_cheap_model()}`\n"
            f"• Expensive: `{get_expensive_model()}`\n"
            f"• Cheap fallback: `{get_cheap_fallback_model() or 'OFF'}`\n\n"
            "`/model -c <name>` — swap cheap\n"
            "`/model -e <name>` — swap expensive\n"
            "`/model -cb <name>` — cheap fallback (or `off`)\n"
            "`/model list` — show all available"
        )

    sub = rest[0].lower()

    if sub == "list":
        lines = ["**Available models:**"]
        for key, (full_id, provider) in AVAILABLE_MODELS.items():
            lines.append(f"• `{key}` → {full_id} ({provider})")
        return "\n".join(lines)

    if sub in ("-c", "-e", "cheap", "expensive"):
        is_cheap = sub in ("-c", "cheap")
        label = "cheap" if is_cheap else "expensive"
        if len(rest) < 2:
            current = get_cheap_model() if is_cheap else get_expensive_model()
            return (
                f"Current {label}: `{current}`\n"
                f"Usage: `/model {sub} <name>`\n"
                "Use `/model list` to see options."
            )
        key = rest[1].lower()
        if key not in AVAILABLE_MODELS:
            return f"❌ Unknown model: `{key}`\nUse `/model list` to see options."
        full_id, provider = AVAILABLE_MODELS[key]
        if is_cheap:
            set_cheap_model(full_id)
            label = "Cheap"
        else:
            set_expensive_model(full_id)
            label = "Expensive"
        return f"✅ {label} → `{full_id}` ({provider})"

    if sub in ("-cb", "cheapfallback"):
        if len(rest) < 2:
            current = get_cheap_fallback_model()
            return (
                f"Current cheap fallback: `{current or 'OFF'}`\n"
                f"Usage: `/model -cb <name>`  (or `/model -cb off`)\n"
                "Use `/model list` to see options."
            )
        key = rest[1].lower()
        if key in ("off", "none", "disable", "-"):
            set_cheap_fallback_model(None)
            return "✅ Cheap fallback → OFF"
        if key not in AVAILABLE_MODELS:
            return f"❌ Unknown model: `{key}`\nUse `/model list` to see options."
        full_id, provider = AVAILABLE_MODELS[key]
        set_cheap_fallback_model(full_id)
        return f"✅ Cheap fallback → `{full_id}` ({provider})"

    return (
        "Unknown `/model` option. Usage:\n"
        "`/model` — show current\n"
        "`/model list` — show all available\n"
        "`/model -c <name>` — swap cheap\n"
        "`/model -e <name>` — swap expensive\n"
        "`/model -cb <name>` — cheap fallback (or `off`)"
    )


async def handle_webhook_chat(request: Request):
    """Handle Flask Chat webhook — n8n-compatible streaming response.

    Accepts:  {"chatinput": "user message", "conversation_id": 123}
    Returns:  SSE lines (begin → items → end)
    """
    body = await request.json()
    user_query = body.get("chatinput", "")
    chat_history = body.get("chat_history", "")

    # Slash-command parity with the Telegram bot (e.g. /model -c qwen)
    if isinstance(user_query, str):
        command_reply = _handle_command(user_query)
        if command_reply is not None:
            ts = int(time.time() * 1000)

            async def command_stream():
                yield await _stream_json_line({
                    "type": "begin",
                    "metadata": {"nodeName": "LowCostLLM", "timestamp": ts},
                })
                yield await _stream_json_line({
                    "type": "item",
                    "content": command_reply,
                    "metadata": {},
                })
                yield await _stream_json_line({
                    "type": "end",
                    "metadata": {"timestamp": int(time.time() * 1000)},
                })

            return StreamingResponse(command_stream(), media_type="text/plain")

    if not user_query:
        async def error_stream():
            yield await _stream_json_line({
                "type": "begin",
                "metadata": {"nodeName": "LowCostLLM", "timestamp": int(time.time() * 1000)},
            })
            yield await _stream_json_line({
                "type": "item",
                "content": "Error: No chatinput provided.",
                "metadata": {},
            })
            yield await _stream_json_line({
                "type": "end",
                "metadata": {"timestamp": int(time.time() * 1000)},
            })
        return StreamingResponse(error_stream(), media_type="text/plain")

    start_ts = int(time.time() * 1000)

    # Run processing in background with heartbeat to keep FlaskChat alive
    import asyncio
    result_answer = None
    result_model = None
    result_images = []
    result_footer = ""
    _exception = None

    async def _process():
        nonlocal result_answer, result_model, result_images, result_footer, _exception
        try:
            set_delivery_context("web")
            try:
                answer, model_used, _images, usage = await process_query(user_query, chat_history)
            finally:
                clear_delivery_context()
            result_answer = answer
            result_model = model_used
            result_images = _images or []
            result_footer = build_calling_card(model_used, usage, cache_id=get_last_cache_id())
        except Exception as e:
            _exception = e

    process_task = asyncio.create_task(_process())

    async def stream():
        yield await _stream_json_line({
            "type": "begin",
            "metadata": {
                "nodeName": "LowCostLLM",
                "timestamp": start_ts,
            },
        })

        # Heartbeat loop — pulse every 10s. Backstop: end the stream with a
        # clean error if processing exceeds the deadline (FlaskChat's own read
        # timeout would otherwise kill the run mid-stream with a brutal 504).
        heartbeat = 0
        deadline_ms = start_ts + int(WEBHOOK_DEADLINE_SECONDS * 1000)
        while not process_task.done():
            done, _ = await asyncio.wait([process_task], timeout=10)
            if not done:
                heartbeat += 1
                if time.time() * 1000 > deadline_ms:
                    process_task.cancel()
                    yield await _stream_json_line({
                        "type": "item",
                        "content": (
                            f"\n\nError: processing timed out after "
                            f"{int(WEBHOOK_DEADLINE_SECONDS)}s — try again."
                        ),
                        "metadata": {},
                    })
                    # MUST send a proper end event — FlaskChat's parser
                    # reports "Response ended prematurely" if the stream
                    # closes without it.
                    yield await _stream_json_line({
                        "type": "end",
                        "metadata": {
                            "nodeName": "LowCostLLM",
                            "error": "processing timed out",
                            "timestamp": int(time.time() * 1000),
                        },
                    })
                    return
                yield await _stream_json_line({
                    "type": "item",
                    "content": "",
                    "metadata": {},
                })

        if _exception:
            yield await _stream_json_line({
                "type": "item",
                "content": f"\n\nError: {str(_exception)[:200]}",
                "metadata": {},
            })
        elif result_answer:
            # Split answer into chunks (~100 chars for smooth streaming)
            chunk_size = 100
            for i in range(0, len(result_answer), chunk_size):
                chunk = result_answer[i:i + chunk_size]
                yield await _stream_json_line({
                    "type": "item",
                    "content": chunk,
                    "metadata": {},
                })
                if len(result_answer) > 300:
                    await asyncio.sleep(0.01)

        # Embed generated images inline (markdown renders them in FlaskChat)
        for url in result_images:
            yield await _stream_json_line({
                "type": "item",
                "content": f"\n\n![Generated image]({url})\n",
                "metadata": {},
            })

        # Calling card — model + real token cost, matching the Telegram footer
        if result_footer:
            yield await _stream_json_line({
                "type": "item",
                "content": result_footer,
                "metadata": {},
            })

        yield await _stream_json_line({
            "type": "end",
            "metadata": {"nodeName": "LowCostLLM", "error": str(_exception)[:200] if _exception else None, "timestamp": int(time.time() * 1000)},
        })

    return StreamingResponse(stream(), media_type="text/plain")
