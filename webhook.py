"""n8n-compatible webhook endpoint for Flask Chat integration.

Flask Chat sends POST with {"chatinput": "message", "conversation_id": 123}
and expects SSE-style JSON lines: begin → item(s) → end.
"""
import time
import json as _json
from fastapi import Request
from fastapi.responses import StreamingResponse

from processor import process_query
from llm import set_delivery_context, clear_delivery_context


async def _stream_json_line(data: dict):
    """Yield one JSON line for the SSE stream."""
    return _json.dumps(data, ensure_ascii=False) + "\n"


async def handle_webhook_chat(request: Request):
    """Handle Flask Chat webhook — n8n-compatible streaming response.

    Accepts:  {"chatinput": "user message", "conversation_id": 123}
    Returns:  SSE lines (begin → items → end)
    """
    body = await request.json()
    user_query = body.get("chatinput", "")
    chat_history = body.get("chat_history", "")

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
    _exception = None

    async def _process():
        nonlocal result_answer, result_model, result_images, _exception
        try:
            set_delivery_context("web")
            try:
                answer, model_used, _images = await process_query(user_query, chat_history)
            finally:
                clear_delivery_context()
            result_answer = answer
            result_model = model_used
            result_images = _images or []
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

        # Heartbeat loop — pulse every 10s
        heartbeat = 0
        while not process_task.done():
            done, _ = await asyncio.wait([process_task], timeout=10)
            if not done:
                heartbeat += 1
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
        else:
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

        yield await _stream_json_line({
            "type": "end",
            "metadata": {"nodeName": "LowCostLLM", "error": str(_exception)[:200] if _exception else None, "timestamp": int(time.time() * 1000)},
        })

    return StreamingResponse(stream(), media_type="text/plain")
