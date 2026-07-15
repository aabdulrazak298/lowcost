"""n8n-compatible webhook endpoint for Flask Chat integration.

Flask Chat sends POST with {"chatinput": "message", "conversation_id": 123}
and expects SSE-style JSON lines: begin → item(s) → end.
"""
import time
import json as _json
from fastapi import Request
from fastapi.responses import StreamingResponse

from processor import process_query


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

    # Delegate to shared processor
    answer, model_used, _images = await process_query(user_query, chat_history)

    # Build the n8n-style stream
    async def stream():
        yield await _stream_json_line({
            "type": "begin",
            "metadata": {
                "nodeName": "LowCostLLM",
                "model": model_used,
                "timestamp": start_ts,
            },
        })

        # Split answer into chunks (~100 chars each for smooth streaming feel)
        chunk_size = 100
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            yield await _stream_json_line({
                "type": "item",
                "content": chunk,
                "metadata": {},
            })
            if len(answer) > 300:
                import asyncio
                await asyncio.sleep(0.01)

        yield await _stream_json_line({
            "type": "end",
            "metadata": {"timestamp": int(time.time() * 1000)},
        })

    return StreamingResponse(stream(), media_type="text/plain")
