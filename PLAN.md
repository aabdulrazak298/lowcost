# LowCostLLM — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build an OpenAI-compatible LLM proxy that caches expensive model answers in SQLite and reuses them to answer similar questions with a cheaper model, cutting API costs by routing fresh queries to Gemini 3.1 Flash and similar/repeat queries to DeepSeek V4 Flash with cached context.

**Architecture:** FastAPI server exposing `/v1/chat/completions`. Inbound query checked against SQLite cache via RapidFuzz `token_sort_ratio`. >=85% match: DeepSeek Flash answers with cached Gemini response as context. <85% match: Gemini generates fresh, answer saved to SQLite. Both models called via OpenAI-compatible HTTP (OpenRouter for Gemini, DeepSeek direct API for Flash).

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, httpx, rapidfuzz, SQLite (stdlib `sqlite3`), python-dotenv

---

## Project Structure

```
~/cloud/projects/lowcostllm/
├── PLAN.md              # This file
├── .env                 # API keys (gitignored)
├── .env.example         # Template without secrets
├── config.py            # Model names, threshold, DB path
├── main.py              # FastAPI app, startup
├── server.py            # Uvicorn launcher
├── db.py                # SQLite schema + CRUD
├── matcher.py           # RapidFuzz similarity logic
├── llm.py               # Model call functions (DeepSeek direct, Gemini via OR)
├── proxy.py             # /v1/chat/completions route + orchestrator
├── requirements.txt     # Dependencies
└── README.md            # Usage docs
```

---

## Task 1: Project scaffold and dependencies

**Objective:** Create the project skeleton with all files stubbed, venv, and dependencies installed.

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `config.py`
- Create: `main.py` (stub)
- Create: `server.py`
- Create: `db.py` (stub)
- Create: `matcher.py` (stub)
- Create: `llm.py` (stub)
- Create: `proxy.py` (stub)
- Create: `README.md` (stub)

**Step 1: Create requirements.txt**

```text
fastapi==0.115.6
uvicorn[standard]==0.34.0
httpx==0.28.1
rapidfuzz==3.12.0
python-dotenv==1.0.1
```

**Step 2: Create .env.example**

```env
# DeepSeek direct API
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash

# OpenRouter API (for Gemini)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
GEMINI_MODEL=google/gemini-3.1-flash-lite
```

**Step 3: Create config.py**

```python
"""Central configuration — all tunables live here."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root
ROOT = Path(__file__).parent
DB_PATH = ROOT / "cache.db"

# Similarity threshold (0-100) — token_sort_ratio >= this triggers cheap model
SIMILARITY_THRESHOLD = int(os.getenv("SIMILARITY_THRESHOLD", "85"))

# DeepSeek (cheap model — direct API)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Gemini via OpenRouter (expensive model)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "google/gemini-3.1-flash-lite")

# Server
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8800"))
```

**Step 4: Create .gitignore**

```gitignore
.env
cache.db
__pycache__/
*.pyc
.venv/
```

**Step 5: Set up venv and install**

```bash
cd ~/cloud/projects/lowcostllm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Verification:** `pip freeze | grep -E "fastapi|uvicorn|httpx|rapidfuzz|dotenv"` shows all 5 packages.

---

## Task 2: SQLite schema and CRUD (db.py)

**Objective:** Create the cache table and functions to insert and query stored Q&A pairs.

**Files:**
- Create: `db.py`

**Step 1: Write db.py**

```python
"""SQLite cache for Q&A pairs."""
import sqlite3
from datetime import datetime, timezone
from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS qa_cache (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query       TEXT    NOT NULL,
                answer      TEXT    NOT NULL,
                model_used  TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_qa_created
            ON qa_cache(created_at DESC)
        """)


def insert_qa(query: str, answer: str, model_used: str) -> int:
    """Insert a Q&A pair. Returns the new row ID."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO qa_cache (query, answer, model_used) VALUES (?, ?, ?)",
            (query, answer, model_used),
        )
        return cur.lastrowid


def get_all_queries() -> list[dict]:
    """Return all cached queries with their IDs, newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, query, answer, model_used, created_at FROM qa_cache ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
```

**Step 2: Verify with a quick test**

```bash
cd ~/cloud/projects/lowcostllm
source .venv/bin/activate
python3 -c "
from db import init_db, insert_qa, get_all_queries
init_db()
rid = insert_qa('how to calibrate pH meter?', 'Step 1: rinse probe...', 'gemini')
rows = get_all_queries()
print(f'Inserted id={rid}, total rows={len(rows)}')
assert rows[0]['query'] == 'how to calibrate pH meter?'
print('OK')
"
```

**Verification:** Prints "OK", `cache.db` file exists.

---

## Task 3: RapidFuzz matcher (matcher.py)

**Objective:** Given a query, find the best matching cached query above the similarity threshold.

**Files:**
- Create: `matcher.py`

**Step 1: Write matcher.py**

```python
"""Similarity matching using RapidFuzz."""
from rapidfuzz import fuzz
from config import SIMILARITY_THRESHOLD


def find_best_match(query: str, candidates: list[dict]) -> dict | None:
    """Return the best matching cached entry if its token_sort_ratio >= threshold.

    Args:
        query: The incoming user query string.
        candidates: List of dicts from get_all_queries(), each with 'id', 'query', 'answer'.

    Returns:
        The best matching dict if above threshold, else None.
    """
    if not candidates:
        return None

    best = None
    best_score = 0

    for entry in candidates:
        score = fuzz.token_sort_ratio(query, entry["query"])
        if score > best_score:
            best_score = score
            best = entry

    if best and best_score >= SIMILARITY_THRESHOLD:
        return best

    return None
```

**Step 2: Verify**

```bash
cd ~/cloud/projects/lowcostllm
source .venv/bin/activate
python3 -c "
from matcher import find_best_match

candidates = [
    {'id': 1, 'query': 'how to calibrate a pH meter', 'answer': 'Rinse probe, buffer 7, buffer 4...'},
    {'id': 2, 'query': 'what is the best VFD brand', 'answer': 'ABB and Schneider are popular...'},
]

# Exact-ish match (word order different)
result = find_best_match('calibrate pH meter how to', candidates)
assert result is not None and result['id'] == 1, f'Expected match id=1, got {result}'
print(f'Match: score implied, id={result[\"id\"]}')

# Completely different query
result2 = find_best_match('how to cook nasi lemak', candidates)
assert result2 is None, f'Expected no match, got {result2}'
print('No match for unrelated query — OK')
"
```

**Verification:** Prints match confirmation and no-match confirmation.

---

## Task 4: LLM calling functions (llm.py)

**Objective:** Async functions to call DeepSeek direct and Gemini via OpenRouter, both as OpenAI-compatible chat completions.

**Files:**
- Create: `llm.py`

**Step 1: Write llm.py**

```python
"""Async LLM callers for DeepSeek (direct) and Gemini (via OpenRouter)."""
import httpx
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    GEMINI_MODEL,
)

DEEPSEEK_CHAT_URL = f"{DEEPSEEK_BASE_URL}/chat/completions"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"


async def call_deepseek(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Call DeepSeek V4 Flash directly. Returns the response text."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(DEEPSEEK_CHAT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def call_gemini(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> tuple[str, str]:
    """Call Gemini via OpenRouter. Returns (response_text, model_used)."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8800",
        "X-Title": "LowCostLLM",
    }
    payload = {
        "model": GEMINI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(OPENROUTER_CHAT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        model = data.get("model", GEMINI_MODEL)
        return text, model
```

**Step 2: Verify (dry-run — needs real keys to fully test)**

```bash
cd ~/cloud/projects/lowcostllm
source .venv/bin/activate
python3 -c "
from llm import call_deepseek, call_gemini
print('llm.py imports OK — functions defined')
print('call_deepseek:', call_deepseek)
print('call_gemini:', call_gemini)
"
```

**Verification:** Imports succeed, function signatures confirmed.

---

## Task 5: Orchestrator + proxy route (proxy.py)

**Objective:** The core logic — extract query, match, route to correct model, save cache, return OpenAI-format response.

**Files:**
- Create: `proxy.py`

**Step 1: Write proxy.py**

```python
"""Core orchestrator: match → route → cache → respond."""
import json
import time
from fastapi import Request
from fastapi.responses import StreamingResponse

from db import get_all_queries, insert_qa
from matcher import find_best_match
from llm import call_deepseek, call_gemini


CHEAP_MODEL_CONTEXT_PROMPT = """A similar question was previously answered by an expert AI. Here is that answer for reference:

---
{expert_answer}
---

Using the expert answer above as your knowledge source, answer the user's new question accurately and helpfully. If the new question differs from the original, adapt the answer appropriately while preserving factual accuracy.

User's question: {user_query}"""


async def handle_chat_completion(request: Request) -> dict:
    """Process a /v1/chat/completions request."""
    body = await request.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 2048)

    # Extract the last user message as the query
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg["content"]
            break

    if not user_query:
        user_query = " ".join(m.get("content", "") for m in messages)

    # Check cache
    cached = get_all_queries()
    match = find_best_match(user_query, cached)

    if match:
        # --- CHEAP PATH: use DeepSeek Flash with cached context ---
        context_prompt = CHEAP_MODEL_CONTEXT_PROMPT.format(
            expert_answer=match["answer"],
            user_query=user_query,
        )
        cheap_messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer accurately using the provided expert reference."},
            {"role": "user", "content": context_prompt},
        ]
        answer = await call_deepseek(cheap_messages, temperature, max_tokens)
        model_used = "deepseek-v4-flash (cached)"
    else:
        # --- EXPENSIVE PATH: use Gemini via OpenRouter ---
        answer, model_used = await call_gemini(messages, temperature, max_tokens)
        insert_qa(user_query, answer, model_used)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
```

**Step 2: Verify syntax**

```bash
cd ~/cloud/projects/lowcostllm
source .venv/bin/activate
python3 -c "from proxy import handle_chat_completion; print('proxy.py OK')"
```

**Verification:** No syntax errors.

---

## Task 6: FastAPI app and server launcher

**Objective:** Wire up the FastAPI app with health check and the chat completions endpoint.

**Files:**
- Create: `main.py`
- Create: `server.py`

**Step 1: Write main.py**

```python
"""FastAPI application for LowCostLLM."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from db import init_db
from proxy import handle_chat_completion


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="LowCostLLM", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    from config import DEEPSEEK_MODEL, GEMINI_MODEL
    return {
        "object": "list",
        "data": [
            {"id": DEEPSEEK_MODEL, "object": "model"},
            {"id": GEMINI_MODEL, "object": "model"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        result = await handle_chat_completion(request)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=500,
        )
```

**Step 2: Write server.py**

```python
"""Uvicorn launcher."""
import uvicorn
from config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
```

**Step 3: Start server and verify health**

```bash
cd ~/cloud/projects/lowcostllm
source .venv/bin/activate
python server.py &
sleep 2
curl -s http://127.0.0.1:8800/health
curl -s http://127.0.0.1:8800/v1/models | python3 -m json.tool
```

**Verification:** Health returns `{"status":"ok"}`, models lists both models.

---

## Task 7: Create .env with real keys

**Objective:** Create the actual `.env` file from your stored credentials.

**Files:**
- Create: `.env`

**Step 1: Write .env**

```env
# DeepSeek direct API (cheap model)
DEEPSEEK_API_KEY=sk-or-v1-your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash

# OpenRouter (Gemini — expensive model)
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
GEMINI_MODEL=google/gemini-3.1-flash-lite

# Matching
SIMILARITY_THRESHOLD=85

# Server
HOST=127.0.0.1
PORT=8800
```

> **You fill in the actual keys.** DeepSeek key is your `sk-or-...d1a6`. OpenRouter key is in `~/cloud/.env` or `~/api/.env`.

---

## Task 8: End-to-end integration test

**Objective:** Send a real chat completion, verify the response format, check that the cache is populated.

**Files:**
- Test (no new files): run curl commands

**Step 1: First query (should hit Gemini)**

```bash
curl -s http://127.0.0.1:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lowcostllm",
    "messages": [{"role": "user", "content": "What are the key steps to commission a VFD?"}],
    "temperature": 0.7
  }' | python3 -m json.tool
```

Expected: response with `"model"` containing "gemini", answer about VFD commissioning.

**Step 2: Verify cache entry**

```bash
cd ~/cloud/projects/lowcostllm
source .venv/bin/activate
python3 -c "
from db import get_all_queries
rows = get_all_queries()
print(f'Cache entries: {len(rows)}')
for r in rows:
    print(f'  id={r[\"id\"]} query={r[\"query\"][:60]}... model={r[\"model_used\"]}')
"
```

Expected: 1 entry with the VFD query.

**Step 3: Similar query (should hit DeepSeek Flash)**

```bash
curl -s http://127.0.0.1:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lowcostllm",
    "messages": [{"role": "user", "content": "commissioning steps for VFD"}],
    "temperature": 0.7
  }' | python3 -m json.tool
```

Expected: response with `"model"` containing "deepseek" or "cached", still good VFD answer.

**Step 4: Verify cache count unchanged**

```bash
cd ~/cloud/projects/lowcostllm
source .venv/bin/activate
python3 -c "
from db import get_all_queries
rows = get_all_queries()
print(f'Cache entries: {len(rows)} (should still be 1)')
"
```

**Verification:** Cache count stays at 1 — similar queries reuse, don't create duplicates.

---

## Task 9: README

**Objective:** Write usage documentation.

**Files:**
- Create: `README.md`

**Step 1: Write README.md**

```markdown
# LowCostLLM

OpenAI-compatible API proxy that reduces LLM costs by caching expensive model answers and reusing them for similar queries with a cheaper model.

## How it works

1. Query arrives at `/v1/chat/completions`
2. Checked against SQLite cache using RapidFuzz `token_sort_ratio`
3. **>=85% similar**: DeepSeek V4 Flash answers using cached Gemini response as context
4. **New topic**: Gemini 3.1 Flash generates fresh answer, saved to cache
5. Response returned in OpenAI format

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Run

```bash
python server.py
# Server at http://127.0.0.1:8800
```

## Usage

```bash
curl http://127.0.0.1:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lowcostllm",
    "messages": [{"role": "user", "content": "Your question here"}]
  }'
```

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (OpenAI format) |

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | DeepSeek direct API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `SIMILARITY_THRESHOLD` | 85 | RapidFuzz score to trigger cheap model |
| `PORT` | 8800 | Server port |
```

---

## Next steps (future enhancements)

After the prototype works end-to-end:

1. **Embedding-based semantic search** — add `sentence-transformers` for catching "pump broken" vs "pump not working" that RapidFuzz misses
2. **Response streaming** — add SSE streaming for both paths
3. **Cache TTL** — auto-expire entries older than N days
4. **Usage stats endpoint** — `/v1/stats` showing cache hit rate, cost saved
5. **Multi-turn conversation support** — extract topic from full message history, not just last user message
