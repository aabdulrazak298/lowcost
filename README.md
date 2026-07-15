# LowCostLLM

Two-tier caching LLM proxy that saves 99%+ on API costs by reusing expensive model answers with a cheap adaptive model.

## How it works

```
User query
    ↓
Fuzzy cache check (RapidFuzz, threshold 48)
    ↓
┌─ Cache hit ─────────────────────────────────────┐
│ DeepSeek V4 Flash adapts cached answer          │
│ Self-checks for relevance (IRRELEVANT guard)    │
│ If off-topic → escalates to expensive path       │
└─────────────────────────────────────────────────┘
    ↓
┌─ Cache miss / IRRELEVANT ───────────────────────┐
│ DeepSeek V4 Pro answers (50 tool-call rounds)    │
│ Answer cached for future reuse                   │
└─────────────────────────────────────────────────┘
    ↓
Stream response back (SSE) + send images
```

## Features

- **Fuzzy caching** — weighted RapidFuzz matching (token_set × 0.6 + partial × 0.4)
- **IRRELEVANT detection** — cheap model rejects unrelated cached answers, escalates to expensive
- **Streaming** — SSE token-by-token, live message editing on Telegram
- **Tool calling** — web search, code execution, image generation, YouTube transcripts, plots
- **Async image generation** — fire-and-forget, Telegram sends when ready
- **Persistent stats** — SQLite with hit rates, model usage, cache analytics
- **Error recovery** — 3 retry levels (loop detection → disable tools → temperature bump)
- **Dual frontends** — Telegram bot + REST API (`POST /v1/chat/completions`)

## Stack

| Component | Tech |
|---|---|
| Server | FastAPI + Uvicorn (systemd, port 8800) |
| Cheap model | DeepSeek V4 Flash (OpenRouter) |
| Expensive model | DeepSeek V4 Pro (native API) |
| Caching | SQLite (cache.db) |
| Fuzzy matching | RapidFuzz |
| Telegram | python-telegram-bot (webhook mode) |
| Tools | SearXNG, Python sandbox, OpenRouter image gen |

## Setup

```bash
# 1. Clone and venv
git clone <repo-url>
cd lowcostllm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Create .env
cp .env.example .env
# Fill in: CHEAP_API_KEY, EXPENSIVE_API_KEY, TELEGRAM_BOT_TOKEN

# 3. Run
python main.py
```

## Environment variables

| Variable | Description |
|---|---|
| `CHEAP_MODEL` | Cheap model ID (default: `deepseek/deepseek-v4-flash`) |
| `CHEAP_API_KEY` | OpenRouter API key |
| `CHEAP_BASE_URL` | API base URL for cheap model |
| `EXPENSIVE_MODEL` | Expensive model ID (default: `deepseek-v4-pro`) |
| `EXPENSIVE_API_KEY` | DeepSeek API key |
| `EXPENSIVE_BASE_URL` | API base URL for expensive model |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (optional) |
| `TELEGRAM_ALLOWED_USERS` | Comma-separated user IDs (optional) |
| `SIMILARITY_THRESHOLD` | Fuzzy match threshold (default: 48) |
| `CACHE_TTL_DAYS` | Cache expiry in days (default: 365) |

## API

### Chat completion

```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "Explain PID control"}],
  "max_tokens": 4096
}
```

### Admin

```bash
GET /admin        # Cache stats + usage metrics
GET /             # Health check
```

## Telegram commands

| Command | Description |
|---|---|
| `/stats` | Usage statistics |
| `/new` | Clear conversation history |
| `/help` | Help text |

## Architecture diagram

![Architecture](generated/architecture.png)

## License

MIT
