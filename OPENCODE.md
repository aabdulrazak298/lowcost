# Using LowCostLLM with OpenCode

## 1. Start the server

```bash
cd /home/one/lowcode
source .venv/bin/activate
python main.py
```

Runs on `http://127.0.0.1:8800`. Keep this terminal open.

## 2. Configure OpenCode

Add this to `opencode.json` in your project root (or `~/.config/opencode/opencode.jsonc` for global):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "lowcostllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LowCostLLM Proxy",
      "options": {
        "baseURL": "http://127.0.0.1:8800/v1"
      },
      "models": {
        "lowcostllm": {
          "name": "LowCostLLM (DeepSeek V4 Pro + Flash cache)"
        }
      }
    }
  }
}
```

## 3. Connect in OpenCode

```
/connect
```

Select **LowCostLLM Proxy**, enter any string as the API key (the proxy ignores it — auth is already in `.env`).

## 4. Select the model

```
/models
```

Pick **LowCostLLM (DeepSeek V4 Pro + Flash cache)**.

## How it works

```
Your query → OpenCode
                 ↓
OpenCode sends request → LowCostLLM Proxy (port 8800)
                 ↓
       Fuzzy cache check (RapidFuzz, threshold 48)
                 ↓
   ┌─ Hit ───────────────────────────────────┐
   │ DeepSeek Chat adapts cached answer       │
   │ IRRELEVANT guard → escalates on mismatch │
   └──────────────────────────────────────────┘
                 ↓
   ┌─ Miss ──────────────────────────────────┐
   │ DeepSeek V4 Pro answers with tools       │
   │ Answer cached for future reuse           │
   └──────────────────────────────────────────┘
                 ↓
       Response → OpenCode (streaming SSE)
```

Cached answers are served by the cheap model (DeepSeek Chat), new answers come from the expensive model (DeepSeek V4 Pro). This saves ~99% on API costs for repeated/similar queries.

## Verify it's working

```bash
# Non-streaming
curl -s http://127.0.0.1:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50}'

# Streaming
curl -s -N http://127.0.0.1:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50,"stream":true}'

# Admin dashboard
curl -s http://127.0.0.1:8800/admin | python3 -m json.tool
```
