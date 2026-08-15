"""Text-to-speech for Telegram voice replies.

Two engines:
  - "edge"   — Microsoft edge-tts (free, no API key). Default.
  - "kokoro" — hexgrad/kokoro-82m via OpenRouter (CHEAP_API_KEY).

Both produce an MP3 which is then transcoded to Opus-in-OGG (Telegram voice
message format) via ffmpeg. The ffmpeg step runs in a worker thread so it
does not block the bot's event loop.
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import edge_tts
import httpx

from config import (
    CHEAP_API_KEY,
    CHEAP_BASE_URL,
    KOKORO_VOICE,
    VOICE_NAME,
    VOICE_RATE,
)

logger = logging.getLogger("lowcostllm.tts")

KOKORO_MODEL = "hexgrad/kokoro-82m"
_REFERER = "https://llm.smartdochub.net"
_TITLE = "LowCostLLM"

# ── Markdown / formatting cleaning ─────────────────────────────────

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # symbols & pictographs
    "\U0001FB00-\U0001FFFF"
    "\U00002600-\U000027BF"   # misc symbols, dingbats
    "\U0001F300-\U0001F64F"   # emoticons
    "\U00002700-\U000027BF"
    "\U0001F680-\U0001F6FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0001F1E6-\U0001F1FF"   # flags
    "]+"
)


def _strip_markdown(text: str) -> str:
    """Remove markdown syntax so text reads naturally. Keeps URLs + emoji."""
    t = text
    t = re.sub(r"```[\s\S]*?```", " ", t)                 # fenced code
    t = re.sub(r"`([^`]*)`", r"\1", t)                    # inline code
    t = t.replace("|", " ")                                # table pipes
    t = re.sub(r"^\s*[-: ]+\s*$", " ", t, flags=re.MULTILINE)  # table sep rows
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.MULTILINE)  # headings
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)              # bold
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)                  # italic
    t = re.sub(r"_([^_]+)_", r"\1", t)
    t = re.sub(r"~~([^~]+)~~", r"\1", t)                  # strikethrough
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)  # bullets
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.MULTILINE)  # numbered
    t = re.sub(r"^\s*>\s?", "", t, flags=re.MULTILINE)    # blockquote
    t = re.sub(r"^\s*[-—_]{3,}\s*$", " ", t, flags=re.MULTILINE)  # hr
    return t


def clean_for_speech(text: str) -> str:
    """Aggressive clean for audio: strip markdown + URLs + emoji."""
    t = _strip_markdown(text)
    t = _URL_RE.sub("link", t)
    t = _EMOJI_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def clean_for_display(text: str) -> str:
    """Light clean for the spoiler transcript: strip markdown, keep URLs + emoji."""
    t = _strip_markdown(text)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ── Engine backends ────────────────────────────────────────────────


async def _edge_to_mp3(text: str, path: str) -> None:
    comm = edge_tts.Communicate(text, VOICE_NAME, rate=VOICE_RATE)
    await comm.save(path)


async def _kokoro_to_mp3(text: str, path: str) -> None:
    url = CHEAP_BASE_URL.rstrip("/") + "/audio/speech"
    headers = {
        "Authorization": f"Bearer {CHEAP_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": _REFERER,
        "X-Title": _TITLE,
    }
    payload = {
        "model": KOKORO_MODEL,
        "input": text,
        "voice": KOKORO_VOICE,
        "response_format": "mp3",  # OpenRouter only accepts mp3/pcm
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)


def _mp3_to_ogg_opus(mp3_path: str, ogg_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", mp3_path,
            "-c:a", "libopus", "-b:a", "48k", "-ar", "48000", "-ac", "1",
            ogg_path,
        ],
        check=True,
        capture_output=True,
    )


# ── Public API ─────────────────────────────────────────────────────


async def text_to_voice_ogg(text: str, engine: str | None = None) -> bytes:
    """Convert text → Opus/OGG voice bytes. `engine`: 'edge' or 'kokoro'."""
    from config import get_voice_engine

    engine = engine or get_voice_engine()
    clean = clean_for_speech(text)
    if not clean:
        clean = "No response."

    with tempfile.TemporaryDirectory() as td:
        mp3 = str(Path(td) / "speech.mp3")
        ogg = str(Path(td) / "speech.ogg")

        if engine == "kokoro":
            await _kokoro_to_mp3(clean, mp3)
        else:
            await _edge_to_mp3(clean, mp3)

        await asyncio.to_thread(_mp3_to_ogg_opus, mp3, ogg)
        return Path(ogg).read_bytes()
