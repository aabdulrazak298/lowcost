"""Shared query processor — cache-check → route → respond.

Both the Flask Chat webhook and Telegram bot call this same function.
"""
import datetime as _dt
import asyncio
import json
import logging
import re as _re
import urllib.request
from config import get_cheap_model, AGENTIC_CACHE
from db import cache_lookup, insert_qa, increment_hit_count
from llm import call_cheap, call_expensive, call_expensive_stream, _clear_generated_images, _get_generated_images, _wait_for_images, get_last_usage
from stats import record_request, record_irrelevant_escalation
from curator import run_curator

logger = logging.getLogger(__name__)

_TODAY = _dt.datetime.now().strftime("%A, %d %B %Y")
_DATE_CONTEXT = f"Today's date is {_TODAY}. Use this for any time-sensitive context."


def history_to_turns(chat_history) -> list[dict]:
    """Normalize history to [{role, content}] turns.

    Accepts either a list of {role, content} dicts (Telegram) or a legacy
    "User: ...\\nAssistant: ..." string (FlaskChat webhook). Returns a list of
    turns with roles constrained to 'user'/'assistant'.
    """
    if not chat_history:
        return []
    if isinstance(chat_history, list):
        out = []
        for h in chat_history:
            if not isinstance(h, dict):
                continue
            role = h.get("role")
            content = h.get("content")
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": str(content)})
        return out
    # legacy string
    turns = []
    for line in str(chat_history).split("\n"):
        if line.startswith("User: "):
            turns.append({"role": "user", "content": line[6:]})
        elif line.startswith("Assistant: "):
            turns.append({"role": "assistant", "content": line[11:]})
    return turns


def turns_to_string(turns: list[dict]) -> str:
    """Inverse: turns -> "User: ...\\nAssistant: ..." (for the dormant agentic path)."""
    return "\n".join(
        f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}"
        for t in turns
    )


_SEARCH_QUERY_PROMPT = (
    "You generate a search key for a semantic cache lookup. Rewrite the user's latest "
    "message into ONE self-contained search query. Resolve pronouns and implicit references "
    "(\"the first one\", \"that video\", \"he said\", \"number 5\") using the conversation "
    "history. If resolved references are provided, describe the topic using their titles "
    "(NOT the raw URL). Include concrete topic words — names, titles, subjects, entities — "
    "so the query is unambiguous on its own. Output ONLY the rewritten query. No quotes, "
    "no markdown, no explanation, no preamble."
)

_URL_RE = _re.compile(r"https?://[^\s<>\"']+")
_YT_API = "http://141.11.17.227:8000/api/youtube/script"
_YT_KEY = "987654321"


def _extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text or "")


def _is_youtube(url: str) -> bool:
    return ("youtube.com" in url) or ("youtu.be" in url)


_VIDEO_ID_RE = _re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def _extract_video_ids(text: str) -> list[str]:
    """Extract unique YouTube video IDs (11-char) from a query."""
    out: list[str] = []
    for vid in _VIDEO_ID_RE.findall(text or ""):
        if vid not in out:
            out.append(vid)
    return out


def _lookup_exact_video(user_query: str) -> dict | None:
    """Exact video-ID cache hit — same video re-ask reuses its summary.

    Runs BEFORE the agentic search: it's free + deterministic, and it still
    works if the transcript VPS is down (we already hold the cached summary).
    """
    from db import lookup_by_video_id

    for vid in _extract_video_ids(user_query):
        m = lookup_by_video_id(vid)
        if m:
            return m
    return None


def _resolve_youtube_title(video_url: str) -> str | None:
    """Fetch video title/channel/duration from the transcript VPS (metadata only).

    Returns None when the video can't be resolved (no transcript / network error).
    """
    try:
        req = urllib.request.Request(
            _YT_API,
            data=json.dumps({"video_url_or_id": video_url}).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": _YT_KEY},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        if not data.get("transcript_available"):
            return None
        meta = data.get("metadata", {}) or {}
        title = str(meta.get("title", "Unknown")).strip()
        channel = str(meta.get("channel") or meta.get("author") or "").strip()
        dur = int(meta.get("duration", 0) or 0) // 60
        out = title
        if channel:
            out += f" — channel: {channel}"
        return f"{out} ({dur} min)"
    except Exception:
        return None


def _resolve_web_title(url: str) -> str | None:
    """Fetch a web page's <title> (fallback: first text) for a search key.

    Returns None when the page can't be resolved (blocked / unreachable).
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 LowCostLLM/0.5"})
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
        m = _re.search(r"<title[^>]*>(.*?)</title>", html, flags=_re.DOTALL | _re.IGNORECASE)
        if m:
            title = _re.sub(r"\s+", " ", m.group(1)).strip()
            if title:
                return title[:300]
        text = _re.sub(r"<[^>]+>", " ", html)
        cleaned = _re.sub(r"\s+", " ", text).strip()
        return cleaned[:300] or None
    except Exception:
        return None


def _resolve_references(user_query: str) -> tuple[list[str], bool]:
    """Resolve every URL in the query to its title/topic.

    Returns (lines, ok). ok is False when any URL couldn't be resolved — the
    caller must then skip the cache and let the final LLM report the failure.
    """
    lines: list[str] = []
    ok = True
    seen: set[str] = set()
    for url in _extract_urls(user_query):
        u = url.rstrip(".,;:!?)")
        if u in seen:
            continue
        seen.add(u)
        topic = _resolve_youtube_title(u) if _is_youtube(u) else _resolve_web_title(u)
        if topic is None:
            ok = False
            continue
        lines.append(f"{u} → {topic}")
    return lines, ok


async def generate_search_query(user_query: str, turns: list[dict]) -> str:
    """Generate a canonical cache-search key from raw query + history + resolved links.

    Used ONLY to drive `cache_lookup` — it never reaches the answer path.

    Returns "" (empty) when a link in the query can't be resolved: there's no
    meaningful cache key to build, so the caller skips the cache and the final
    LLM fetches the link itself and tells the user what happened.
    """
    try:
        refs, ok = await asyncio.to_thread(_resolve_references, user_query)
        if not ok:
            return ""

        history = turns_to_string(turns[-12:])
        parts = []
        if history:
            parts.append(f"Conversation history:\n{history}")
        if refs:
            parts.append("Resolved references (what each link is about):\n" + "\n".join(refs))
        parts.append(f"Latest message:\n{user_query}")

        messages = [
            {"role": "system", "content": _SEARCH_QUERY_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

        key = await call_cheap(messages, tools=[])
        key = (key or "").strip().strip('"\'`').strip()
        if not key or key.startswith("(error") or key.startswith("(no response"):
            return user_query
        return key
    except Exception:
        logger.warning("Search-query generation failed — falling back to raw query")
        return user_query


_REFERENTIAL_RE = _re.compile(
    r"(?i)\bnumber\s*\d+\b"                       # "number 5"
    r"|\bthe\s+(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"
    r"|\b(?:first|second|third|fourth|fifth)\s+(?:one|item|option|point|bullet)\b"
    r"|\b(?:the|that|this)\s+(?:one|item|option|point|part)\b"
    r"|\bwhat\s+about\s+(?:it|that|them|those|this)\s*$"
    r"|\b(?:it|that|this|them|those)\s*$"
)


def _is_referential(query: str) -> bool:
    """True if query is a short deictic follow-up whose meaning depends on the
    prior turn — must NOT be cache-matched blind."""
    q = (query or "").strip()
    if not q:
        return False
    if len(q.split()) > 10:          # only short follow-ups are referential
        return False
    return bool(_REFERENTIAL_RE.search(q))


_REJECTION_PHRASES = (
    # Metadata leak — model cites the cached "expert" material instead of answering.
    "expert answer", "expert material", "expert says", "expert reference",
    "expert ai", "cached answer", "cached expert", "cached response",
    "source material", "provided information", "provided reference",
    "provided by an expert",
    # Topic mismatch — model noticed the cached answer is about a different subject.
    "unrelated", "different topic", "completely different", "not related",
    "off-topic", "off topic", "wrong topic", "mismatch", "nothing to do with",
    "no relation", "different subject", "another topic", "not the same topic",
    "fresh answer", "from scratch",
)


def _is_rejection(answer: str) -> bool:
    """Return True if the cheap model rejected the cached answer.

    Signals (scoped to the first 300 chars to avoid false positives in a
    legitimate answer body):
      1. The literal IRRELEVANT keyword.
      2. Metadata-leak phrases ("expert answer", "reference", ...).
      3. Topic-mismatch phrases ("unrelated", "different topic", ...).

    This errs on the side of escalating: a false positive only costs one
    expensive call, while a false negative serves unrelated content as a
    "cached" hit.
    """
    if not answer:
        return True
    clean = _re.sub(r"</?think\w*>", "", answer, flags=_re.IGNORECASE).strip()
    head = clean[:300].lower()
    if "irrelevant" in head:
        return True
    return any(p in head for p in _REJECTION_PHRASES)


def _is_escalate(answer: str) -> bool:
    """True if the cheap model asked to escalate (no usable cache match)."""
    if not answer:
        return True
    return answer.strip().upper().startswith("ESCALATE")


AGENTIC_CACHE_PROMPT = """You answer the user using a cache of previous expert answers as your knowledge base.

You have a `search_cache` tool. Follow these steps exactly:
1. Rewrite the user's question into a concrete, specific question. Resolve any vague or implicit wording (like "the first one" or "that thing") using the conversation context.
2. Call search_cache with that question.
3. If the tool returns a RELATED cached question+answer, use it as your FOUNDATION — not a constraint — and expand or adapt it to fully answer the user. Augment with anything you know. Do NOT mention the cache, "expert", or say "based on".
4. If the tool returns NO MATCH or an answer about a DIFFERENT topic, do NOT answer the question — reply with exactly one word and nothing else:

ESCALATE"""


async def _agentic_cache_flow(user_query: str, chat_history: str) -> tuple[bool, str]:
    """Cheap model orchestrates cache retrieval via the search_cache tool.

    Returns (is_escalate, answer). When is_escalate is True, answer is ''.
    """
    from llm import search_cache

    messages = [
        {"role": "system", "content": f"{AGENTIC_CACHE_PROMPT}\n\n{_DATE_CONTEXT}"},
    ]
    if chat_history:
        messages.append({
            "role": "system",
            "content": f"Previous conversation:\n{chat_history[-2000:]}",
        })
    messages.append({"role": "user", "content": user_query})
    answer = await call_cheap(messages, tools=[search_cache])
    if _is_escalate(answer):
        return True, ""
    return False, answer


async def process_query(
    user_query: str,
    chat_history: str = "",
    system_prompt: str | None = None,
) -> tuple[str, str, list[str], dict]:
    """Process a user query through the two-tier cache → cheap → expensive pipeline.

    Args:
        user_query: The user's current message.
        chat_history: Previous conversation text (for multi-turn context).
        system_prompt: Optional override system prompt for expensive path.

    Returns:
        (answer_text, model_used_label, generated_image_paths, usage_dict)
    """
    # Clear any images from a previous request
    _clear_generated_images()

    turns = history_to_turns(chat_history)

    rejected_match = None

    if AGENTIC_CACHE:
        # Dormant: cheap model orchestrates cache retrieval via search_cache tool.
        is_escalate, answer = await _agentic_cache_flow(user_query, turns_to_string(turns))
        usage = get_last_usage()
        if not is_escalate:
            model_used = f"{get_cheap_model()} (agentic-cached)"
            record_request(hit=True, model=model_used)
            return answer, model_used, _get_generated_images(), usage
        match = None
    else:
        # Referential follow-up ("number 5")? Its meaning lives in the thread,
        # not in any cached Q&A — skip the cache and let the expensive model
        # answer from full history.
        if _is_referential(user_query) and turns:
            record_request(hit=False, model="referential-bypass")
            match = None
        else:
            match = _lookup_exact_video(user_query)
            if match is None:
                match_query = await generate_search_query(user_query, turns)
                match = await cache_lookup(match_query) if match_query else None

    if match:
        # --- CHEAP PATH: history replayed as turns, cached answer as EXAMPLE ---
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant. {_DATE_CONTEXT}\n"
                    "Use web_search directly to get fresh, current information whenever "
                    "the example is stale, time-sensitive, or incomplete — search and "
                    "answer yourself; never ask the user for permission to search.\n"
                    "Use run_code to execute Python for ANY arithmetic, calculation, "
                    "enumeration, or multi-step computation — never compute in your head.\n"
                    "If the user's message contains an [Example] block about a DIFFERENT "
                    "topic than their actual question, ignore it and answer from the "
                    "conversation instead. If you still cannot answer, reply with exactly "
                    "one word: IRRELEVANT"
                ),
            },
        ]
        messages.extend(turns[-12:])
        messages.append({"role": "user", "content": (
            f"[Example — a similar question and how it was solved. Follow this "
            f"pattern and adapt it to the new question.]\n"
            f"{match['answer']}\n\n"
            f"{user_query}"
        )})
        answer = await call_cheap(messages, tools=None, reasoning=True)
        usage = get_last_usage()

        # Self-check: did the cheap model reject the cached answer?
        if _is_rejection(answer):
            rejected_match = match
            record_request(hit=False, model="irrelevant-escalated")
            logger.info(
                "irrelevant-escalated: query=%r cached=%r rejection=%r",
                user_query[:120], match["query"][:120], answer[:120],
            )
            record_irrelevant_escalation(user_query, match["query"], answer)
        else:
            increment_hit_count(match["id"])
            model_used = f"{get_cheap_model()} (cached)"
            record_request(hit=True, model=model_used)
            return answer, model_used, _get_generated_images(), usage

    # --- EXPENSIVE PATH: history replayed as turns, no cache blob ---
    messages = [{"role": "system", "content": _DATE_CONTEXT}]
    messages.extend(turns[-16:])
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})
    answer, model_used = await call_expensive(messages)
    usage = get_last_usage()
    insert_qa(user_query, answer, model_used)
    record_request(hit=False, model=model_used)

    # Curator: if we rejected a cached candidate, let the expensive model
    # judge it and evict it if poisoned.
    if rejected_match is not None:
        await run_curator(rejected_match)

    return answer, model_used, _get_generated_images(), usage


async def process_query_stream(
    user_query: str,
    callback,
    chat_history: str = "",
    system_prompt: str | None = None,
) -> tuple[str, list[str]]:
    """Same as process_query but streams expensive-model response via callback.

    callback(chunk: str) is called for each text chunk as it arrives.
    Returns (model_used, generated_image_paths).
    """
    _clear_generated_images()

    turns = history_to_turns(chat_history)

    rejected_match = None

    if AGENTIC_CACHE:
        is_escalate, answer = await _agentic_cache_flow(user_query, turns_to_string(turns))
        if not is_escalate:
            model_used = f"{get_cheap_model()} (agentic-cached)"
            record_request(hit=True, model=model_used)
            if asyncio.iscoroutinefunction(callback):
                await callback(answer)
            else:
                callback(answer)
            return model_used, _get_generated_images()
        match = None
    else:
        if _is_referential(user_query) and turns:
            record_request(hit=False, model="referential-bypass")
            match = None
        else:
            match = _lookup_exact_video(user_query)
            if match is None:
                match_query = await generate_search_query(user_query, turns)
                match = await cache_lookup(match_query) if match_query else None

    if match:
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant. {_DATE_CONTEXT}\n"
                    "Use web_search directly to get fresh, current information whenever "
                    "the example is stale, time-sensitive, or incomplete — search and "
                    "answer yourself; never ask the user for permission to search.\n"
                    "Use run_code to execute Python for ANY arithmetic, calculation, "
                    "enumeration, or multi-step computation — never compute in your head.\n"
                    "If the user's message contains an [Example] block about a DIFFERENT "
                    "topic than their actual question, ignore it and answer from the "
                    "conversation instead. If you still cannot answer, reply with exactly "
                    "one word: IRRELEVANT"
                ),
            },
        ]
        messages.extend(turns[-12:])
        messages.append({"role": "user", "content": (
            f"[Example — a similar question and how it was solved. Follow this "
            f"pattern and adapt it to the new question.]\n"
            f"{match['answer']}\n\n"
            f"{user_query}"
        )})
        answer = await call_cheap(messages, tools=None, reasoning=True)

        if _is_rejection(answer):
            rejected_match = match
            record_request(hit=False, model="irrelevant-escalated")
            logger.info(
                "irrelevant-escalated: query=%r cached=%r rejection=%r",
                user_query[:120], match["query"][:120], answer[:120],
            )
            record_irrelevant_escalation(user_query, match["query"], answer)
        else:
            increment_hit_count(match["id"])
            model_used = f"{get_cheap_model()} (cached)"
            record_request(hit=True, model=model_used)
            if asyncio.iscoroutinefunction(callback):
                await callback(answer)
            else:
                callback(answer)
            return model_used, _get_generated_images()

    # Expensive path with streaming
    messages = [{"role": "system", "content": _DATE_CONTEXT}]
    messages.extend(turns[-16:])
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})

    logger.info(f"Starting streaming for query: {user_query[:80]}")
    answer, model_used = await call_expensive_stream(messages, callback)
    logger.info(f"Streaming complete: {len(answer)} chars, model={model_used}")
    insert_qa(user_query, answer, model_used)
    record_request(hit=False, model=model_used)

    # Curator: if we rejected a cached candidate, let the expensive model
    # judge it and evict it if poisoned.
    if rejected_match is not None:
        await run_curator(rejected_match)

    return model_used, _get_generated_images()
