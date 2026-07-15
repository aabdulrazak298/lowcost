"""Conversation memory — sliding window with auto-summarization.

Mirrors the FlaskChat chat_history format so both interfaces share the same
context structure. When a user exceeds MAX_MESSAGES, the oldest portion is
summarised by the cheap model and the raw messages are pruned.
"""
import logging
from config import CHEAP_MODEL
from db import (
    save_message,
    get_history,
    get_message_count,
    delete_messages,
    build_history_string,
)
from llm import call_cheap

logger = logging.getLogger(__name__)

MAX_MESSAGES = 20       # trigger summarisation when exceeded
SUMMARISE_COUNT = 10    # number of oldest messages to compress at once

SUMMARISE_PROMPT = """Summarise this conversation excerpt in 1-2 sentences.
Focus on key facts, decisions, and context that would help answer follow-up questions.
Write in third person past tense.

Conversation:
{conversation}

Summary:"""


async def _summarise_old_messages(user_id: int) -> None:
    """Summarise the oldest SUMMARISE_COUNT messages and replace them with a summary."""
    messages = get_history(user_id, limit=SUMMARISE_COUNT)
    if len(messages) < 5:
        return  # not enough to bother summarising

    # Build the conversation text for summarisation
    conv_text = build_history_string(messages)

    # Use cheap model to summarise
    try:
        summary = await call_cheap([
            {
                "role": "system",
                "content": "You are a concise summariser. Output only the summary, no preamble.",
            },
            {
                "role": "user",
                "content": SUMMARISE_PROMPT.format(conversation=conv_text),
            },
        ])
        summary = summary.strip()
    except Exception as e:
        logger.warning(f"Summarisation failed for user {user_id}: {e}")
        return

    if not summary or len(summary) < 10:
        return

    # Delete the old messages
    old_ids = [m["id"] for m in messages]
    # Use the oldest message's timestamp so the summary sorts before remaining messages
    oldest_ts = messages[0]["created_at"]
    delete_messages(user_id, old_ids)

    # Insert the summary as an assistant message with the oldest timestamp
    save_message(user_id, "assistant", f"[Earlier: {summary}]", created_at=oldest_ts)

    logger.info(
        f"User {user_id}: summarised {len(messages)} messages → "
        f"{len(summary)} chars"
    )


async def load_and_prepare_history(user_id: int) -> str:
    """Load conversation history for a user, auto-summarising if needed.

    Returns a chat_history string in FlaskChat format (User:/Assistant: lines),
    or empty string for new conversations.
    """
    count = get_message_count(user_id)

    # Auto-summarise if over the limit
    if count > MAX_MESSAGES:
        await _summarise_old_messages(user_id)

    # Load remaining history
    messages = get_history(user_id, limit=MAX_MESSAGES)
    if not messages:
        return ""

    return build_history_string(messages)


async def save_exchange(user_id: int, user_query: str, assistant_answer: str) -> None:
    """Save a complete exchange to the user's history."""
    save_message(user_id, "user", user_query)
    save_message(user_id, "assistant", assistant_answer)
