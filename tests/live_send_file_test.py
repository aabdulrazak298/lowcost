"""Live test: deliver a real document to Azuan's Telegram via the send_file tool path."""
import asyncio
import sys

sys.path.insert(0, ".")

from llm import _send_file_impl, set_delivery_context, clear_delivery_context
from config import TELEGRAM_BOT_TOKEN

CHAT = 915519325


async def main() -> None:
    set_delivery_context("telegram", CHAT, TELEGRAM_BOT_TOKEN)
    try:
        out = _send_file_impl(
            "send_file_test.md",
            "# Send-file test\n\n"
            "This file was delivered by the new send_file tool.\n\n"
            "- .md support\n"
            "- .txt support\n"
            "- delivered directly via sendDocument\n",
        )
        print("TOOL:", out)
    finally:
        clear_delivery_context()


if __name__ == "__main__":
    asyncio.run(main())
