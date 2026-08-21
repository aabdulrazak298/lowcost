"""Unit test: _wrap_error_detection must raise on finish_reason='error'
instead of returning partial content (the 2026-08-20 mid-sentence cut)."""
import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import _wrap_error_detection


class FakeCompletions:
    def __init__(self, finish_reason, content):
        self._fr = finish_reason
        self._content = content

    async def create(self, *args, **kwargs):
        msg = SimpleNamespace(content=self._content, tool_calls=None)
        choice = SimpleNamespace(finish_reason=self._fr, message=msg)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, finish_reason, content):
        self.chat = SimpleNamespace(completions=FakeCompletions(finish_reason, content))


async def main():
    # 1) finish_reason='error' with partial content -> MUST raise
    ok_client = FakeClient("stop", "Complete answer.")
    wrapped_ok = _wrap_error_detection(ok_client)
    r = await wrapped_ok.chat.completions.create(messages=[])
    assert r.choices[0].finish_reason == "stop", "normal response must pass through"

    err_client = FakeClient("error", "China censors public mourning as it holds former")
    wrapped_err = _wrap_error_detection(err_client)
    try:
        await wrapped_err.chat.completions.create(messages=[])
        print("FAIL: finish_reason=error did NOT raise")
        sys.exit(1)
    except RuntimeError as e:
        print("PASS: finish_reason=error raised RuntimeError")
        print("  msg:", str(e)[:110])

    # 2) streaming requests must NOT be intercepted
    class FakeStreamCompletions:
        async def create(self, *args, **kwargs):
            assert kwargs.get("stream") is True
            return "stream-obj"

    class FakeStreamClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeStreamCompletions())

    sc = _wrap_error_detection(FakeStreamClient())
    out = await sc.chat.completions.create(messages=[], stream=True)
    print("PASS: stream=True passed through untouched ->", out)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
