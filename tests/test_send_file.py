"""Tests for the send_file tool — filename safety, Telegram multipart delivery, web upload.

Run:  .venv/bin/python tests/test_send_file.py
"""
import os
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CHEAP_API_KEY", "test-key")
os.environ.setdefault("EXPENSIVE_API_KEY", "test-key")

from llm import (
    _sanitize_filename,
    _send_telegram_document,
    _upload_file_for_url,
    _send_file_impl,
    set_delivery_context,
    clear_delivery_context,
)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ── filename sanitization ────────────────────────────────────────

def test_sanitize_filename() -> None:
    print("sanitize_filename:")
    check("keeps .txt", _sanitize_filename("notes.txt") == "notes.txt")
    check("keeps .MD case", _sanitize_filename("REPORT.MD") == "REPORT.MD")
    check("keeps .csv", _sanitize_filename("data.csv") == "data.csv")
    check("strips dirs", _sanitize_filename("../../etc/passwd") == "passwd.md")
    check("strips backslash dirs", _sanitize_filename("a\\b\\c.md") == "c.md")
    check("defaults no-ext to .md", _sanitize_filename("noext") == "noext.md")
    check("unknown ext defaults to .md", _sanitize_filename("weird.exe") == "weird.md")
    check("empty defaults", _sanitize_filename("  ") == "file.md")


# ── telegram multipart send ──────────────────────────────────────

class _FakeResp:
    status_code = 200

    def __init__(self, body: dict) -> None:
        self._body = body

    def json(self) -> dict:
        return self._body


def test_telegram_send_ok() -> None:
    print("telegram sendDocument:")
    with mock.patch("httpx.Client") as mc:
        mc.return_value.__enter__.return_value.post.return_value = _FakeResp({"ok": True})
        ok, detail = _send_telegram_document(123, "TOKEN", "test.md", "# hi")
        check("returns ok", ok and not detail)
        call = mc.return_value.__enter__.return_value.post.call_args
        check("hits sendDocument", call[0][0].endswith("/botTOKEN/sendDocument"))
        check("chat_id str", call.kwargs["data"]["chat_id"] == "123")
        fname, fdata, _ctype = call.kwargs["files"]["document"]
        check("filename passed", fname == "test.md")
        check("utf-8 bytes passed", fdata == "# hi".encode("utf-8"))


def test_telegram_send_fail() -> None:
    with mock.patch("httpx.Client") as mc:
        mc.return_value.__enter__.return_value.post.return_value = _FakeResp({"ok": False, "description": "nope"})
        ok, detail = _send_telegram_document(123, "TOKEN", "test.md", "x")
        check("reports failure", not ok and "nope" in detail)


def test_telegram_send_exception() -> None:
    with mock.patch("httpx.Client", side_effect=RuntimeError("boom")):
        ok, detail = _send_telegram_document(123, "TOKEN", "test.md", "x")
        check("reports exception", not ok and "boom" in detail)


# ── web upload ───────────────────────────────────────────────────

def test_web_upload_ok() -> None:
    print("web upload:")
    with mock.patch("subprocess.run") as mr:
        mr.return_value.stdout = '{"download_url": "https://api.smartdochub.net/uploads/x.md", "status": "success"}'
        url = _upload_file_for_url("x.md", "content")
        check("returns url", url == "https://api.smartdochub.net/uploads/x.md")


def test_web_upload_fail() -> None:
    with mock.patch("subprocess.run") as mr:
        mr.return_value.stdout = '{"status": "error"}'
        check("returns empty on error", _upload_file_for_url("x.md", "c") == "")


# ── tool routing by delivery context ─────────────────────────────

def test_tool_telegram_ctx() -> None:
    print("send_file routing:")
    with mock.patch("llm._send_telegram_document", return_value=(True, "")) as m:
        set_delivery_context("telegram", 123, "TOKEN")
        try:
            out = _send_file_impl("hello.txt", "Hello world")
        finally:
            clear_delivery_context()
        check("telegram ctx sends", "sent to the chat" in out)
        m.assert_called_once_with(123, "TOKEN", "hello.txt", "Hello world")


def test_tool_telegram_ctx_failure() -> None:
    with mock.patch("llm._send_telegram_document", return_value=(False, "err")):
        set_delivery_context("telegram", 123, "TOKEN")
        try:
            out = _send_file_impl("hello.txt", "Hello world")
        finally:
            clear_delivery_context()
        check("telegram failure reported", "delivery failed" in out)


def test_tool_web_ctx() -> None:
    with mock.patch("llm._upload_file_for_url", return_value="https://x.md") as m:
        set_delivery_context("web")
        try:
            out = _send_file_impl("hello.md", "Hello")
        finally:
            clear_delivery_context()
        check("web ctx returns url", "https://x.md" in out)
        m.assert_called_once_with("hello.md", "Hello")


def test_tool_no_ctx_saves_locally() -> None:
    out = _send_file_impl("tmp_test.md", "content")
    check("no ctx saves locally", "saved to" in out)


def test_tool_size_cap() -> None:
    big = "x" * 400_000
    out = _send_file_impl("big.txt", big)
    check("oversize rejected", "too large" in out)


def test_tool_sanitizes_name() -> None:
    with mock.patch("llm._send_telegram_document", return_value=(True, "")) as m:
        set_delivery_context("telegram", 123, "TOKEN")
        try:
            _send_file_impl("../../evil", "x")
        finally:
            clear_delivery_context()
        check("sanitizes filename in call", m.call_args[0][2] == "evil.md")


if __name__ == "__main__":
    test_sanitize_filename()
    test_telegram_send_ok()
    test_telegram_send_fail()
    test_telegram_send_exception()
    test_web_upload_ok()
    test_web_upload_fail()
    test_tool_telegram_ctx()
    test_tool_telegram_ctx_failure()
    test_tool_web_ctx()
    test_tool_no_ctx_saves_locally()
    test_tool_size_cap()
    test_tool_sanitizes_name()
    print(f"\n{'-' * 40}\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
