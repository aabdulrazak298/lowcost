"""Cheap-model fallback tests — retry policy, fallback switching, disable, persist.

No real API calls: _run_cheap_agent and _client_for_model are monkeypatched.
The model_overrides DB row is snapshotted and restored so live settings
(Azuan's /model choices) are never clobbered.

Run: cd ~/cloud/projects/lowcostllm && .venv/bin/python tests/test_cheap_fallback.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import RateLimitError  # noqa: F401  (kept as doc reference — see _RateLimit429)

import config
import db
import llm

PRIMARY = "qwen/qwen3.7-flash"
FALLBACK = "deepseek/deepseek-v4-flash"

MSGS = [{"role": "user", "content": "hi"}]


class _RateLimit429(Exception):
    """Stand-in for a provider 429. The fallback logic catches generic
    Exception, so we don't need openai's APIStatusError constructor
    (which demands response/body objects)."""

# ── Helpers ───────────────────────────────────────────────────────

def _snapshot_overrides() -> dict:
    conn = db.get_conn()
    row = conn.execute(
        "SELECT cheap_override, expensive_override, cheap_fallback_override "
        "FROM model_overrides WHERE id = 1"
    ).fetchone()
    return dict(row) if row else {}


def _restore_overrides(snap: dict) -> None:
    conn = db.get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO model_overrides "
        "(id, cheap_override, expensive_override, cheap_fallback_override) "
        "VALUES (1, ?, ?, ?)",
        (snap.get("cheap_override"), snap.get("expensive_override"),
         snap.get("cheap_fallback_override")),
    )
    conn.commit()
    config._cheap_override = snap.get("cheap_override")
    config._expensive_override = snap.get("expensive_override")
    config._cheap_fallback_override = snap.get("cheap_fallback_override")


def _configure(cheap: str = PRIMARY, fallback: str | None = FALLBACK) -> None:
    config._cheap_override = cheap
    config._cheap_fallback_override = fallback


def _mk_usage(model: str) -> dict:
    return {"prompt_tokens": 1, "completion_tokens": 1, "model": model}


def _fake_agent_runner(calls: list, primary_fail_plan: list[bool], primary_out: str, fallback_out: str):
    """Return an async _run_cheap_agent replacement.

    primary_fail_plan[i] == True → the i-th primary call raises; otherwise it
    returns primary_out. Fallback calls always return fallback_out.
    """
    primary_idx = 0

    async def fake_run(model_id, messages, temperature, max_tokens, tools, reasoning):
        nonlocal primary_idx
        calls.append(model_id)
        if model_id == PRIMARY:
            if primary_idx < len(primary_fail_plan) and primary_fail_plan[primary_idx]:
                primary_idx += 1
                raise _RateLimit429("429 rate limited")
            primary_idx += 1
            llm._last_usage = _mk_usage(model_id)
            return primary_out
        llm._last_usage = _mk_usage(model_id)
        return fallback_out

    return fake_run


# ── Tests ─────────────────────────────────────────────────────────

def test_primary_succeeds_first_try():
    _configure()
    calls: list[str] = []
    llm._run_cheap_agent = _fake_agent_runner(calls, [False], "PRIMARY-OK", "FALLBACK-OK")
    llm.CHEAP_FALLBACK_RETRIES = 2
    out = asyncio.run(llm.call_cheap(MSGS))
    assert out == "PRIMARY-OK", out
    assert calls == [PRIMARY], calls
    assert llm.get_last_usage()["model"] == PRIMARY


def test_retries_then_succeeds():
    _configure()
    calls: list[str] = []
    llm._run_cheap_agent = _fake_agent_runner(calls, [True, False], "PRIMARY-OK", "FALLBACK-OK")
    llm.CHEAP_FALLBACK_RETRIES = 2
    out = asyncio.run(llm.call_cheap(MSGS))
    assert out == "PRIMARY-OK", out
    # Failed once (429), retried, succeeded — NO fallback call.
    assert calls == [PRIMARY, PRIMARY], calls


def test_fallback_after_all_retries():
    _configure()
    calls: list[str] = []
    llm._run_cheap_agent = _fake_agent_runner(calls, [True, True], "PRIMARY-OK", "FALLBACK-OK")
    llm.CHEAP_FALLBACK_RETRIES = 2
    out = asyncio.run(llm.call_cheap(MSGS))
    assert out == "FALLBACK-OK", out
    assert calls == [PRIMARY, PRIMARY, FALLBACK], calls
    # Calling card must show the model that actually answered.
    assert llm.get_last_usage()["model"] == FALLBACK


def test_fallback_disabled_returns_error():
    _configure(fallback=None)  # None → env default; "" → explicitly off
    config._cheap_fallback_override = ""
    calls: list[str] = []
    llm._run_cheap_agent = _fake_agent_runner(calls, [True, True], "PRIMARY-OK", "FALLBACK-OK")
    llm.CHEAP_FALLBACK_RETRIES = 2
    out = asyncio.run(llm.call_cheap(MSGS))
    assert out.startswith("(error:"), out
    assert calls == [PRIMARY, PRIMARY], calls  # no fallback attempted


def test_fallback_same_as_primary_no_dup():
    _configure(cheap=PRIMARY, fallback=PRIMARY)
    calls: list[str] = []
    llm._run_cheap_agent = _fake_agent_runner(calls, [True, True], "PRIMARY-OK", "FALLBACK-OK")
    llm.CHEAP_FALLBACK_RETRIES = 2
    out = asyncio.run(llm.call_cheap(MSGS))
    assert out.startswith("(error:"), out
    assert calls == [PRIMARY, PRIMARY], calls  # no duplicate fallback call


def test_call_cheap_raw_fallback():
    _configure()
    calls: list[str] = []

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)
            self.finish_reason = "stop"

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        def __init__(self, model_id):
            self.model_id = model_id

        async def create(self, **kw):
            calls.append(self.model_id)
            if self.model_id == PRIMARY:
                raise _RateLimit429("429 rate limited")
            return _Resp("FALLBACK-RAW-OK")

    class _Chat:
        def __init__(self, model_id):
            self.completions = _Completions(model_id)

    class _Client:
        def __init__(self, model_id):
            self.chat = _Chat(model_id)

    original = llm._client_for_model
    llm._client_for_model = lambda model_id: _Client(model_id)
    llm.CHEAP_FALLBACK_RETRIES = 2
    try:
        resp, model = asyncio.run(llm.call_cheap_raw(MSGS, extra_body={"reasoning": {"enabled": False}}))
    finally:
        llm._client_for_model = original
    assert resp.choices[0].message.content == "FALLBACK-RAW-OK"
    assert model == FALLBACK, model
    assert calls == [PRIMARY, PRIMARY, FALLBACK], calls


def test_config_set_get_persist_roundtrip():
    snap = _snapshot_overrides()
    try:
        config.set_cheap_fallback_model(FALLBACK)
        assert config.get_cheap_fallback_model() == FALLBACK
        row = db.get_conn().execute(
            "SELECT cheap_fallback_override FROM model_overrides WHERE id = 1"
        ).fetchone()
        assert row["cheap_fallback_override"] == FALLBACK, row

        # Disable → stored as "" and getter returns falsy
        config.set_cheap_fallback_model(None)
        assert not config.get_cheap_fallback_model()
        row = db.get_conn().execute(
            "SELECT cheap_fallback_override FROM model_overrides WHERE id = 1"
        ).fetchone()
        assert row["cheap_fallback_override"] == "", row
    finally:
        _restore_overrides(snap)


# ── Runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in _tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(_tests) - failed}/{len(_tests)} passed")
    sys.exit(1 if failed else 0)
