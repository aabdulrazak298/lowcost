"""Unit tests for multi-turn context fix — history_to_turns + referential guard.

Pure functions, no LLM calls. Run: python -m pytest tests/test_multiturn.py -v
"""
from processor import history_to_turns, turns_to_string, _is_referential


def test_history_to_turns_list():
    assert history_to_turns([{"role": "user", "content": "hi"}]) == \
        [{"role": "user", "content": "hi"}]


def test_history_to_turns_list_skips_bad_entries():
    assert history_to_turns([
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "nope"},   # wrong role
        "not a dict",                              # non-dict
        {"role": "assistant", "content": ""},      # empty content
        {"role": "assistant", "content": "hello"},
    ]) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_history_to_turns_string():
    s = "User: hello\nAssistant: hi there"
    assert history_to_turns(s) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_history_to_turns_empty():
    assert history_to_turns("") == []
    assert history_to_turns(None) == []
    assert history_to_turns([]) == []


def test_turns_roundtrip():
    turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert history_to_turns(turns_to_string(turns)) == turns


def test_referential_true():
    for q in ["number 5", "number 10", "the second one", "what about that",
              "the first item", "the third option", "what about it", "that one"]:
        assert _is_referential(q) is True, q


def test_referential_false():
    for q in ["what is the capital of France",
              "explain the EU defies US trade",
              "write a python function to sort a list",
              "tell me about the industrial revolution"]:
        assert _is_referential(q) is False, q
