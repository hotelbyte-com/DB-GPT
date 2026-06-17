"""Tests for Claude-compatible proxy helpers."""

from types import SimpleNamespace

from dbgpt.model.proxy.llms.claude import (
    _request_stream_enabled,
    _token_count_value,
)


def test_request_stream_enabled_honors_context_stream_false():
    assert _request_stream_enabled({"context": {"stream": False}}) is False


def test_request_stream_enabled_defaults_to_true_for_streaming_adapter():
    assert _request_stream_enabled({}) is True


def test_token_count_value_accepts_anthropic_sdk_object():
    result = SimpleNamespace(input_tokens=42)

    assert _token_count_value(result) == 42


def test_token_count_value_accepts_legacy_int_and_dict():
    assert _token_count_value(7) == 7
    assert _token_count_value({"input_tokens": 9}) == 9
