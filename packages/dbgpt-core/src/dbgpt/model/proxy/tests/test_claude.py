"""Tests for Claude-compatible proxy helpers."""

from types import SimpleNamespace

import pytest

from dbgpt.core import ModelMessage, ModelRequest
from dbgpt.model.proxy.llms.claude import (
    ClaudeLLMClient,
    _anthropic_usage,
    _inline_system_messages,
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


def test_anthropic_usage_includes_cached_input_tokens():
    usage = SimpleNamespace(
        input_tokens=0,
        cache_creation_input_tokens=11,
        cache_read_input_tokens=29,
        output_tokens=3,
    )

    assert _anthropic_usage(usage) == {
        "prompt_tokens": 40,
        "completion_tokens": 3,
        "total_tokens": 43,
    }


def test_inline_system_messages_prefixes_first_user_message():
    messages = [{"role": "user", "content": "answer the question"}]

    inlined = _inline_system_messages(messages, ["be careful"])

    assert inlined == [
        {
            "role": "user",
            "content": (
                "System instructions (follow silently; do not summarize or restate):\n"
                "be careful\n\n"
                "User request:\n"
                "answer the question"
            ),
        }
    ]
    assert messages == [{"role": "user", "content": "answer the question"}]


def test_inline_system_messages_preserves_prompt_when_only_system_exists():
    assert _inline_system_messages([], ["be careful"]) == [
        {
            "role": "user",
            "content": (
                "System instructions (follow silently; do not summarize or restate):\n"
                "be careful"
            ),
        }
    ]


def test_build_request_preserves_explicit_zero_temperature():
    client = ClaudeLLMClient(model="test-model", model_alias="test-model")
    request = ModelRequest(
        model="test-model",
        messages=[ModelMessage(role="user", content="hi")],
        temperature=0.0,
    )

    payload = client._build_request(request)

    assert "temperature" in payload
    assert payload["temperature"] == 0.0


@pytest.mark.asyncio
async def test_stream_emits_final_anthropic_usage():
    stream = _FakeMessageStream()
    client = ClaudeLLMClient(
        model="test-model",
        model_alias="test-model",
        client=SimpleNamespace(messages=_FakeMessages(stream)),
    )
    request = ModelRequest(
        model="test-model",
        messages=[ModelMessage(role="user", content="hi")],
    )

    outputs = [output async for output in client.generate_stream(request)]

    assert outputs[-1].text == "OK"
    assert outputs[-1].usage == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }
    assert stream.final_message_requested


class _FakeMessages:
    def __init__(self, stream):
        self._stream = stream

    def stream(self, **kwargs):
        return self._stream


class _FakeMessageStream:
    def __init__(self):
        self.current_message_snapshot = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=0, output_tokens=0)
        )
        self.text_stream = self._text_stream()
        self.final_message_requested = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def _text_stream(self):
        yield "OK"

    async def get_final_message(self):
        self.final_message_requested = True
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=2))
