"""Tests for Claude-compatible proxy helpers."""

from types import SimpleNamespace

import pytest

from dbgpt.core import ModelMessage, ModelMessageRoleType, ModelRequest
from dbgpt.core.schema.api import ErrorCode
from dbgpt.model.proxy.llms.claude import (
    ClaudeLLMClient,
    _anthropic_usage,
    _inline_system_messages,
    _request_stream_enabled,
    _token_count_value,
)

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


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
        messages=[ModelMessage(role=ModelMessageRoleType.HUMAN, content="hi")],
        temperature=0.0,
    )

    payload = client._build_request(request)

    assert "temperature" in payload
    assert payload["temperature"] == 0.0


def test_build_request_maps_structured_output_to_forced_anthropic_tool():
    client = ClaudeLLMClient(model="MiniMax-M3", model_alias="MiniMax-M3")
    request = ModelRequest(
        model="MiniMax-M3",
        messages=[ModelMessage(role=ModelMessageRoleType.HUMAN, content="hi")],
        response_format=RESPONSE_FORMAT,
    )

    payload = client._build_request(request)

    assert client.supports_response_format is True
    assert "output_config" not in payload
    assert payload["tools"] == [
        {
            "name": "answer",
            "description": "Emit the final structured response.",
            "input_schema": RESPONSE_FORMAT["json_schema"]["schema"],
        }
    ]
    assert payload["tool_choice"] == {
        "type": "tool",
        "name": "answer",
    }


@pytest.mark.asyncio
async def test_generate_serializes_forced_tool_input_as_structured_response():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="internal"),
            SimpleNamespace(type="tool_use", name="answer", input={"answer": "可以"}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=4, output_tokens=3),
    )
    client = ClaudeLLMClient(
        model="MiniMax-M3",
        model_alias="MiniMax-M3",
        client=SimpleNamespace(messages=_SuccessfulMessages(response)),
    )
    request = ModelRequest(
        model="MiniMax-M3",
        messages=[ModelMessage(role=ModelMessageRoleType.HUMAN, content="hi")],
        response_format=RESPONSE_FORMAT,
    )

    output = await client.generate(request)

    assert output.error_code == 0
    assert output.text == '{"answer":"可以"}'


@pytest.mark.asyncio
async def test_generate_rejects_missing_forced_tool_response():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="not json")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=4, output_tokens=3),
    )
    client = ClaudeLLMClient(
        model="MiniMax-M3",
        model_alias="MiniMax-M3",
        client=SimpleNamespace(messages=_SuccessfulMessages(response)),
    )
    request = ModelRequest(
        model="MiniMax-M3",
        messages=[ModelMessage(role=ModelMessageRoleType.HUMAN, content="hi")],
        response_format=RESPONSE_FORMAT,
    )

    output = await client.generate(request)

    assert output.error_code == ErrorCode.VALIDATION_TYPE_ERROR.value
    assert output.model_context == {
        "upstream_error": {
            "kind": "structured_output_invalid",
            "status_code": None,
        }
    }


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


@pytest.mark.asyncio
async def test_generate_classifies_anthropic_rate_limit_without_leaking_body():
    client = ClaudeLLMClient(
        model="test-model",
        model_alias="test-model",
        client=SimpleNamespace(messages=_FailingMessages()),
    )
    request = ModelRequest(
        model="test-model",
        messages=[ModelMessage(role="user", content="hi")],
    )

    output = await client.generate(request)

    assert output.error_code == ErrorCode.RATE_LIMIT.value
    assert output.model_context == {
        "upstream_error": {"kind": "rate_limit", "status_code": 429}
    }
    assert "secret-provider-body" not in output.text


@pytest.mark.asyncio
async def test_generate_keeps_unknown_provider_400_as_upstream_error():
    client = ClaudeLLMClient(
        model="MiniMax-M3",
        model_alias="MiniMax-M3",
        client=SimpleNamespace(messages=_StructuredOutputRejectingMessages()),
    )
    request = ModelRequest(
        model="MiniMax-M3",
        messages=[ModelMessage(role=ModelMessageRoleType.HUMAN, content="hi")],
        response_format=RESPONSE_FORMAT,
    )

    output = await client.generate(request)

    assert output.error_code == ErrorCode.INTERNAL_ERROR.value
    assert output.model_context == {
        "upstream_error": {
            "kind": "upstream_error",
            "status_code": 400,
        }
    }
    assert "secret-provider-body" not in output.text


class _FakeMessages:
    def __init__(self, stream):
        self._stream = stream

    def stream(self, **kwargs):
        return self._stream


class _SuccessfulMessages:
    def __init__(self, response):
        self._response = response

    async def create(self, **_kwargs):
        return self._response


class _TypedProviderError(Exception):
    status_code = 429


class _FailingMessages:
    async def create(self, **_kwargs):
        raise _TypedProviderError("secret-provider-body")


class _StructuredOutputRejectingError(Exception):
    status_code = 400


class _StructuredOutputRejectingMessages:
    async def create(self, **_kwargs):
        raise _StructuredOutputRejectingError("secret-provider-body")


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
