import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from dbgpt.core import ModelOutput, ModelRequest, ModelRequestContext
from dbgpt.core.schema.api import ErrorCode, JSONSchemaResponseFormat
from dbgpt_app.openapi.api_v2 import check_chat_request, no_stream_wrapper
from dbgpt_app.scene import base_chat
from dbgpt_app.scene.base_chat import BaseChat, ChatParam
from dbgpt_app.scene.exceptions import ContextAppException
from dbgpt_client.schema import ChatCompletionRequestBody

STRUCTURED_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "knowledge_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


def test_chat_completion_request_validates_typed_json_schema():
    request = ChatCompletionRequestBody(
        model="test-model",
        messages="hello",
        response_format=STRUCTURED_RESPONSE_FORMAT,
    )

    assert request.response_format.to_provider_dict() == STRUCTURED_RESPONSE_FORMAT


def test_chat_completion_request_rejects_invalid_json_schema():
    invalid_format = {
        **STRUCTURED_RESPONSE_FORMAT,
        "json_schema": {
            **STRUCTURED_RESPONSE_FORMAT["json_schema"],
            "schema": {"type": "not-a-json-schema-type"},
        },
    }

    request = ChatCompletionRequestBody(
        model="test-model",
        messages="hello",
        response_format=invalid_format,
    )

    with pytest.raises(HTTPException) as exc_info:
        check_chat_request(request)
    assert exc_info.value.status_code == 400
    assert (
        exc_info.value.detail["error"]["code"]
        == "invalid_response_format_schema"
    )


@pytest.mark.parametrize(
    "unsafe_schema",
    [
        {"type": "string"},
        {"$ref": "http://127.0.0.1:9/internal-schema"},
        {"type": "string", "pattern": "(a+)+$"},
        {"type": "object", "properties": {"x": {"$ref": "file:///etc/passwd"}}},
    ],
)
def test_chat_completion_request_rejects_network_file_and_regex_schemas(
    unsafe_schema,
):
    response_format = {
        **STRUCTURED_RESPONSE_FORMAT,
        "json_schema": {
            **STRUCTURED_RESPONSE_FORMAT["json_schema"],
            "schema": unsafe_schema,
        },
    }
    request = ChatCompletionRequestBody(
        model="test-model",
        messages="hello",
        response_format=response_format,
    )

    with pytest.raises(HTTPException) as exc_info:
        check_chat_request(request)
    assert exc_info.value.status_code == 400
    assert (
        exc_info.value.detail["error"]["code"]
        == "invalid_response_format_schema"
    )


def test_chat_completion_request_allows_keyword_shaped_business_fields():
    response_format = {
        **STRUCTURED_RESPONSE_FORMAT,
        "json_schema": {
            **STRUCTURED_RESPONSE_FORMAT["json_schema"],
            "schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "$ref": {"type": "string"},
                },
                "required": ["pattern", "$ref"],
                "additionalProperties": False,
            },
        },
    }

    check_chat_request(
        ChatCompletionRequestBody(
            model="test-model",
            messages="hello",
            response_format=response_format,
        )
    )


def test_structured_output_fails_closed_for_streaming_api_requests():
    request = ChatCompletionRequestBody(
        model="test-model",
        messages="hello",
        stream=True,
        response_format=STRUCTURED_RESPONSE_FORMAT,
    )

    with pytest.raises(HTTPException) as exc_info:
        check_chat_request(request)

    assert exc_info.value.status_code == 400
    assert (
        exc_info.value.detail["error"]["code"]
        == "unsupported_response_format_stream"
    )


def test_explicit_stream_mode_overrides_scene_template_default():
    param = ChatParam(
        chat_session_id="conv-1",
        current_user_input="hello",
        model_name="test-model",
        select_param=None,
        chat_mode=None,
        stream=False,
    )

    assert param.stream_mode(template_default=True) is False


def test_chat_param_exposes_validated_provider_response_format():
    response_format = JSONSchemaResponseFormat.model_validate(
        STRUCTURED_RESPONSE_FORMAT
    )
    param = ChatParam(
        chat_session_id="conv-1",
        current_user_input="hello",
        model_name="test-model",
        select_param=None,
        chat_mode=None,
        response_format=response_format,
    )

    assert param.provider_response_format() == STRUCTURED_RESPONSE_FORMAT


@pytest.mark.asyncio
async def test_base_chat_adds_response_format_to_provider_model_request(monkeypatch):
    response_format = JSONSchemaResponseFormat.model_validate(
        STRUCTURED_RESPONSE_FORMAT
    )
    param = ChatParam(
        chat_session_id="conv-1",
        current_user_input="hello",
        model_name="test-model",
        select_param=None,
        chat_mode=None,
        stream=False,
        response_format=response_format,
    )
    model_request = ModelRequest(
        model="test-model",
        messages=[],
        context=ModelRequestContext(),
    )

    class Composer:
        def __init__(self, **_kwargs):
            pass

        async def call(self, **_kwargs):
            return model_request

    class CurrentMessage:
        start_date = None
        tokens = None

        def get_history_message(self):
            return []

        def start_new_round(self):
            pass

        def add_user_message(self, _content):
            pass

    class Chat:
        _chat_param = param
        current_message = CurrentMessage()
        current_user_input = SimpleNamespace(content="hello")
        history_messages = []
        chat_mode = SimpleNamespace(value=lambda: "chat_normal")
        llm_model = "test-model"
        prompt_template = SimpleNamespace(
            stream_out=False, prompt=None, str_history=False
        )
        llm_client = None
        _message_version = "v2"
        llm_echo = False
        model_cache_enable = False

        async def prepare_input_values(self):
            return {}

        def llm_temperature(self):
            return 0

        def llm_max_new_tokens(self):
            return 100

        def memory_config(self):
            return None

    monkeypatch.setattr(base_chat, "AppChatComposerOperator", Composer)

    request = await BaseChat._build_model_request(Chat())

    assert request.response_format == STRUCTURED_RESPONSE_FORMAT


@pytest.mark.asyncio
async def test_no_stream_wrapper_preserves_real_model_usage():
    request = SimpleNamespace(conv_uid="conv-1", model="test-model")

    class Chat:
        async def nostream_call_with_output(self):
            return (
                "OK",
                ModelOutput.build(
                    "OK",
                    usage={"prompt_tokens": 11, "completion_tokens": 3},
                ),
            )

        async def stream_call(self, **_kwargs):
            raise AssertionError(
                "non-stream API must not call the streaming provider path"
            )

    response = await no_stream_wrapper(request, Chat())

    assert response.choices[0].message.content == "OK"
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 14


@pytest.mark.asyncio
async def test_no_stream_wrapper_accepts_schema_valid_json_with_chinese_quotes():
    request = ChatCompletionRequestBody(
        model="test-model",
        messages="hello",
        conv_uid="conv-1",
        response_format=STRUCTURED_RESPONSE_FORMAT,
    )
    content = '{"answer":"选择“新增”后保存"}'

    class Chat:
        async def nostream_call_with_output(self):
            return "rendered view must not replace typed JSON", ModelOutput.build(
                content
            )

    response = await no_stream_wrapper(request, Chat())

    assert response.choices[0].message.content == content


@pytest.mark.asyncio
async def test_no_stream_wrapper_fails_closed_on_invalid_structured_output():
    request = ChatCompletionRequestBody(
        model="test-model",
        messages="hello",
        conv_uid="conv-1",
        response_format=STRUCTURED_RESPONSE_FORMAT,
    )

    class Chat:
        async def nostream_call_with_output(self):
            content = '{"answer":"选择"新增"后保存"}'
            return content, ModelOutput.build(content)

    response = await no_stream_wrapper(request, Chat())
    body = json.loads(response.body)

    assert response.status_code == 502
    assert body["error"]["type"] == "structured_output_error"
    assert body["error"]["code"] == "structured_output_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_code", "kind", "upstream_status", "response_status"),
    [
        (ErrorCode.RATE_LIMIT.value, "rate_limit", 429, 429),
        (ErrorCode.INTERNAL_ERROR.value, "upstream_error", 503, 502),
    ],
)
async def test_no_stream_wrapper_returns_typed_sanitized_provider_error(
    error_code, kind, upstream_status, response_status
):
    request = SimpleNamespace(conv_uid="conv-1", model="test-model")
    provider_output = ModelOutput(
        text="provider body with secret-key-must-not-leak",
        error_code=error_code,
        model_context={
            "upstream_error": {
                "kind": kind,
                "status_code": upstream_status,
            }
        },
    )

    class Chat:
        async def nostream_call_with_output(self):
            return "", provider_output

    response = await no_stream_wrapper(request, Chat())
    body = json.loads(response.body)

    assert response.status_code == response_status
    assert body["error"]["code"] == kind
    assert body["error"]["upstream_status"] == upstream_status
    assert body["error"]["type"] == "upstream_provider_error"
    assert "secret-key-must-not-leak" not in response.body.decode()


@pytest.mark.asyncio
async def test_no_stream_wrapper_returns_typed_unsupported_provider_error():
    request = SimpleNamespace(conv_uid="conv-1", model="test-model")
    provider_output = ModelOutput(
        text="Selected model provider does not support structured output.",
        error_code=ErrorCode.VALIDATION_TYPE_ERROR.value,
        model_context={
            "upstream_error": {
                "kind": "structured_output_unsupported",
                "status_code": None,
            }
        },
    )

    class Chat:
        async def nostream_call_with_output(self):
            return "", provider_output

    response = await no_stream_wrapper(request, Chat())
    body = json.loads(response.body)

    assert response.status_code == 422
    assert body["error"]["type"] == "structured_output_error"
    assert body["error"]["code"] == "structured_output_unsupported"


@pytest.mark.asyncio
async def test_no_stream_chat_preserves_failed_model_output(monkeypatch):
    failed_output = ModelOutput(
        text="Upstream model provider rate limited the request.",
        error_code=ErrorCode.RATE_LIMIT.value,
        model_context={"upstream_error": {"kind": "rate_limit", "status_code": 429}},
    )

    response, final_output = await _run_no_stream_context_error(
        monkeypatch, failed_output
    )

    assert response == ""
    assert final_output is failed_output


@pytest.mark.asyncio
async def test_no_stream_chat_does_not_mask_post_processing_failure(monkeypatch):
    successful_output = ModelOutput.build("provider answered")
    response, final_output = await _run_no_stream_context_error(
        monkeypatch, successful_output
    )

    assert response == ""
    assert final_output is None


async def _run_no_stream_context_error(monkeypatch, model_output):
    class CurrentMessage:
        def add_view_message(self, _message):
            pass

        def end_current_round(self):
            pass

    class Chat:
        current_message = CurrentMessage()
        _executor = None

        async def _build_model_request(self):
            return SimpleNamespace(to_dict=lambda: {}, span_id=None)

        async def _no_streaming_call_with_retry(self, _payload):
            raise ContextAppException("processing failed", "safe error", model_output)

        def current_ai_response(self):
            return ""

        def message_adjust(self):
            pass

    async def call_immediately(_executor, func, *args):
        return func(*args)

    monkeypatch.setattr(base_chat, "blocking_func_to_async", call_immediately)

    return await BaseChat.nostream_call_with_output(Chat())
