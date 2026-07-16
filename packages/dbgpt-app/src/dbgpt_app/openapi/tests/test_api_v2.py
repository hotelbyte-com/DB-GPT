import json
from types import SimpleNamespace

import pytest

from dbgpt.core import ModelOutput
from dbgpt.core.schema.api import ErrorCode
from dbgpt_app.openapi.api_v2 import no_stream_wrapper
from dbgpt_app.scene import base_chat
from dbgpt_app.scene.base_chat import BaseChat, ChatParam
from dbgpt_app.scene.exceptions import ContextAppException


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
