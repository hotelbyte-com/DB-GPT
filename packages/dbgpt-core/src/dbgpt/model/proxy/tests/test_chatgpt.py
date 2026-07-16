"""Tests for OpenAI-compatible proxy error handling."""

from types import SimpleNamespace

import pytest

from dbgpt.core import ModelInferenceMetrics, ModelMessage, ModelOutput, ModelRequest
from dbgpt.core.schema.api import ErrorCode
from dbgpt.model.cluster.worker.default_worker import DefaultModelWorker
from dbgpt.model.proxy.llms.chatgpt import OpenAILLMClient
from dbgpt.model.proxy.llms.provider_error import (
    StructuredOutputUnsupportedError,
)
from dbgpt.model.proxy.llms.proxy_model import ProxyModel
from dbgpt.model.utils.llm_utils import parse_model_request

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


def test_openai_request_passes_through_structured_response_format():
    client = OpenAILLMClient(
        model="test-model",
        model_alias="test-model",
        openai_client=_FailingOpenAIClient(),
    )
    request = ModelRequest(
        model="test-model",
        messages=[ModelMessage(role="user", content="hi")],
        response_format=RESPONSE_FORMAT,
    )

    payload = client._build_request(request)

    assert payload["response_format"] == RESPONSE_FORMAT


def test_openai_request_without_schema_preserves_legacy_payload():
    client = OpenAILLMClient(
        model="test-model",
        model_alias="test-model",
        openai_client=_FailingOpenAIClient(),
    )
    request = ModelRequest(
        model="test-model",
        messages=[ModelMessage(role="user", content="hi")],
    )

    assert "response_format" not in client._build_request(request)


def test_model_request_parser_rejects_schema_for_unsupported_provider():
    params = {
        "messages": [ModelMessage(role="user", content="hi")],
        "response_format": RESPONSE_FORMAT,
    }

    with pytest.raises(StructuredOutputUnsupportedError):
        parse_model_request(params, "test-model")


def test_model_request_parser_preserves_schema_for_supported_provider():
    params = {
        "messages": [ModelMessage(role="user", content="hi")],
        "response_format": RESPONSE_FORMAT,
    }

    request = parse_model_request(
        params, "test-model", response_format_supported=True
    )

    assert request.response_format == RESPONSE_FORMAT
    assert request.to_dict()["response_format"] == RESPONSE_FORMAT


def test_worker_rejects_structured_output_for_unregistered_provider_capability():
    worker = object.__new__(DefaultModelWorker)
    worker.model = object()

    with pytest.raises(StructuredOutputUnsupportedError):
        worker._validate_response_format_support(
            {"response_format": RESPONSE_FORMAT}
        )

    output = worker.generate({"response_format": RESPONSE_FORMAT})

    assert output.error_code == ErrorCode.VALIDATION_TYPE_ERROR.value
    assert output.model_context == {
        "upstream_error": {
            "kind": "structured_output_unsupported",
            "status_code": None,
        }
    }


def test_worker_allows_declared_structured_output_provider_capability():
    model = object.__new__(ProxyModel)
    model.proxy_llm_client = SimpleNamespace(supports_response_format=True)
    worker = object.__new__(DefaultModelWorker)
    worker.model = model

    worker._validate_response_format_support({"response_format": RESPONSE_FORMAT})


def test_worker_preserves_typed_provider_error_context():
    worker = object.__new__(DefaultModelWorker)
    request_context = {"request_marker": "preserved"}
    provider_output = ModelOutput(
        text="Upstream model provider returned invalid structured output.",
        error_code=ErrorCode.VALIDATION_TYPE_ERROR.value,
        model_context={
            "upstream_error": {
                "kind": "structured_output_invalid",
                "status_code": None,
            }
        },
    )

    output, _, _, _ = worker._handle_output(
        provider_output,
        previous_response="",
        model_context=request_context,
        last_metrics=ModelInferenceMetrics.create_metrics(),
        is_first_generate=True,
    )

    assert output.model_context == {
        "request_marker": "preserved",
        "upstream_error": {
            "kind": "structured_output_invalid",
            "status_code": None,
        },
    }
    assert request_context == {"request_marker": "preserved"}


@pytest.mark.asyncio
async def test_generate_classifies_openai_rate_limit_without_leaking_body():
    client = OpenAILLMClient(
        model="test-model",
        model_alias="test-model",
        openai_client=_FailingOpenAIClient(),
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


class _TypedProviderError(Exception):
    status_code = 429


class _FailingCompletions:
    async def create(self, **_kwargs):
        raise _TypedProviderError("secret-provider-body")


class _FailingOpenAIClient:
    default_headers = {}
    chat = SimpleNamespace(completions=_FailingCompletions())
