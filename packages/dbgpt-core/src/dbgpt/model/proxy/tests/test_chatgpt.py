"""Tests for OpenAI-compatible proxy error handling."""

from types import SimpleNamespace

import pytest

from dbgpt.core import ModelMessage, ModelRequest
from dbgpt.core.schema.api import ErrorCode
from dbgpt.model.proxy.llms.chatgpt import OpenAILLMClient


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
