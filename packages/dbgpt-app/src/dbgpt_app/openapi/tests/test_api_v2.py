from types import SimpleNamespace

import pytest

from dbgpt.core import ModelOutput
from dbgpt_app.openapi.api_v2 import no_stream_wrapper


@pytest.mark.asyncio
async def test_no_stream_wrapper_preserves_real_model_usage():
    request = SimpleNamespace(conv_uid="conv-1", model="test-model")

    class Chat:
        async def stream_call(self, *, text_output, incremental):
            assert text_output is False
            assert incremental is False
            yield ModelOutput.build(
                "OK",
                usage={"prompt_tokens": 11, "completion_tokens": 3},
            )

    response = await no_stream_wrapper(request, Chat())

    assert response.choices[0].message.content == "OK"
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 14
