from types import SimpleNamespace

import pytest

from dbgpt.core import ModelOutput
from dbgpt_app.openapi.api_v2 import no_stream_wrapper
from dbgpt_app.scene.base_chat import ChatParam


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
