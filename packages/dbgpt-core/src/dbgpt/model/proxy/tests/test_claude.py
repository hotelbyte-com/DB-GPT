from types import SimpleNamespace

import pytest

from dbgpt.core import ModelMessage, ModelRequest
from dbgpt.model.proxy.llms.claude import ClaudeLLMClient


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
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=7, output_tokens=2)
        )
