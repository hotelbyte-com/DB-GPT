import json

import pytest

from dbgpt_app.openapi.api_v1.react_agent_sse import (
    REACT_AGENT_SSE_V1,
    REACT_AGENT_SSE_V1_REGISTRY,
    classify_react_agent_error,
    emit_react_agent_event,
    terminal_error_events,
)


def _payload(line: str) -> dict:
    assert line.startswith("data: ") and line.endswith("\n\n")
    return json.loads(line.removeprefix("data: "))


@pytest.mark.parametrize(
    ("event", "status"),
    [
        ({"type": "context.status", "state": "normal"}, "update"),
        (
            {
                "type": "step.start",
                "step": 1,
                "id": "step-1",
                "title": "query",
                "detail": "read",
            },
            "running",
        ),
        (
            {"type": "step.output", "step": 1, "id": "step-1", "detail": "row"},
            "streaming",
        ),
        (
            {
                "type": "step.chunk",
                "id": "step-1",
                "output_type": "text",
                "content": "row",
            },
            "streaming",
        ),
        ({"type": "step.meta", "id": "step-1", "action": "sql_query"}, "streaming"),
        ({"type": "step.done", "id": "step-1", "status": "done"}, "done"),
        ({"type": "plan.update", "tasks": []}, "updated"),
        ({"type": "final", "content": "governed answer"}, "success"),
        (
            {
                "type": "error",
                "code": "react_parser_error",
                "message": "agent output could not be parsed",
            },
            "failed",
        ),
        ({"type": "done", "status": "done"}, "done"),
        ({"type": "done", "status": "failed"}, "failed"),
    ],
)
def test_v1_emitter_registers_every_supported_typed_event(event, status):
    payload = _payload(emit_react_agent_event(event))
    assert payload["contractVersion"] == REACT_AGENT_SSE_V1
    assert payload["type"] in REACT_AGENT_SSE_V1_REGISTRY
    assert payload["status"] == status


@pytest.mark.parametrize(
    "event",
    [
        {"type": "unknown"},
        {"type": "step.chunk", "id": "", "content": "row"},
        {"type": "step.output", "step": "1", "id": "step-1", "detail": "row"},
        {"type": "step.done", "id": "step-1", "status": "success"},
        {"type": "final", "content": ""},
        {"type": "final", "content": "answer", "status": "failed"},
        {"type": "error", "code": "", "message": "failed"},
        {"type": "done", "status": "complete"},
    ],
)
def test_v1_emitter_rejects_unknown_or_ambiguous_frames(event):
    with pytest.raises(ValueError):
        emit_react_agent_event(event)


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (RuntimeError("provider authentication failed"), "provider_auth_error"),
        (RuntimeError("model backend unavailable"), "model_backend_error"),
        (RuntimeError("OutputParserException"), "react_parser_error"),
        (RuntimeError("unexpected agent failure"), "react_agent_error"),
    ],
)
def test_terminal_failure_is_error_then_failed_done_never_final(exc, code):
    assert classify_react_agent_error(exc) == code
    events = [_payload(line) for line in terminal_error_events(exc)]
    assert [event["type"] for event in events] == ["error", "done"]
    assert events[0]["code"] == code
    assert events[0]["status"] == "failed"
    assert events[1]["status"] == "failed"
    assert all(event["type"] != "final" for event in events)
