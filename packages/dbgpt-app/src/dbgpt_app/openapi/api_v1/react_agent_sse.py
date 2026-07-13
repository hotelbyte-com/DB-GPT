"""Versioned typed SSE contract for the ReAct Agent endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping

REACT_AGENT_SSE_V1 = "react-agent-sse.v1"


@dataclass(frozen=True)
class ReActSSEEventSpec:
    required: frozenset[str]
    fixed_status: str | None = None
    allowed_statuses: frozenset[str] = frozenset()


def close_terminate_step(
    round_step_map: MutableMapping[int, str],
    round_num: int,
) -> str | None:
    step_id = round_step_map.pop(round_num, None)
    if not step_id:
        return None
    return emit_react_agent_event(
        {"type": "step.done", "id": step_id, "status": "done"}
    )


REACT_AGENT_SSE_V1_REGISTRY: Mapping[str, ReActSSEEventSpec] = {
    "context.status": ReActSSEEventSpec(frozenset({"state"}), "update"),
    "step.start": ReActSSEEventSpec(
        frozenset({"step", "id", "title", "detail"}), "running"
    ),
    "step.output": ReActSSEEventSpec(frozenset({"step", "id", "detail"}), "streaming"),
    "step.chunk": ReActSSEEventSpec(
        frozenset({"id", "output_type", "content"}), "streaming"
    ),
    "step.meta": ReActSSEEventSpec(frozenset({"id"}), "streaming"),
    "step.done": ReActSSEEventSpec(
        frozenset({"id", "status"}), allowed_statuses=frozenset({"done", "failed"})
    ),
    "plan.update": ReActSSEEventSpec(frozenset({"tasks"}), "updated"),
    "final": ReActSSEEventSpec(frozenset({"content"}), "success"),
    "error": ReActSSEEventSpec(frozenset({"code", "message"}), "failed"),
    "done": ReActSSEEventSpec(
        frozenset({"status"}), allowed_statuses=frozenset({"done", "failed"})
    ),
}

REACT_AGENT_ERROR_MESSAGES = {
    "provider_auth_error": "The configured model provider rejected authentication.",
    "model_backend_error": "The configured model backend was unavailable.",
    "react_parser_error": "The ReAct agent output could not be parsed.",
    "react_agent_error": "The ReAct agent did not complete successfully.",
}


def emit_react_agent_event(event: Dict[str, Any]) -> str:
    payload = dict(event)
    event_type = str(payload.get("type") or "").strip()
    spec = REACT_AGENT_SSE_V1_REGISTRY.get(event_type)
    if spec is None:
        raise ValueError(f"unsupported ReAct SSE event type: {event_type!r}")
    supplied_version = payload.get("contractVersion")
    if supplied_version not in (None, REACT_AGENT_SSE_V1):
        raise ValueError("unsupported ReAct SSE contract version")
    payload["contractVersion"] = REACT_AGENT_SSE_V1
    payload["type"] = event_type

    missing = [field for field in spec.required if field not in payload]
    if missing:
        raise ValueError(f"ReAct SSE event {event_type!r} is missing {sorted(missing)}")
    for field in ("id", "title", "detail", "code", "message", "content"):
        if field in spec.required and isinstance(payload.get(field), str):
            if not payload[field].strip():
                raise ValueError(f"ReAct SSE event {event_type!r} has empty {field}")

    supplied_status = payload.get("status")
    if spec.fixed_status is not None:
        if supplied_status not in (None, spec.fixed_status):
            raise ValueError(f"ReAct SSE event {event_type!r} has invalid status")
        payload["status"] = spec.fixed_status
    elif supplied_status not in spec.allowed_statuses:
        raise ValueError(f"ReAct SSE event {event_type!r} has invalid status")

    if event_type == "error" and payload["code"] not in REACT_AGENT_ERROR_MESSAGES:
        raise ValueError("ReAct SSE error code is not registered")
    if event_type == "plan.update" and not isinstance(payload["tasks"], list):
        raise ValueError("ReAct SSE plan.update tasks must be a list")
    if event_type in {"step.start", "step.output"}:
        if not isinstance(payload["step"], int) or payload["step"] <= 0:
            raise ValueError(f"ReAct SSE {event_type} step must be positive")

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def classify_react_agent_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "auth" in text or "unauthorized" in text or "forbidden" in text:
        return "provider_auth_error"
    if "outputparser" in text or "parser" in text or "could not parse" in text:
        return "react_parser_error"
    if "model" in text or "backend" in text or "provider" in text:
        return "model_backend_error"
    return "react_agent_error"


def terminal_error_events(exc: BaseException) -> Iterable[str]:
    code = classify_react_agent_error(exc)
    yield emit_react_agent_event(
        {
            "type": "error",
            "code": code,
            "message": REACT_AGENT_ERROR_MESSAGES[code],
        }
    )
    yield emit_react_agent_event({"type": "done", "status": "failed"})
