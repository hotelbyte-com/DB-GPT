"""Typed ReAct SSE facade over the configured manufacturing Mongo chat_data app."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional

from pydantic import ConfigDict, model_validator

from dbgpt._private.pydantic import BaseModel, Field
from dbgpt.component import ComponentType
from dbgpt.model.cluster import WorkerManagerFactory
from dbgpt_app.openapi.api_v1.react_agent_sse import (
    emit_react_agent_event,
    terminal_error_events,
)
from dbgpt_app.openapi.api_view_model import ConversationVo
from dbgpt_ext.datasource.nosql.mongo_chat_data import (
    MongoChatDataRouter,
    validate_data_provenance,
)

MANUFACTURING_DATA_AGENT_SOURCE = "manufacturing-agent-os-data-agent"
MANUFACTURING_QUERY_CONTRACT_VERSION = "manufacturing.data-query/v1"


class ManufacturingTimeWindowV1(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_bounds(self) -> "ManufacturingTimeWindowV1":
        if self.end <= self.start:
            raise ValueError("manufacturing query time window must have positive width")
        return self


class ManufacturingQueryContractV1(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    version: Literal[MANUFACTURING_QUERY_CONTRACT_VERSION]
    logical_group: Literal["manufacturing"] = Field(alias="logicalGroup")
    intent_class: str = Field(alias="intentClass", min_length=1)
    sources: List[str] = Field(min_length=1)
    read_only: Literal[True] = Field(alias="readOnly")
    time_window: ManufacturingTimeWindowV1 = Field(alias="timeWindow")
    row_limit: int = Field(alias="rowLimit", ge=1, le=100)


def is_manufacturing_mongo_source(dialogue: ConversationVo) -> bool:
    ext_info = dialogue.ext_info
    return bool(
        isinstance(ext_info, dict)
        and ext_info.get("source") == MANUFACTURING_DATA_AGENT_SOURCE
    )


def parse_manufacturing_query_contract(
    ext_info: Optional[Dict[str, Any]],
) -> ManufacturingQueryContractV1:
    if not isinstance(ext_info, dict) or "query_contract" not in ext_info:
        raise ValueError("manufacturing query contract is required")
    raw = ext_info["query_contract"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, Mapping):
        raise ValueError("manufacturing query contract must be an object")
    return ManufacturingQueryContractV1.model_validate(raw)


async def stream_manufacturing_mongo_query(
    dialogue: ConversationVo,
    *,
    router: Optional[MongoChatDataRouter] = None,
    worker_manager: Any = None,
) -> AsyncGenerator[str, None]:
    """Run the same configured Mongo path as chat_data and emit strict SSE v1."""

    try:
        contract = parse_manufacturing_query_contract(dialogue.ext_info)
        select_param = str(dialogue.select_param or "").strip()
        if select_param != contract.logical_group:
            raise ValueError("manufacturing query logical group mismatch")
        router = router or MongoChatDataRouter.from_env()
        if not router.can_handle(select_param):
            raise RuntimeError(
                "configured manufacturing Mongo chat_data app unavailable"
            )
        if worker_manager is None:
            from dbgpt._private.config import Config

            system_app = Config().SYSTEM_APP
            worker_manager = system_app.get_component(
                ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
            ).create()
        response = await router.answer(
            chat_param=select_param,
            prompt=_contract_bound_prompt(dialogue.user_input, contract),
            model=str(dialogue.model_name or "").strip(),
            worker_manager=worker_manager,
            temperature=dialogue.temperature,
            max_new_tokens=dialogue.max_new_tokens,
            conv_uid=dialogue.conv_uid,
        )
        final_content, artifact_receipt = _validated_response_receipt(response)
    except Exception as exc:
        for event in terminal_error_events(exc):
            yield event
        return

    step_id = "step-1"
    yield emit_react_agent_event({"type": "context.status", "state": "normal"})
    yield emit_react_agent_event(
        {
            "type": "step.start",
            "step": 1,
            "id": step_id,
            "title": "manufacturing_mongo_chat_data",
            "detail": "Execute the configured read-only manufacturing Mongo query",
        }
    )
    yield emit_react_agent_event(
        {
            "type": "step.meta",
            "id": step_id,
            "action": "manufacturing_mongo_chat_data",
            "action_intention": "Query the configured manufacturing source",
            "action_reason": "The request carries a validated read-only contract",
            "action_input": contract.model_dump_json(by_alias=True),
            "title": "manufacturing_mongo_chat_data",
        }
    )
    yield emit_react_agent_event(
        {
            "type": "step.chunk",
            "id": step_id,
            "output_type": "json",
            "content": artifact_receipt,
        }
    )
    yield emit_react_agent_event({"type": "step.done", "id": step_id, "status": "done"})
    yield emit_react_agent_event({"type": "final", "content": final_content})
    yield emit_react_agent_event({"type": "done", "status": "done"})


def _contract_bound_prompt(
    user_input: Any, contract: ManufacturingQueryContractV1
) -> str:
    question = str(user_input or "").strip()
    return (
        "Authoritative manufacturing query contract (JSON):\n"
        f"{contract.model_dump_json(by_alias=True)}\n\n"
        "User question and context:\n"
        f"{question}"
    )


def _validated_response_receipt(response: Any) -> tuple[str, Dict[str, Any]]:
    if not isinstance(response, Mapping):
        raise ValueError("manufacturing Mongo response is not an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("manufacturing Mongo response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("manufacturing Mongo response has no final content")
    raw = response.get("raw")
    artifact = response.get("artifact")
    if not isinstance(raw, Mapping) or not isinstance(artifact, Mapping):
        raise ValueError("manufacturing Mongo response has no typed evidence")
    rows = raw.get("rows")
    source = raw.get("source")
    if not isinstance(rows, list) or not isinstance(source, str) or not source.strip():
        raise ValueError("manufacturing Mongo response source receipt is incomplete")
    artifact_rows = artifact.get("rows")
    artifact_source = artifact.get("source")
    artifact_type = artifact.get("type")
    if not isinstance(artifact_type, str) or not artifact_type.strip():
        raise ValueError("manufacturing Mongo response artifact receipt is incomplete")
    if artifact_source != source or artifact_rows != rows:
        raise ValueError("manufacturing Mongo response evidence is inconsistent")
    provenance = validate_data_provenance(
        response.get("provenance"),
        source=source,
        row_count=len(rows),
    )
    return content.strip(), provenance
