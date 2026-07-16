"""No-LLM streaming path for declared governed data-query contracts."""

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Callable

from dbgpt.core import StorageConversation
from dbgpt_app.openapi.api_v1.react_agent_sse import emit_react_agent_event
from dbgpt_app.openapi.api_view_model import ConversationVo
from dbgpt_app.scene.chat_db.query_contract import DataQueryContract
from dbgpt_app.scene.chat_db.query_contract_compiler import (
    ContractCompilation,
    compile_data_query_contract,
)
from dbgpt_app.scene.chat_db.query_contract_execution import (
    GovernedQueryOutcome,
    execute_compiled_data_query,
)
from dbgpt_serve.conversation.serve import Serve as ConversationServe
from dbgpt_serve.datasource.manages import ConnectorManager

logger = logging.getLogger(__name__)

ContractResolver = Callable[[ConversationVo, str], tuple[Any, Any]]


async def stream_governed_query_contract(
    dialogue: ConversationVo,
    *,
    system_app: Any,
    contract_resolver: ContractResolver,
) -> AsyncGenerator[str, None]:
    """Compile and execute a typed query contract without constructing an LLM."""

    user_input = _extract_user_question(dialogue.user_input)
    contract, source_resolution, compilation = _resolve_and_compile(
        dialogue, user_input, contract_resolver
    )
    connector, source_name, source_type, schema = _load_connector(
        source_resolution, system_app
    )
    if contract is None:
        contract = DataQueryContract.model_validate(
            {
                "version": "hotelbyte.data-query/v1",
                "logical_group": "hotel-be",
            }
        )
    outcome = execute_compiled_data_query(
        compilation,
        contract,
        connector,
        physical_source_name=source_name,
        physical_source_type=source_type,
        source_resolution=(
            source_resolution.model_dump() if source_resolution is not None else None
        ),
        schema=schema,
    )
    async for event in _outcome_events(contract, outcome):
        yield event
    _persist_history(dialogue, user_input, contract, outcome, system_app)
    yield _sse_event({"type": "final", "content": outcome.final_content})
    yield _sse_event({"type": "done", "status": "done"})


def _resolve_and_compile(
    dialogue: ConversationVo,
    user_input: str,
    resolver: ContractResolver,
) -> tuple[Any, Any, ContractCompilation]:
    contract = None
    source_resolution = None
    try:
        contract, source_resolution = resolver(dialogue, user_input)
        if contract is None:
            raise ValueError("query_contract_missing")
        compilation = compile_data_query_contract(contract)
    except Exception as exc:
        logger.warning(
            "Governed data-query contract resolution failed: %s",
            type(exc).__name__,
            exc_info=exc,
        )
        compilation = ContractCompilation(
            status="invalid",
            gap_kind="contract_invalid",
            reason=f"contract resolution failed with {type(exc).__name__}",
        )
    return contract, source_resolution, compilation


def _load_connector(
    source_resolution: Any, system_app: Any
) -> tuple[Any, str, str, dict[str, Any]]:
    connector = None
    source_name = ""
    source_type = ""
    schema = {"status": "source_unavailable", "tables": [], "error_type": ""}
    if source_resolution is None or source_resolution.status != "selected":
        return connector, source_name, source_type, schema

    source_name = source_resolution.selected or ""
    source_type = source_resolution.selected_type or ""
    try:
        manager = ConnectorManager.get_instance(system_app)
        connector = manager.get_connector(source_name)
        tables = sorted(str(name) for name in connector.get_table_names())
        schema = {"status": "loaded", "tables": tables, "error_type": ""}
    except Exception as exc:
        logger.warning(
            "Governed datasource bootstrap failed for %s: %s",
            source_name,
            type(exc).__name__,
            exc_info=exc,
        )
        connector = None
        schema = {
            "status": "unavailable",
            "tables": [],
            "error_type": type(exc).__name__,
        }
    return connector, source_name, source_type, schema


async def _outcome_events(
    contract: DataQueryContract, outcome: GovernedQueryOutcome
) -> AsyncGenerator[str, None]:
    step_id = "step-1"
    yield _sse_event(
        {
            "type": "step.start",
            "step": 1,
            "id": step_id,
            "title": "execute_compiled_contract",
            "detail": "Deterministic contract compilation and governed SQL execution",
        }
    )
    yield _sse_event(
        {
            "type": "step.meta",
            "id": step_id,
            "thought": None,
            "action_intention": "Execute the versioned data-query contract",
            "action_reason": "The contract has a registered deterministic compiler",
            "action": "execute_compiled_contract",
            "action_input": _contract_action_input(contract),
            "title": "execute_compiled_contract",
        }
    )
    for chunk in outcome.chunks:
        yield _sse_event(
            {
                "type": "step.chunk",
                "id": step_id,
                "output_type": chunk.get("output_type", "text"),
                "content": chunk.get("content"),
            }
        )
    status = "done" if outcome.status == "executed" else "failed"
    yield _sse_event({"type": "step.done", "id": step_id, "status": status})


def _persist_history(
    dialogue: ConversationVo,
    user_input: str,
    contract: DataQueryContract,
    outcome: GovernedQueryOutcome,
    system_app: Any,
) -> None:
    status = "done" if outcome.status == "executed" else "failed"
    history_step = {
        "id": "step-1",
        "title": "execute_compiled_contract",
        "detail": "Deterministic contract compilation and governed SQL execution",
        "thought": None,
        "action": "execute_compiled_contract",
        "action_input": _contract_action_input(contract),
        "outputs": outcome.chunks,
        "status": status,
    }
    try:
        conv_serve = ConversationServe.get_instance(system_app)
        storage_conv = StorageConversation(
            conv_uid=dialogue.conv_uid or str(uuid.uuid4()),
            chat_mode=dialogue.chat_mode or "chat_react_agent",
            user_name=dialogue.user_name,
            sys_code=dialogue.sys_code,
            summary=dialogue.user_input,
            app_code=dialogue.app_code,
            conv_storage=conv_serve.conv_storage,
            message_storage=conv_serve.message_storage,
        )
        storage_conv.save_to_storage()
        storage_conv.start_new_round()
        storage_conv.add_user_message(user_input)
        storage_conv.add_view_message(
            json.dumps(
                {
                    "version": 1,
                    "type": "governed-data-query",
                    "final_content": outcome.final_content,
                    "steps": [history_step],
                    "task_plan": [],
                    "generated_images": [],
                },
                ensure_ascii=False,
            )
        )
        storage_conv.end_current_round()
        storage_conv.save_to_storage()
    except Exception:
        logger.warning("Failed to persist governed query history", exc_info=True)


def _extract_user_question(value: Any) -> str:
    text = str(value or "").strip()
    for marker in ("\n用户问题:\n", "\n用户问题：\n", "\nUser question:\n"):
        if marker in text:
            question = text.rsplit(marker, 1)[1].strip()
            if question:
                return question
    return text


def _contract_action_input(contract: DataQueryContract) -> str:
    return json.dumps(
        {
            "contract_id": contract.contract_id,
            "contract_version": contract.version,
            "logical_group": contract.logical_group,
        },
        ensure_ascii=False,
    )


def _sse_event(payload: dict[str, Any]) -> str:
    return emit_react_agent_event(payload)
