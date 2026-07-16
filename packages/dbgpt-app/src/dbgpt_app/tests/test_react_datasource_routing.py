import json
from unittest.mock import MagicMock, patch

import pytest

from dbgpt_app.openapi.api_v1.agentic_data_api import (
    _general_react_agent_stream,
    _react_agent_contract_resolution,
    _react_agent_database_name,
    _react_agent_stream,
)
from dbgpt_app.openapi.api_view_model import ConversationVo
from dbgpt_app.scene.chat_db.datasource_router import DatasourceResolution


def test_react_datasource_routing_prefers_explicit_database_name():
    calls = []

    def resolver(select_param, user_input, system_app, hints, user_id):
        calls.append((select_param, user_id))
        return select_param

    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info={"database_name": "hoteldev"},
        user_input="count bookings",
        user_name="alice",
    )

    assert (
        _react_agent_database_name(
            dialogue,
            "count bookings",
            resolver=resolver,
            user_id="alice",
        )
        == "hoteldev"
    )
    assert calls == [("hoteldev", "alice")]


def test_react_datasource_routing_resolves_logical_select_param():
    calls = []

    def resolver(select_param, user_input, system_app, hints, user_id):
        calls.append((select_param, user_input, hints, user_id))
        return "hoteldev"

    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info={},
        user_input="count bookings",
        user_name="alice",
    )

    assert (
        _react_agent_database_name(
            dialogue,
            "count bookings",
            resolver=resolver,
            user_id="alice",
        )
        == "hoteldev"
    )
    assert calls == [("hotel-be", "count bookings", None, "alice")]


def test_react_datasource_routing_fails_closed_on_resolver_error():
    def resolver(select_param, user_input, system_app, hints, user_id):
        raise RuntimeError("resolver unavailable")

    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info={},
        user_input="count bookings",
        user_name="alice",
    )

    with pytest.raises(RuntimeError, match="datasource_resolution_failed"):
        _react_agent_database_name(
            dialogue,
            "count bookings",
            resolver=resolver,
            user_id="alice",
        )


def test_react_datasource_routing_passes_structured_hints_once():
    calls = []

    def resolver(select_param, user_input, system_app, hints, user_id):
        calls.append((select_param, user_input, hints, user_id))
        return "hblog_ns"

    hints = {"allowed_tables": ["hb_log"]}
    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info=hints,
        user_input="supplier failure rate",
        user_name="alice",
    )

    assert (
        _react_agent_database_name(
            dialogue,
            "supplier failure rate",
            resolver=resolver,
            hints=hints,
            user_id="alice",
        )
        == "hblog_ns"
    )
    assert calls == [("hotel-be", "supplier failure rate", hints, "alice")]


def test_react_datasource_routing_rejects_unauthorized_explicit_database():
    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info={"database_name": "bob_tdengine"},
        user_input="supplier failure rate",
        user_name="alice",
    )
    dao = MagicMock()
    dao.get_db_list.return_value = []

    with (
        patch(
            "dbgpt_app.scene.chat_db.datasource_router.ConnectConfigDao",
            return_value=dao,
        ),
        pytest.raises(PermissionError, match="insufficient_scope"),
    ):
        _react_agent_database_name(
            dialogue,
            "supplier failure rate",
            user_id="alice",
        )


@pytest.mark.asyncio
async def test_unauthorized_legacy_source_returns_typed_gap_before_connector_use():
    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info={"database_name": "bob_tdengine"},
        user_input="supplier failure rate",
        user_name="alice",
    )
    dao = MagicMock()
    dao.get_db_list.return_value = []

    with (
        patch(
            "dbgpt_app.scene.chat_db.datasource_router.ConnectConfigDao",
            return_value=dao,
        ),
        patch(
            "dbgpt_serve.datasource.manages.ConnectorManager.get_instance"
        ) as connector_manager,
    ):
        events = [
            json.loads(event.removeprefix("data: "))
            async for event in _general_react_agent_stream(dialogue)
        ]

    assert [event["type"] for event in events] == [
        "step.start",
        "step.chunk",
        "step.done",
        "done",
    ]
    assert "insufficient_scope" in events[1]["content"]
    assert events[-1]["status"] == "failed"
    assert all(event["status"] != "success" for event in events)
    connector_manager.assert_not_called()


@pytest.mark.asyncio
async def test_router_failure_returns_source_gap_before_llm_or_connector_use():
    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info={},
        user_input="supplier failure rate",
        user_name="alice",
    )

    with (
        patch(
            "dbgpt_app.scene.chat_db.datasource_router.resolve_chat_data_source",
            side_effect=RuntimeError("metadata unavailable"),
        ),
        patch(
            "dbgpt_serve.datasource.manages.ConnectorManager.get_instance"
        ) as connector_manager,
    ):
        events = [
            json.loads(event.removeprefix("data: "))
            async for event in _general_react_agent_stream(dialogue)
        ]

    assert [event["type"] for event in events] == [
        "step.start",
        "step.chunk",
        "step.done",
        "done",
    ]
    assert "source_unavailable" in events[1]["content"]
    assert events[-1]["status"] == "failed"
    assert all(event["status"] != "success" for event in events)
    connector_manager.assert_not_called()


def _contract_ext_info():
    return {
        "source": "hotel-be-data-agent",
        "query_contract": {
            "version": "hotelbyte.data-query/v1",
            "logical_group": "hotel-be",
            "required_capabilities": ["supplier_reliability"],
        },
    }


def test_typed_contract_resolution_keeps_physical_source_inside_dbgpt():
    calls = []

    def resolver(select_param, user_input, system_app, *, contract, user_id):
        calls.append((select_param, user_input, contract.logical_group, user_id))
        return DatasourceResolution(
            logical_group="hotel-be",
            status="selected",
            selected="hblog_ns",
            selected_type="tdengine",
            required_capabilities=["supplier_reliability"],
        )

    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info=_contract_ext_info(),
        user_input="supplier reliability",
        user_name="alice",
    )

    contract, resolution = _react_agent_contract_resolution(
        dialogue, "supplier reliability", resolver=resolver
    )

    assert contract.logical_group == "hotel-be"
    assert resolution.selected == "hblog_ns"
    assert calls == [("hotel-be", "supplier reliability", "hotel-be", "alice")]


def test_typed_contract_rejects_caller_physical_datasource_override():
    ext_info = _contract_ext_info()
    ext_info["database_name"] = "hblog_ns"
    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info=ext_info,
        user_input="supplier reliability",
        user_name="alice",
    )

    with pytest.raises(ValueError, match="physical_datasource_override_forbidden"):
        _react_agent_contract_resolution(dialogue, "supplier reliability")


@pytest.mark.parametrize("source", [None, "other-agent"])
def test_typed_contract_rejects_missing_or_wrong_source(source):
    ext_info = _contract_ext_info()
    if source is None:
        ext_info.pop("source")
    else:
        ext_info["source"] = source
    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info=ext_info,
        user_input="supplier reliability",
        user_name="alice",
    )

    with pytest.raises(ValueError, match="query_contract_source_invalid"):
        _react_agent_contract_resolution(dialogue, "supplier reliability")


@pytest.mark.asyncio
async def test_declared_contract_stream_never_enters_general_llm_loop(monkeypatch):
    calls = []

    async def governed(dialogue, **kwargs):
        calls.append("governed")
        yield "data: governed\n\n"

    async def general(dialogue):
        raise AssertionError("typed contract must not enter the LLM loop")
        yield  # pragma: no cover

    monkeypatch.setattr(
        "dbgpt_app.scene.chat_db.governed_query_stream.stream_governed_query_contract",
        governed,
    )
    monkeypatch.setattr(
        "dbgpt_app.openapi.api_v1.agentic_data_api._general_react_agent_stream",
        general,
    )
    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info=_contract_ext_info(),
        user_input="supplier reliability",
        user_name="alice",
    )

    events = [event async for event in _react_agent_stream(dialogue)]

    assert events == ["data: governed\n\n"]
    assert calls == ["governed"]


@pytest.mark.parametrize("source", [None, "other-agent"])
@pytest.mark.asyncio
async def test_declared_contract_with_invalid_source_returns_typed_gap(
    monkeypatch, source
):
    async def general(dialogue):
        raise AssertionError("declared contract must not enter the general LLM loop")
        yield  # pragma: no cover

    monkeypatch.setattr(
        "dbgpt_app.openapi.api_v1.agentic_data_api._general_react_agent_stream",
        general,
    )
    ext_info = _contract_ext_info()
    if source is None:
        ext_info.pop("source")
    else:
        ext_info["source"] = source
    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info=ext_info,
        user_input="supplier reliability",
        user_name="alice",
    )

    events = [
        json.loads(event.removeprefix("data: "))
        async for event in _react_agent_stream(dialogue)
    ]

    assert "Data query gap [contract_invalid]" in events[2]["content"]
    assert all(event["type"] != "final" for event in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["status"] == "failed"
    assert all(event["status"] != "success" for event in events)
