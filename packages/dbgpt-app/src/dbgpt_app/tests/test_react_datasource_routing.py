import pytest

from dbgpt_app.openapi.api_v1.agentic_data_api import (
    _react_agent_contract_resolution,
    _react_agent_database_name,
    _react_agent_stream,
)
from dbgpt_app.openapi.api_view_model import ConversationVo
from dbgpt_app.scene.chat_db.datasource_router import DatasourceResolution


def test_react_datasource_routing_prefers_explicit_database_name():
    def resolver(select_param, user_input, system_app):
        raise AssertionError("explicit database_name should not call resolver")

    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info={"database_name": "hoteldev"},
        user_input="count bookings",
    )

    assert (
        _react_agent_database_name(dialogue, "count bookings", resolver=resolver)
        == "hoteldev"
    )


def test_react_datasource_routing_resolves_logical_select_param():
    calls = []

    def resolver(select_param, user_input, system_app, hints):
        calls.append((select_param, user_input, hints))
        return "hoteldev"

    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info={},
        user_input="count bookings",
    )

    assert (
        _react_agent_database_name(dialogue, "count bookings", resolver=resolver)
        == "hoteldev"
    )
    assert calls == [("hotel-be", "count bookings", None)]


def test_react_datasource_routing_falls_back_to_select_param_on_resolver_error():
    def resolver(select_param, user_input, system_app, hints):
        raise RuntimeError("resolver unavailable")

    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info={},
        user_input="count bookings",
    )

    assert (
        _react_agent_database_name(dialogue, "count bookings", resolver=resolver)
        == "hotel-be"
    )


def test_react_datasource_routing_passes_structured_hints_once():
    calls = []

    def resolver(select_param, user_input, system_app, hints):
        calls.append((select_param, user_input, hints))
        return "hblog_ns"

    hints = {"allowed_tables": ["hb_log"]}
    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info=hints,
        user_input="supplier failure rate",
    )

    assert (
        _react_agent_database_name(
            dialogue,
            "supplier failure rate",
            resolver=resolver,
            hints=hints,
        )
        == "hblog_ns"
    )
    assert calls == [("hotel-be", "supplier failure rate", hints)]


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

    def resolver(select_param, user_input, system_app, *, contract):
        calls.append((select_param, user_input, contract.logical_group))
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
    )

    contract, resolution = _react_agent_contract_resolution(
        dialogue, "supplier reliability", resolver=resolver
    )

    assert contract.logical_group == "hotel-be"
    assert resolution.selected == "hblog_ns"
    assert calls == [("hotel-be", "supplier reliability", "hotel-be")]


def test_typed_contract_rejects_caller_physical_datasource_override():
    ext_info = _contract_ext_info()
    ext_info["database_name"] = "hblog_ns"
    dialogue = ConversationVo(
        select_param="hotel-be",
        ext_info=ext_info,
        user_input="supplier reliability",
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
    )

    events = [event async for event in _react_agent_stream(dialogue)]

    assert "Data query gap [contract_invalid]" in "".join(events)
    assert any('"type": "done", "status": "done"' in event for event in events)
