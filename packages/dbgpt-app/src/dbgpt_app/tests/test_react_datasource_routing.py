from dbgpt_app.openapi.api_v1.agentic_data_api import _react_agent_database_name
from dbgpt_app.openapi.api_view_model import ConversationVo


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

    def resolver(select_param, user_input, system_app):
        calls.append((select_param, user_input))
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
    assert calls == [("hotel-be", "count bookings")]


def test_react_datasource_routing_falls_back_to_select_param_on_resolver_error():
    def resolver(select_param, user_input, system_app):
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
