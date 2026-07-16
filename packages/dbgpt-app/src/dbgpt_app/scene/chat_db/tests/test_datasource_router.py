"""Unit tests for the chat_data datasource router.

These cover the issue-3191 regression: the ``hotel-be`` logical group used to
exclude the TDengine datasource, so every operational-log / time-series
question fell through to MySQL and failed with ``Table 'hotel_user.hb_log'
doesn't exist``. The router must now (a) include TDengine in the group and
(b) honor an ``allowed_tables`` contract hint so a governed query resolves to
the datasource that actually owns the table.
"""

import json
from unittest.mock import patch

import pytest

from dbgpt_app.scene import ChatParam, ChatScene
from dbgpt_app.scene.chat_db.auto_execute.chat import ChatWithDbAutoExecute
from dbgpt_app.scene.chat_db.datasource_router import (
    _choose_candidate,
    _hint_tables,
    _hotel_be_default_candidates,
    resolve_chat_data_source,
)
from dbgpt_app.scene.chat_db.professional_qa.chat import ChatWithDbQA


def _ds(db_name, db_type="mysql", comment="", user_id=None):
    return {
        "db_name": db_name,
        "db_type": db_type,
        "comment": comment,
        "user_id": user_id,
    }


class _FakeDao:
    """Stand-in for ConnectConfigDao that returns canned datasource rows."""

    def __init__(self, rows):
        self._rows = rows

    def get_db_list(self, db_name=None, user_id=None):
        return [
            row
            for row in self._rows
            if (not db_name or row["db_name"] == db_name)
            and (not user_id or not row.get("user_id") or row.get("user_id") == user_id)
        ]

    def get_by_names(self, db_name):
        return [r for r in self._rows if r["db_name"] == db_name]


def test_hotel_be_group_includes_tdengine():
    dao = _FakeDao(
        [
            _ds("hotel_user", "mysql", "hotel user mysql"),
            _ds("hblog_ns", "tdengine", "operational logs"),
        ]
    )
    candidates = _hotel_be_default_candidates(dao)
    assert "hblog_ns" in candidates
    assert "hotel_user" in candidates


def test_hint_tables_accepts_list_and_json_string():
    assert _hint_tables({"allowed_tables": ["hb_log"]}) == ["hb_log"]
    assert _hint_tables({"hotel_allowed_tables": '["hb_log"]'}) == ["hb_log"]
    assert _hint_tables(None) == []
    assert _hint_tables({}) == []


@patch("dbgpt_app.scene.chat_db.datasource_router._datasource_metadata")
def test_allowed_tables_hint_routes_to_tdengine(mock_metadata):
    """The hotelRates failure-rate question never says 'hblog' or 'tdengine',
    but its contract pins allowed_tables=['hb_log']. The hint must push the
    TDengine datasource above MySQL even though MySQL owns the 'hotel' token."""
    dao = _FakeDao(
        [
            _ds("hotel_user", "mysql", "hotel user mysql"),
            _ds("hblog_ns", "tdengine", "operational logs"),
        ]
    )

    # Metadata scoring still runs; make MySQL look textually closer to the
    # question so only the contract hint can flip the winner.
    def metadata_side_effect(candidate, _dao, user_id=None):
        return "hotel user mysql" if candidate == "hotel_user" else "operational logs"

    mock_metadata.side_effect = metadata_side_effect

    def fake_tables(candidate):
        return ["hb_log"] if candidate == "hblog_ns" else ["users", "orders"]

    with patch(
        "dbgpt_serve.datasource.manages.ConnectorManager.get_instance"
    ) as mock_cm:
        instance = mock_cm.return_value
        instance.get_connector.side_effect = lambda c: type(
            "Conn", (), {"get_table_names": lambda self: fake_tables(c)}
        )()

        chosen = _choose_candidate(
            ["hotel_user", "hblog_ns"],
            "Which suppliers had the highest hotelRates error rate last 24h?",
            system_app=object(),
            dao=dao,
            hints={"allowed_tables": ["hb_log"]},
        )
    assert chosen == "hblog_ns"


@patch("dbgpt_app.scene.chat_db.datasource_router._choose_candidate")
@patch("dbgpt_app.scene.chat_db.datasource_router._resolve_candidates")
def test_resolve_passes_hints_through(mock_resolve, mock_choose):
    mock_resolve.return_value = ["hotel_user", "hblog_ns"]
    mock_choose.return_value = "hblog_ns"
    got = resolve_chat_data_source(
        "hotel-be",
        "some question",
        system_app=object(),
        hints={"allowed_tables": ["hb_log"]},
    )
    assert got == "hblog_ns"
    # hints is the 5th positional argument to _choose_candidate
    assert mock_choose.call_args.args[4] == {"allowed_tables": ["hb_log"]}


@pytest.mark.parametrize(
    "select_param",
    ["bob_tdengine", "bob_tdengine,bob_mysql"],
)
def test_legacy_router_rejects_physical_and_inline_cross_user_access(select_param):
    dao = _FakeDao(
        [
            _ds("bob_tdengine", "tdengine", user_id="bob"),
            _ds("bob_mysql", "mysql", user_id="bob"),
        ]
    )
    with (
        patch(
            "dbgpt_app.scene.chat_db.datasource_router.ConnectConfigDao",
            return_value=dao,
        ),
        patch(
            "dbgpt_serve.datasource.manages.ConnectorManager.get_instance"
        ) as connector_manager,
        pytest.raises(PermissionError, match="insufficient_scope"),
    ):
        resolve_chat_data_source(
            select_param,
            "supplier failure rate",
            system_app=object(),
            hints={"allowed_tables": ["hb_log"]},
            user_id="alice",
        )
    connector_manager.assert_not_called()


def test_legacy_router_keeps_shared_datasource_accessible():
    dao = _FakeDao([_ds("shared_tdengine", "tdengine")])
    with patch(
        "dbgpt_app.scene.chat_db.datasource_router.ConnectConfigDao",
        return_value=dao,
    ):
        selected = resolve_chat_data_source(
            "shared_tdengine",
            "supplier failure rate",
            system_app=object(),
            user_id="alice",
        )
    assert selected == "shared_tdengine"


def test_configured_group_filters_candidates_by_authenticated_user(monkeypatch):
    monkeypatch.setenv(
        "DBGPT_CHAT_DATA_GROUPS",
        json.dumps({"private-group": ["bob_tdengine", "shared_tdengine"]}),
    )
    dao = _FakeDao(
        [
            _ds("bob_tdengine", "tdengine", user_id="bob"),
            _ds("shared_tdengine", "tdengine"),
        ]
    )
    with patch(
        "dbgpt_app.scene.chat_db.datasource_router.ConnectConfigDao",
        return_value=dao,
    ):
        selected = resolve_chat_data_source(
            "private-group",
            "supplier failure rate",
            system_app=object(),
            user_id="alice",
        )
    assert selected == "shared_tdengine"


def test_legacy_hint_only_scores_the_users_authorized_tdengine():
    dao = _FakeDao(
        [
            _ds("hotel_user", "mysql"),
            _ds("alice_tdengine", "tdengine", user_id="alice"),
            _ds("bob_tdengine", "tdengine", user_id="bob"),
        ]
    )

    def fake_tables(candidate):
        return ["hb_log"] if candidate.endswith("tdengine") else ["users"]

    with (
        patch(
            "dbgpt_app.scene.chat_db.datasource_router.ConnectConfigDao",
            return_value=dao,
        ),
        patch(
            "dbgpt_serve.datasource.manages.ConnectorManager.get_instance"
        ) as mock_cm,
    ):
        instance = mock_cm.return_value
        instance.get_connector.side_effect = lambda candidate: type(
            "Conn", (), {"get_table_names": lambda self: fake_tables(candidate)}
        )()
        selected = resolve_chat_data_source(
            "hotel-be",
            "supplier failure rate",
            system_app=object(),
            hints={"allowed_tables": ["hb_log"]},
            user_id="alice",
        )

    assert selected == "alice_tdengine"
    assert all(
        call.args[0] != "bob_tdengine" for call in instance.get_connector.call_args_list
    )


@pytest.mark.parametrize(
    ("chat_class", "chat_scene", "resolver_path"),
    [
        (
            ChatWithDbAutoExecute,
            ChatScene.ChatWithDbExecute,
            "dbgpt_app.scene.chat_db.auto_execute.chat.resolve_chat_data_source",
        ),
        (
            ChatWithDbQA,
            ChatScene.ChatWithDbQA,
            "dbgpt_app.scene.chat_db.professional_qa.chat.resolve_chat_data_source",
        ),
    ],
)
def test_authenticated_chat_data_scenes_forward_user_identity_before_connector(
    monkeypatch, chat_class, chat_scene, resolver_path
):
    calls = []

    def reject_cross_user(select_param, user_input, system_app, hints, user_id=None):
        calls.append((select_param, user_input, hints, user_id))
        raise PermissionError("insufficient_scope")

    monkeypatch.setattr(resolver_path, reject_cross_user)
    chat_param = ChatParam(
        chat_session_id="acl-proof",
        current_user_input="supplier failure rate",
        model_name="test-model",
        select_param="bob_tdengine",
        chat_mode=chat_scene,
        user_name="alice",
        ext_info={"allowed_tables": ["hb_log"]},
    )

    with (
        patch(
            "dbgpt_serve.datasource.manages.ConnectorManager.get_instance"
        ) as connector_manager,
        pytest.raises(PermissionError, match="insufficient_scope"),
    ):
        chat_class(chat_param, object())

    assert calls == [
        (
            "bob_tdengine",
            "supplier failure rate",
            {"allowed_tables": ["hb_log"]},
            "alice",
        )
    ]
    connector_manager.assert_not_called()
