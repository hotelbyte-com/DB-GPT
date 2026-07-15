"""Unit tests for the chat_data datasource router.

These cover the issue-3191 regression: the ``hotel-be`` logical group used to
exclude the TDengine datasource, so every operational-log / time-series
question fell through to MySQL and failed with ``Table 'hotel_user.hb_log'
doesn't exist``. The router must now (a) include TDengine in the group and
(b) honor an ``allowed_tables`` contract hint so a governed query resolves to
the datasource that actually owns the table.
"""

from unittest.mock import patch

from dbgpt_app.scene.chat_db.datasource_router import (
    _choose_candidate,
    _hint_tables,
    _hotel_be_default_candidates,
    resolve_chat_data_source,
)


def _ds(db_name, db_type="mysql", comment=""):
    return {"db_name": db_name, "db_type": db_type, "comment": comment}


class _FakeDao:
    """Stand-in for ConnectConfigDao that returns canned datasource rows."""

    def __init__(self, rows):
        self._rows = rows

    def get_db_list(self, db_name=None, user_id=None):
        if db_name:
            return [r for r in self._rows if r["db_name"] == db_name]
        return list(self._rows)

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
    def metadata_side_effect(candidate, _dao):
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
        "hotel-be", "some question", system_app=object(), hints={"allowed_tables": ["hb_log"]}
    )
    assert got == "hblog_ns"
    # hints is the 5th positional argument to _choose_candidate
    assert mock_choose.call_args.args[4] == {"allowed_tables": ["hb_log"]}
