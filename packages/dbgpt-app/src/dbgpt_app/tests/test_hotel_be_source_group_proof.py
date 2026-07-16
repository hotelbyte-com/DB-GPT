import json
from unittest.mock import MagicMock

from dbgpt_app.scene.chat_db.datasource_router import (
    resolve_chat_data_source_with_evidence,
)
from dbgpt_app.scene.chat_db.query_contract import DataQueryContract


def _contract(*capabilities: str, tables=None, columns=None) -> DataQueryContract:
    semantic = {"allowed_tables": tables or []}
    if columns:
        semantic["required_filters"] = [
            {"column": column, "operator": "neq", "value": ""} for column in columns
        ]
    return DataQueryContract.model_validate(
        {
            "version": "hotelbyte.data-query/v1",
            "logical_group": "hotel-be",
            "required_capabilities": list(capabilities),
            "semantic": semantic,
        }
    )


def _dao() -> MagicMock:
    dao = MagicMock()
    existing = {
        "hotel": {"db_name": "hotel", "db_type": "mysql", "comment": "catalog"},
        "hblog_ns": {
            "db_name": "hblog_ns",
            "db_type": "tdengine",
            "comment": "operational telemetry",
        },
    }
    dao.get_by_names.side_effect = (
        lambda name: [existing[name]] if name in existing else []
    )
    dao.get_db_list.side_effect = lambda db_name=None: (
        [existing[db_name]] if db_name in existing else list(existing.values())
    )
    return dao


def _system_app(tdengine_tables=None) -> MagicMock:
    mysql = MagicMock()
    mysql.get_table_names.return_value = ["hotel_names", "hotel_catalog"]
    mysql.get_columns.return_value = [{"name": "hotel_id"}]
    tdengine = MagicMock()
    tdengine.get_table_names.return_value = (
        ["hb_log", "hb_log_supplier_reliability_daily"]
        if tdengine_tables is None
        else tdengine_tables
    )
    tdengine.get_columns.return_value = [
        {"name": "ts"},
        {"name": "api_in_path"},
        {"name": "api_out_supplier"},
        {"name": "api_out_path"},
        {"name": "biz_error_code"},
        {"name": "output_http_status_code"},
    ]

    manager = MagicMock()
    manager.get_connector.side_effect = lambda name: {
        "hotel": mysql,
        "hblog_ns": tdengine,
    }[name]
    return MagicMock(), manager


def test_capability_contract_selects_tdengine_without_question_keyword_routing(
    monkeypatch,
):
    monkeypatch.setenv(
        "DBGPT_CHAT_DATA_GROUPS",
        json.dumps(
            {
                "hotel-be": {
                    "candidates": [
                        {
                            "name": "hotel",
                            "capabilities": ["hotel_catalog"],
                            "priority": 10,
                        },
                        {
                            "name": "hblog_ns",
                            "capabilities": [
                                "operational_logs",
                                "supplier_reliability",
                                "time_series",
                            ],
                            "priority": 20,
                        },
                    ]
                }
            }
        ),
    )

    system_app, connector_manager = _system_app()
    resolution = resolve_chat_data_source_with_evidence(
        "hotel-be",
        "Which hotelRates suppliers have the highest error rate?",
        system_app,
        contract=_contract(
            "supplier_reliability",
            "time_series",
            tables=["hb_log"],
            columns=[
                "api_in_path",
                "api_out_supplier",
                "api_out_path",
                "biz_error_code",
                "output_http_status_code",
                "ts",
            ],
        ),
        dao=_dao(),
        connector_manager=connector_manager,
    )

    assert resolution.status == "selected"
    assert resolution.selected == "hblog_ns"
    assert resolution.selected_type == "tdengine"
    assert resolution.required_capabilities == ["supplier_reliability", "time_series"]
    assert resolution.required_columns == [
        "api_in_path",
        "api_out_supplier",
        "api_out_path",
        "biz_error_code",
        "output_http_status_code",
        "ts",
    ]
    assert resolution.candidates[0].name == "hblog_ns"
    assert resolution.candidates[0].healthy


def test_missing_capability_returns_typed_gap_instead_of_mysql_fallback(monkeypatch):
    monkeypatch.setenv(
        "DBGPT_CHAT_DATA_GROUPS",
        json.dumps(
            {
                "hotel-be": {
                    "candidates": [
                        {
                            "name": "hotel",
                            "capabilities": ["hotel_catalog"],
                            "priority": 10,
                        }
                    ]
                }
            }
        ),
    )

    system_app, connector_manager = _system_app()
    resolution = resolve_chat_data_source_with_evidence(
        "hotel-be",
        "Which hotelRates suppliers have the highest error rate?",
        system_app,
        contract=_contract("supplier_reliability", "time_series"),
        dao=_dao(),
        connector_manager=connector_manager,
    )

    assert resolution.status == "capability_unavailable"
    assert resolution.selected is None
    assert resolution.gap_kind == "source_unavailable"
    assert resolution.candidates[0].name == "hotel"
    assert resolution.candidates[0].missing_capabilities == [
        "supplier_reliability",
        "time_series",
    ]


def test_empty_schema_is_not_reported_as_a_healthy_source(monkeypatch):
    monkeypatch.setenv(
        "DBGPT_CHAT_DATA_GROUPS",
        json.dumps(
            {
                "hotel-be": {
                    "candidates": [
                        {
                            "name": "hblog_ns",
                            "capabilities": ["supplier_reliability", "time_series"],
                            "priority": 10,
                        }
                    ]
                }
            }
        ),
    )

    system_app, connector_manager = _system_app(tdengine_tables=[])
    resolution = resolve_chat_data_source_with_evidence(
        "hotel-be",
        "supplier reliability",
        system_app,
        contract=_contract("supplier_reliability", "time_series"),
        dao=_dao(),
        connector_manager=connector_manager,
    )

    assert resolution.status == "schema_unavailable"
    assert resolution.selected is None
    assert resolution.gap_kind == "source_unavailable"
    assert not resolution.candidates[0].healthy
    assert resolution.candidates[0].error == "schema_empty"


def test_missing_required_table_is_a_schema_gap(monkeypatch):
    monkeypatch.setenv(
        "DBGPT_CHAT_DATA_GROUPS",
        json.dumps(
            {
                "hotel-be": {
                    "candidates": [
                        {
                            "name": "hblog_ns",
                            "capabilities": ["supplier_reliability", "time_series"],
                            "priority": 10,
                        }
                    ]
                }
            }
        ),
    )
    system_app, connector_manager = _system_app(tdengine_tables=["hb_log_hourly"])

    resolution = resolve_chat_data_source_with_evidence(
        "hotel-be",
        "supplier reliability",
        system_app,
        contract=_contract(
            "supplier_reliability", "time_series", tables=["hblog_ns.hb_log"]
        ),
        dao=_dao(),
        connector_manager=connector_manager,
    )

    assert resolution.status == "schema_unavailable"
    assert resolution.candidates[0].error == "required_tables_missing"
    assert resolution.candidates[0].missing_tables == ["hb_log"]


def test_missing_required_column_is_a_schema_gap(monkeypatch):
    monkeypatch.setenv(
        "DBGPT_CHAT_DATA_GROUPS",
        json.dumps(
            {
                "hotel-be": {
                    "candidates": [
                        {
                            "name": "hblog_ns",
                            "capabilities": ["supplier_reliability", "time_series"],
                            "priority": 10,
                        }
                    ]
                }
            }
        ),
    )
    system_app, connector_manager = _system_app()
    connector_manager.get_connector("hblog_ns").get_columns.return_value = [
        {"name": "ts"},
        {"name": "api_out_supplier"},
    ]

    resolution = resolve_chat_data_source_with_evidence(
        "hotel-be",
        "supplier reliability",
        system_app,
        contract=_contract(
            "supplier_reliability",
            "time_series",
            tables=["hb_log"],
            columns=["api_in_path", "api_out_supplier", "api_out_path"],
        ),
        dao=_dao(),
        connector_manager=connector_manager,
    )

    assert resolution.status == "schema_unavailable"
    assert resolution.candidates[0].error == "required_columns_missing"
    assert resolution.candidates[0].missing_columns == ["api_in_path", "api_out_path"]
