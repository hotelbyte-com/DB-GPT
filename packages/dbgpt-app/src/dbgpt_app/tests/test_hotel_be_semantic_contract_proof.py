"""Proof cases for #21706 business semantics."""

# SQL fixtures intentionally keep the failure predicate readable in one line.
# ruff: noqa: E501

import json

from dbgpt_app.scene.chat_db.query_contract import (
    DataQueryContract,
    validate_result_semantics,
    validate_sql_semantics,
)


def _issue_21706_contract() -> DataQueryContract:
    return DataQueryContract.model_validate(
        {
            "version": "hotelbyte.data-query/v1",
            "logical_group": "hotel-be",
            "required_capabilities": [
                "operational_logs",
                "supplier_reliability",
                "time_series",
            ],
            "semantic": {
                "allowed_tables": [
                    "hb_log",
                ],
                "strict_predicates": True,
                "required_filters": [
                    {
                        "column": "api_in_path",
                        "operator": "eq",
                        "value": "/api/search/hotelRates",
                    },
                    {
                        "column": "api_out_supplier",
                        "operator": "neq",
                        "value": "",
                    },
                    {
                        "column": "api_out_path",
                        "operator": "neq",
                        "value": "",
                    },
                ],
                "required_dimensions": [
                    {"alias": "supplier", "column": "api_out_supplier"}
                ],
                "required_group_by": ["api_out_supplier"],
                "required_order_by": [
                    {"expression": "error_rate_pct", "direction": "desc"},
                    {"expression": "total_requests", "direction": "desc"},
                ],
                "raw_failure_rate": {
                    "rate_alias": "error_rate_pct",
                    "failed_count_alias": "failed_requests",
                    "total_count_alias": "total_requests",
                    "business_error_column": "biz_error_code",
                    "http_status_column": "output_http_status_code",
                    "http_failure_min": 400,
                    "scale": 100,
                    "zero_denominator_policy": "not_evaluable",
                },
                "required_result_columns": [
                    "supplier",
                    "failed_requests",
                    "total_requests",
                    "error_rate_pct",
                ],
                "rolling_window": {
                    "column": "ts",
                    "hours": 24,
                },
                "limit_max": 100,
            },
        }
    )


VALID_SQL = """
SELECT
  api_out_supplier AS supplier,
  SUM(CASE WHEN biz_error_code != '' OR output_http_status_code >= 400 THEN 1 ELSE 0 END) AS failed_requests,
  COUNT(*) AS total_requests,
  SUM(CASE WHEN biz_error_code != '' OR output_http_status_code >= 400 THEN 1 ELSE 0 END)
    * 100.0 / NULLIF(COUNT(*), 0) AS error_rate_pct
FROM hb_log
WHERE api_in_path = '/api/search/hotelRates'
  AND api_out_supplier != ''
  AND api_out_path != ''
  AND ts >= NOW() - INTERVAL 24 HOUR
GROUP BY api_out_supplier
ORDER BY error_rate_pct DESC NULLS LAST, total_requests DESC
LIMIT 100
"""


def test_issue_21706_ast_contract_accepts_the_declared_semantics():
    evidence = validate_sql_semantics(VALID_SQL, _issue_21706_contract())

    assert evidence.valid, evidence.model_dump_json(indent=2)
    assert evidence.tables == ["hb_log"]
    assert evidence.dimension_aliases == ["supplier"]
    assert evidence.metric_aliases == ["error_rate_pct"]
    assert evidence.failure_predicate_verified
    assert evidence.rolling_window_hours == 24
    assert evidence.zero_denominator_policy == "not_evaluable"
    assert evidence.fingerprint


def test_issue_21706_ast_contract_accepts_equivalent_scale_placement():
    equivalent_sql = VALID_SQL.replace(
        "* 100.0 / NULLIF(COUNT(*), 0) AS error_rate_pct",
        "/ NULLIF(COUNT(*), 0) * 100.0 AS error_rate_pct",
    )

    evidence = validate_sql_semantics(equivalent_sql, _issue_21706_contract())

    assert evidence.valid, evidence.model_dump_json(indent=2)
    assert evidence.failure_predicate_verified


def test_issue_21706_ast_contract_rejects_shape_only_false_positive():
    wrong_sql = """
    SELECT
      api_out_supplier AS supplier,
      SUM(CASE WHEN output_http_status_code >= 500 THEN 1 ELSE 0 END) AS failed_requests,
      COUNT(*) AS total_requests,
      AVG(output_http_status_code) AS error_rate_pct
    FROM hblog_ns.hb_log
    WHERE api_in_path = '/api/search/rateCount'
      AND ts >= NOW() - INTERVAL 1 DAY
    GROUP BY api_out_supplier
    ORDER BY total_requests DESC, error_rate_pct ASC
    LIMIT 100
    """

    evidence = validate_sql_semantics(wrong_sql, _issue_21706_contract())

    assert not evidence.valid
    assert (
        "required_filter_missing:api_in_path=/api/search/hotelRates" in evidence.errors
    )
    assert "required_filter_missing:api_out_supplier!=" in evidence.errors
    assert "required_filter_missing:api_out_path!=" in evidence.errors
    assert "raw_failed_count_invalid:failed_requests" in evidence.errors
    assert "raw_failure_rate_invalid:error_rate_pct" in evidence.errors
    assert "order_by_mismatch:error_rate_pct:desc" in evidence.errors
    assert "order_by_mismatch:total_requests:desc" in evidence.errors


def test_issue_21706_ast_contract_rejects_calendar_day_and_zero_percent_fallback():
    wrong_sql = """
    SELECT
      api_out_supplier AS supplier,
      SUM(CASE WHEN biz_error_code != '' OR output_http_status_code >= 400 THEN 1 ELSE 0 END) AS failed_requests,
      COUNT(*) AS total_requests,
      COALESCE(
        SUM(CASE WHEN biz_error_code != '' OR output_http_status_code >= 400 THEN 1 ELSE 0 END)
          * 100.0 / NULLIF(COUNT(*), 0),
        0
      ) AS error_rate_pct
    FROM hblog_ns.hb_log
    WHERE api_in_path = '/api/search/hotelRates'
      AND api_out_supplier != ''
      AND api_out_path != ''
      AND ts >= TODAY()
    GROUP BY api_out_supplier
    ORDER BY error_rate_pct DESC, total_requests DESC
    LIMIT 100
    """

    evidence = validate_sql_semantics(wrong_sql, _issue_21706_contract())

    assert not evidence.valid
    assert "rolling_window_missing:ts:24h" in evidence.errors
    assert "raw_failure_rate_invalid:error_rate_pct" in evidence.errors


def test_issue_21706_ast_contract_requires_the_declared_table():
    evidence = validate_sql_semantics(
        VALID_SQL.replace("FROM hb_log", ""),
        _issue_21706_contract(),
    )

    assert not evidence.valid
    assert "table_required_missing:hb_log" in evidence.errors


def test_issue_21706_ast_contract_rejects_semantically_narrower_extra_filter():
    narrowed_sql = VALID_SQL.replace(
        "AND api_out_path != ''",
        "AND api_out_path != '' AND seller_entity_id = 123",
    )

    evidence = validate_sql_semantics(narrowed_sql, _issue_21706_contract())

    assert not evidence.valid
    assert any(
        error.startswith("unexpected_predicate:seller_entity_id = 123")
        for error in evidence.errors
    )


def test_issue_21706_ast_contract_rejects_self_join_result_distortion():
    joined_sql = VALID_SQL.replace(
        "FROM hb_log",
        "FROM hb_log a JOIN hb_log b ON a.log_id = b.log_id",
    )

    evidence = validate_sql_semantics(joined_sql, _issue_21706_contract())

    assert not evidence.valid
    assert "strict_table_shape_mismatch" in evidence.errors
    assert "unexpected_join" in evidence.errors


def test_issue_21706_ast_contract_rejects_cross_database_qualification():
    evidence = validate_sql_semantics(
        VALID_SQL.replace("FROM hb_log", "FROM another_database.hb_log"),
        _issue_21706_contract(),
    )

    assert not evidence.valid
    assert "qualified_table_forbidden" in evidence.errors


def test_issue_21706_ast_contract_rejects_constant_supplier_dimension():
    evidence = validate_sql_semantics(
        VALID_SQL.replace("api_out_supplier AS supplier", "'all' AS supplier"),
        _issue_21706_contract(),
    )

    assert not evidence.valid
    assert "dimension_projection_invalid:supplier" in evidence.errors


def test_issue_21706_ast_contract_rejects_narrowed_failure_predicate():
    evidence = validate_sql_semantics(
        VALID_SQL.replace(
            "biz_error_code != '' OR output_http_status_code >= 400",
            "(biz_error_code != '' AND api_out_supplier = 'A') "
            "OR output_http_status_code >= 400",
        ),
        _issue_21706_contract(),
    )

    assert not evidence.valid
    assert "raw_failed_count_invalid:failed_requests" in evidence.errors
    assert "raw_failure_rate_invalid:error_rate_pct" in evidence.errors


def test_query_contract_serializes_without_a_physical_datasource_name():
    payload = json.loads(_issue_21706_contract().model_dump_json())

    assert payload["logical_group"] == "hotel-be"
    assert payload["required_capabilities"] == [
        "operational_logs",
        "supplier_reliability",
        "time_series",
    ]
    assert "database_name" not in payload
    assert "datasource_name" not in payload


def test_issue_21706_result_equivalence_accepts_formula_and_sort_order():
    evidence = validate_result_semantics(
        ["supplier", "failed_requests", "total_requests", "error_rate_pct"],
        [
            ["A", 3, 10, 30.0],
            ["B", 2, 10, 20.0],
            ["C", 1, 10, 10.0],
        ],
        _issue_21706_contract(),
    )

    assert evidence.valid, evidence.model_dump_json(indent=2)
    assert evidence.status == "evaluated"
    assert evidence.checked_rows == 3


def test_issue_21706_result_equivalence_rejects_wrong_formula_and_sort_order():
    evidence = validate_result_semantics(
        ["supplier", "failed_requests", "total_requests", "error_rate_pct"],
        [
            ["A", 1, 10, 10.0],
            ["B", 5, 10, 25.0],
        ],
        _issue_21706_contract(),
    )

    assert not evidence.valid
    assert "result_formula_mismatch:row=1:error_rate_pct" in evidence.errors
    assert "result_order_mismatch:row=1" in evidence.errors


def test_issue_21706_result_equivalence_rejects_blank_supplier_dimension():
    evidence = validate_result_semantics(
        ["supplier", "failed_requests", "total_requests", "error_rate_pct"],
        [["", 1, 10, 10.0]],
        _issue_21706_contract(),
    )

    assert not evidence.valid
    assert "result_dimension_invalid:row=0:supplier" in evidence.errors


def test_issue_21706_no_traffic_is_not_evaluable_instead_of_zero_percent():
    no_rows = validate_result_semantics(
        ["supplier", "failed_requests", "total_requests", "error_rate_pct"],
        [],
        _issue_21706_contract(),
    )
    assert no_rows.valid
    assert no_rows.status == "not_evaluable"
    assert no_rows.reason == "no_matching_traffic"

    fabricated_zero = validate_result_semantics(
        ["supplier", "failed_requests", "total_requests", "error_rate_pct"],
        [["NoTrafficSupplier", 0, 0, 0.0]],
        _issue_21706_contract(),
    )
    assert not fabricated_zero.valid
    assert (
        "zero_denominator_must_be_null:row=0:error_rate_pct" in fabricated_zero.errors
    )

    impossible_group = validate_result_semantics(
        ["supplier", "failed_requests", "total_requests", "error_rate_pct"],
        [["NoTrafficSupplier", 0, 0, None]],
        _issue_21706_contract(),
    )
    assert not impossible_group.valid
    assert "grouped_count_zero_invalid" in impossible_group.errors
