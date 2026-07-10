"""Typed HotelByte data-query contract and SQL semantic evidence.

The contract deliberately carries a logical source group and business semantics,
never a physical datasource name. DB-GPT remains responsible for resolving the
physical datasource and attaching execution evidence.
"""

import hashlib
import json
import math
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from dbgpt._private.pydantic import BaseModel, Field

DATA_QUERY_CONTRACT_VERSION = "hotelbyte.data-query/v1"


class FilterRequirement(BaseModel):
    column: str
    operator: Literal["eq", "neq"] = "eq"
    value: str


class OrderRequirement(BaseModel):
    expression: str
    direction: Literal["asc", "desc"] = "asc"


class DimensionRequirement(BaseModel):
    """Bind a business result alias to one source column."""

    alias: str
    column: str


class RatioMetricRequirement(BaseModel):
    alias: str
    numerator_column: str
    denominator_column: str
    scale: float = 100.0


class RollingWindowRequirement(BaseModel):
    column: str
    hours: int


class RawFailureRateRequirement(BaseModel):
    """Raw-log failure rate aligned with HotelByte Sessions semantics."""

    rate_alias: str
    failed_count_alias: str
    total_count_alias: str
    business_error_column: str
    http_status_column: str
    http_failure_min: int = 400
    scale: float = 100.0
    zero_denominator_policy: Literal["not_evaluable"] = "not_evaluable"


class SQLSemanticContract(BaseModel):
    allowed_tables: List[str] = Field(default_factory=list)
    required_filters: List[FilterRequirement] = Field(default_factory=list)
    strict_predicates: bool = False
    required_dimensions: List[DimensionRequirement] = Field(default_factory=list)
    required_group_by: List[str] = Field(default_factory=list)
    required_order_by: List[OrderRequirement] = Field(default_factory=list)
    ratio_metrics: List[RatioMetricRequirement] = Field(default_factory=list)
    raw_failure_rate: Optional[RawFailureRateRequirement] = None
    required_result_columns: List[str] = Field(default_factory=list)
    rolling_window: Optional[RollingWindowRequirement] = None
    time_column: Optional[str] = None
    require_time_window: bool = False
    limit_max: int = 100


class DataQueryContract(BaseModel):
    version: Literal[DATA_QUERY_CONTRACT_VERSION]
    logical_group: str
    required_capabilities: List[str] = Field(default_factory=list)
    semantic: SQLSemanticContract = Field(default_factory=SQLSemanticContract)


class SQLSemanticEvidence(BaseModel):
    valid: bool = False
    errors: List[str] = Field(default_factory=list)
    tables: List[str] = Field(default_factory=list)
    result_columns: List[str] = Field(default_factory=list)
    dimension_aliases: List[str] = Field(default_factory=list)
    metric_aliases: List[str] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    order_by: List[str] = Field(default_factory=list)
    filters: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)
    failure_predicate_verified: bool = False
    rolling_window_hours: Optional[int] = None
    zero_denominator_policy: str = ""
    fingerprint: str = ""


class ResultSemanticEvidence(BaseModel):
    valid: bool = False
    errors: List[str] = Field(default_factory=list)
    row_count: int = 0
    checked_rows: int = 0
    status: Literal["evaluated", "not_evaluable", "invalid"] = "invalid"
    reason: str = ""


def parse_data_query_contract(
    ext_info: Optional[Dict[str, Any]],
) -> Optional[DataQueryContract]:
    """Parse the versioned contract from ReAct ``ext_info``.

    Invalid declared contracts raise instead of silently degrading to an
    ungoverned request. Absence remains valid for non-HotelByte callers.
    """

    if not isinstance(ext_info, dict) or "query_contract" not in ext_info:
        return None
    raw = ext_info.get("query_contract")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError("query_contract must be a JSON object")
    return DataQueryContract.model_validate(raw)


def validate_sql_semantics(
    sql: str, contract: DataQueryContract
) -> SQLSemanticEvidence:
    """Validate generated SQL against the typed business contract.

    SQLGlot supplies the AST. This validator intentionally fails closed on
    parser errors or unsupported structures; prompt text is never accepted as
    evidence that a semantic requirement was satisfied.
    """

    evidence = SQLSemanticEvidence()
    try:
        statement = parse_one(str(sql or "").strip())
    except (ParseError, ValueError) as exc:
        evidence.errors.append(f"sql_parse_failed:{type(exc).__name__}")
        return evidence

    if not isinstance(statement, exp.Select):
        evidence.errors.append("select_required")
        return evidence

    normalized_sql = statement.sql(pretty=False, normalize=True)
    evidence.fingerprint = hashlib.sha256(normalized_sql.encode("utf-8")).hexdigest()
    evidence.tables = sorted(
        {_normalized_name(table.name) for table in statement.find_all(exp.Table)}
    )

    projections = list(statement.expressions)
    projection_by_alias: Dict[str, exp.Expression] = {}
    for projection in projections:
        alias = _normalized_name(projection.alias_or_name)
        if alias:
            projection_by_alias[alias] = projection
            evidence.result_columns.append(alias)

    semantic = contract.semantic
    allowed_tables = {_normalized_name(name) for name in semantic.allowed_tables}
    for table in sorted(allowed_tables):
        if table not in evidence.tables:
            evidence.errors.append(f"table_required_missing:{table}")
    for table in evidence.tables:
        if allowed_tables and table not in allowed_tables:
            evidence.errors.append(f"table_not_allowed:{table}")

    actual_columns = set(evidence.result_columns)
    for required in semantic.required_result_columns:
        normalized = _normalized_name(required)
        if normalized not in actual_columns:
            evidence.errors.append(f"result_column_missing:{normalized}")

    for dimension in semantic.required_dimensions:
        alias = _normalized_name(dimension.alias)
        projection = projection_by_alias.get(alias)
        if projection is None or not _matches_dimension_projection(
            projection, dimension
        ):
            evidence.errors.append(f"dimension_projection_invalid:{alias}")
        else:
            evidence.dimension_aliases.append(alias)

    evidence.filters = _literal_filters(statement)
    for required_filter in semantic.required_filters:
        column = _normalized_name(required_filter.column)
        values = evidence.filters.get(column, {}).get(required_filter.operator, [])
        if required_filter.value not in values:
            operator = "=" if required_filter.operator == "eq" else "!="
            evidence.errors.append(
                f"required_filter_missing:{column}{operator}{required_filter.value}"
            )
    if semantic.strict_predicates:
        evidence.errors.extend(_strict_predicate_errors(statement, semantic))

    group = statement.args.get("group")
    if group:
        evidence.group_by = [_expression_name(item) for item in group.expressions]
    for required_group in semantic.required_group_by:
        normalized = _normalized_name(required_group)
        if normalized not in evidence.group_by:
            evidence.errors.append(f"group_by_missing:{normalized}")

    order = statement.args.get("order")
    ordered = list(order.expressions) if order else []
    evidence.order_by = [
        f"{_expression_name(item.this)}:{'desc' if item.args.get('desc') else 'asc'}"
        for item in ordered
    ]
    for index, requirement in enumerate(semantic.required_order_by):
        expected = f"{_normalized_name(requirement.expression)}:{requirement.direction}"
        if index >= len(evidence.order_by) or evidence.order_by[index] != expected:
            evidence.errors.append(f"order_by_mismatch:{expected}")
    if semantic.strict_predicates:
        evidence.errors.extend(_strict_query_shape_errors(statement, semantic))

    for metric in semantic.ratio_metrics:
        alias = _normalized_name(metric.alias)
        projection = projection_by_alias.get(alias)
        if projection is None or not _matches_ratio_metric(projection, metric):
            evidence.errors.append(f"ratio_metric_invalid:{alias}")
        else:
            evidence.metric_aliases.append(alias)

    if semantic.raw_failure_rate is not None:
        failure_requirement = semantic.raw_failure_rate
        failed_alias = _normalized_name(failure_requirement.failed_count_alias)
        failed_projection = projection_by_alias.get(failed_alias)
        if failed_projection is None or not _matches_raw_failed_count(
            failed_projection, failure_requirement
        ):
            evidence.errors.append(f"raw_failed_count_invalid:{failed_alias}")

        total_alias = _normalized_name(failure_requirement.total_count_alias)
        total_projection = projection_by_alias.get(total_alias)
        if total_projection is None or not _is_count_all(total_projection):
            evidence.errors.append(f"raw_total_count_invalid:{total_alias}")

        rate_alias = _normalized_name(failure_requirement.rate_alias)
        rate_projection = projection_by_alias.get(rate_alias)
        if rate_projection is None or not _matches_raw_failure_rate(
            rate_projection, failure_requirement
        ):
            evidence.errors.append(f"raw_failure_rate_invalid:{rate_alias}")
        else:
            evidence.failure_predicate_verified = True
            evidence.metric_aliases.append(rate_alias)
            evidence.zero_denominator_policy = (
                failure_requirement.zero_denominator_policy
            )

    if semantic.rolling_window is not None:
        rolling = semantic.rolling_window
        if _has_rolling_window(statement, rolling):
            evidence.rolling_window_hours = rolling.hours
        else:
            evidence.errors.append(
                f"rolling_window_missing:{_normalized_name(rolling.column)}:"
                f"{rolling.hours}h"
            )

    if semantic.require_time_window:
        where = statement.args.get("where")
        time_column = _normalized_name(semantic.time_column)
        where_columns = (
            {_normalized_name(column.name) for column in where.find_all(exp.Column)}
            if where is not None
            else set()
        )
        if not time_column or time_column not in where_columns:
            evidence.errors.append(f"time_window_missing:{time_column or 'unknown'}")

    limit = statement.args.get("limit")
    limit_value = _literal_number(limit.expression) if limit is not None else None
    if limit_value is None:
        evidence.errors.append("limit_missing")
    elif limit_value > semantic.limit_max:
        evidence.errors.append(
            f"limit_exceeded:{int(limit_value)}>{semantic.limit_max}"
        )

    evidence.valid = not evidence.errors
    return evidence


def validate_result_semantics(
    columns: Sequence[str],
    rows: Sequence[Any],
    contract: DataQueryContract,
) -> ResultSemanticEvidence:
    """Validate executed rows against the declared metric and ordering contract."""

    evidence = ResultSemanticEvidence(row_count=len(rows))
    normalized_columns = [_normalized_name(column) for column in columns]
    for required in contract.semantic.required_result_columns:
        normalized = _normalized_name(required)
        if normalized not in normalized_columns:
            evidence.errors.append(f"result_column_missing:{normalized}")

    normalized_rows = [_normalize_result_row(normalized_columns, row) for row in rows]
    evidence.checked_rows = len(normalized_rows)

    for index, row in enumerate(normalized_rows):
        for dimension in contract.semantic.required_dimensions:
            alias = _normalized_name(dimension.alias)
            value = row.get(alias)
            if value is None or (isinstance(value, str) and not value.strip()):
                evidence.errors.append(f"result_dimension_invalid:row={index}:{alias}")

    requirement = contract.semantic.raw_failure_rate
    positive_traffic = False
    if requirement is not None:
        failed_alias = _normalized_name(requirement.failed_count_alias)
        total_alias = _normalized_name(requirement.total_count_alias)
        rate_alias = _normalized_name(requirement.rate_alias)
        for index, row in enumerate(normalized_rows):
            failed = _numeric_value(row.get(failed_alias))
            total = _numeric_value(row.get(total_alias))
            rate = _numeric_value(row.get(rate_alias))
            if failed is None or total is None or failed < 0 or total < 0:
                evidence.errors.append(f"result_count_invalid:row={index}")
                continue
            if failed > total:
                evidence.errors.append(f"failed_exceeds_total:row={index}")
            if total == 0:
                if row.get(rate_alias) is not None:
                    evidence.errors.append(
                        f"zero_denominator_must_be_null:row={index}:{rate_alias}"
                    )
                continue
            positive_traffic = True
            expected = failed * requirement.scale / total
            if rate is None or not math.isclose(
                rate, expected, rel_tol=1e-7, abs_tol=1e-7
            ):
                evidence.errors.append(
                    f"result_formula_mismatch:row={index}:{rate_alias}"
                )

    for index in range(1, len(normalized_rows)):
        if not _rows_in_declared_order(
            normalized_rows[index - 1],
            normalized_rows[index],
            contract.semantic.required_order_by,
        ):
            evidence.errors.append(f"result_order_mismatch:row={index}")

    if (
        requirement is not None
        and normalized_rows
        and not positive_traffic
        and contract.semantic.required_group_by
    ):
        evidence.errors.append("grouped_count_zero_invalid")

    if evidence.errors:
        evidence.status = "invalid"
    elif not normalized_rows:
        evidence.status = "not_evaluable"
        evidence.reason = "no_matching_traffic"
    elif requirement is not None and not positive_traffic:
        evidence.status = "not_evaluable"
        evidence.reason = "zero_traffic_samples"
    else:
        evidence.status = "evaluated"
    evidence.valid = not evidence.errors
    return evidence


def _normalize_result_row(columns: Sequence[str], row: Any) -> Dict[str, Any]:
    mapping: Optional[Mapping[Any, Any]] = None
    if isinstance(row, Mapping):
        mapping = row
    elif hasattr(row, "_mapping"):
        mapping = row._mapping
    if mapping is not None:
        return {_normalized_name(key): value for key, value in mapping.items()}
    values = list(row) if isinstance(row, (list, tuple)) else [row]
    return {
        column: values[index] if index < len(values) else None
        for index, column in enumerate(columns)
    }


def _numeric_value(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rows_in_declared_order(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    requirements: Sequence[OrderRequirement],
) -> bool:
    for requirement in requirements:
        key = _normalized_name(requirement.expression)
        previous_value = previous.get(key)
        current_value = current.get(key)
        if previous_value == current_value:
            continue
        # Unevaluable samples are always ranked behind evaluated traffic.
        if previous_value is None:
            return False
        if current_value is None:
            return True
        previous_number = _numeric_value(previous_value)
        current_number = _numeric_value(current_value)
        if previous_number is not None and current_number is not None:
            if requirement.direction == "desc":
                return previous_number > current_number
            return previous_number < current_number
        if requirement.direction == "desc":
            return str(previous_value) > str(current_value)
        return str(previous_value) < str(current_value)
    return True


def _matches_ratio_metric(
    projection: exp.Expression, requirement: RatioMetricRequirement
) -> bool:
    expression = projection.this if isinstance(projection, exp.Alias) else projection
    divisions = list(expression.find_all(exp.Div))
    if isinstance(expression, exp.Div):
        divisions.insert(0, expression)
    for division in divisions:
        numerator_column, scale = _aggregate_and_scale(division.this)
        denominator_column = _aggregate_column(division.expression)
        if (
            numerator_column == _normalized_name(requirement.numerator_column)
            and denominator_column == _normalized_name(requirement.denominator_column)
            and abs(scale - requirement.scale) < 1e-9
        ):
            return True
    return False


def _matches_dimension_projection(
    projection: exp.Expression, requirement: DimensionRequirement
) -> bool:
    expression = projection.this if isinstance(projection, exp.Alias) else projection
    expression = _unwrap(expression)
    return isinstance(expression, exp.Column) and _normalized_name(
        expression.name
    ) == _normalized_name(requirement.column)


def _matches_raw_failed_count(
    projection: exp.Expression, requirement: RawFailureRateRequirement
) -> bool:
    expression = projection.this if isinstance(projection, exp.Alias) else projection
    return _is_failure_sum(expression, requirement)


def _matches_raw_failure_rate(
    projection: exp.Expression, requirement: RawFailureRateRequirement
) -> bool:
    expression = projection.this if isinstance(projection, exp.Alias) else projection
    # A zero-fallback COALESCE fabricates 0% for an unevaluable sample.
    if isinstance(expression, exp.Coalesce):
        return False
    expression = _unwrap(expression)
    factors: List[tuple[exp.Expression, int]] = []
    if not _collect_ratio_factors(expression, 1, factors):
        return False
    failure_terms = 0
    denominator_terms = 0
    scale = 1.0
    for factor, direction in factors:
        factor = _unwrap(factor)
        numeric = _literal_number(factor)
        if numeric is not None:
            if numeric == 0 and direction < 0:
                return False
            scale = scale * numeric if direction > 0 else scale / numeric
        elif direction > 0 and _is_failure_sum(factor, requirement):
            failure_terms += 1
        elif direction < 0 and _is_safe_count_denominator(factor):
            denominator_terms += 1
        else:
            return False
    return (
        failure_terms == 1
        and denominator_terms == 1
        and abs(scale - requirement.scale) < 1e-9
    )


def _collect_ratio_factors(
    expression: exp.Expression,
    direction: int,
    factors: List[tuple[exp.Expression, int]],
) -> bool:
    expression = _unwrap(expression)
    if isinstance(expression, exp.Mul):
        return _collect_ratio_factors(
            expression.this, direction, factors
        ) and _collect_ratio_factors(expression.expression, direction, factors)
    if isinstance(expression, exp.Div):
        return _collect_ratio_factors(
            expression.this, direction, factors
        ) and _collect_ratio_factors(expression.expression, -direction, factors)
    factors.append((expression, direction))
    return True


def _is_safe_count_denominator(expression: exp.Expression) -> bool:
    expression = _unwrap(expression)
    return (
        isinstance(expression, exp.Nullif)
        and _is_count_all(expression.this)
        and _literal_number(expression.expression) == 0
    )


def _is_failure_sum(
    expression: exp.Expression, requirement: RawFailureRateRequirement
) -> bool:
    expression = _unwrap(expression)
    if not isinstance(expression, exp.Sum):
        return False
    case = _unwrap(expression.this)
    if not isinstance(case, exp.Case):
        return False
    ifs = list(case.args.get("ifs") or [])
    if len(ifs) != 1 or not isinstance(ifs[0], exp.If):
        return False
    if _literal_number(ifs[0].args.get("true")) != 1:
        return False
    if _literal_number(case.args.get("default")) != 0:
        return False
    condition = _unwrap(ifs[0].this)
    if not isinstance(condition, exp.Or):
        return False
    branches = [condition.this, condition.expression]
    return any(_is_business_error(branch, requirement) for branch in branches) and any(
        _is_http_failure(branch, requirement) for branch in branches
    )


def _is_business_error(
    expression: exp.Expression, requirement: RawFailureRateRequirement
) -> bool:
    expression = _unwrap(expression)
    if not isinstance(expression, exp.NEQ):
        return False
    expected = _normalized_name(requirement.business_error_column)
    return (
        isinstance(expression.this, exp.Column)
        and _normalized_name(expression.this.name) == expected
        and isinstance(expression.expression, exp.Literal)
        and str(expression.expression.this) == ""
    ) or (
        isinstance(expression.expression, exp.Column)
        and _normalized_name(expression.expression.name) == expected
        and isinstance(expression.this, exp.Literal)
        and str(expression.this.this) == ""
    )


def _is_http_failure(
    expression: exp.Expression, requirement: RawFailureRateRequirement
) -> bool:
    expression = _unwrap(expression)
    expected = _normalized_name(requirement.http_status_column)
    return (
        isinstance(expression, exp.GTE)
        and isinstance(expression.this, exp.Column)
        and _normalized_name(expression.this.name) == expected
        and _literal_number(expression.expression) == requirement.http_failure_min
    ) or (
        isinstance(expression, exp.LTE)
        and isinstance(expression.expression, exp.Column)
        and _normalized_name(expression.expression.name) == expected
        and _literal_number(expression.this) == requirement.http_failure_min
    )


def _is_count_all(expression: exp.Expression) -> bool:
    expression = expression.this if isinstance(expression, exp.Alias) else expression
    expression = _unwrap(expression)
    if not isinstance(expression, exp.Count):
        return False
    return isinstance(_unwrap(expression.this), exp.Star)


def _has_rolling_window(
    statement: exp.Select, requirement: RollingWindowRequirement
) -> bool:
    where = statement.args.get("where")
    if where is None or requirement.hours <= 0:
        return False
    expected_column = _normalized_name(requirement.column)
    comparisons = list(where.find_all(exp.GTE)) + list(where.find_all(exp.GT))
    for comparison in comparisons:
        if _matches_rolling_comparison(comparison, requirement, expected_column):
            return True
    return False


def _matches_rolling_comparison(
    comparison: exp.Expression,
    requirement: RollingWindowRequirement,
    expected_column: Optional[str] = None,
) -> bool:
    if not isinstance(comparison, (exp.GTE, exp.GT)) or not isinstance(
        comparison.this, exp.Column
    ):
        return False
    expected_column = expected_column or _normalized_name(requirement.column)
    if _normalized_name(comparison.this.name) != expected_column:
        return False
    boundary = _unwrap(comparison.expression)
    if not isinstance(boundary, exp.Sub) or not _is_now(boundary.this):
        return False
    hours = _interval_hours(_unwrap(boundary.expression))
    return hours is not None and abs(hours - requirement.hours) < 1e-9


def _strict_predicate_errors(
    statement: exp.Select, semantic: SQLSemanticContract
) -> List[str]:
    where = statement.args.get("where")
    if where is None:
        return ["strict_predicates_where_missing"]
    expected_filters = {
        (
            _normalized_name(requirement.column),
            requirement.operator,
            requirement.value,
        )
        for requirement in semantic.required_filters
    }
    errors: List[str] = []
    for predicate in _flatten_and_predicates(where.this):
        signature = _literal_filter_signature(predicate)
        if signature in expected_filters:
            continue
        if semantic.rolling_window is not None and _matches_rolling_comparison(
            predicate, semantic.rolling_window
        ):
            continue
        if _is_current_time_upper_bound(predicate, semantic.time_column):
            continue
        errors.append(f"unexpected_predicate:{predicate.sql(pretty=False)}")
    return errors


def _strict_query_shape_errors(
    statement: exp.Select, semantic: SQLSemanticContract
) -> List[str]:
    errors: List[str] = []
    tables = [_normalized_name(table.name) for table in statement.find_all(exp.Table)]
    required_tables = [_normalized_name(table) for table in semantic.allowed_tables]
    if sorted(tables) != sorted(required_tables):
        errors.append("strict_table_shape_mismatch")
    if any(
        table.args.get("db") is not None or table.args.get("catalog") is not None
        for table in statement.find_all(exp.Table)
    ):
        errors.append("qualified_table_forbidden")
    if statement.args.get("joins"):
        errors.append("unexpected_join")
    if statement.args.get("with_") or any(statement.find_all(exp.Subquery)):
        errors.append("unexpected_subquery")
    for clause in ("having", "qualify", "distinct"):
        if statement.args.get(clause) is not None:
            errors.append(f"unexpected_clause:{clause}")

    result_columns = [
        _normalized_name(projection.alias_or_name)
        for projection in statement.expressions
        if _normalized_name(projection.alias_or_name)
    ]
    required_result_columns = [
        _normalized_name(column) for column in semantic.required_result_columns
    ]
    if sorted(result_columns) != sorted(required_result_columns):
        errors.append("strict_result_shape_mismatch")

    group = statement.args.get("group")
    group_by = [_expression_name(item) for item in group.expressions] if group else []
    required_group_by = [
        _normalized_name(column) for column in semantic.required_group_by
    ]
    if sorted(group_by) != sorted(required_group_by):
        errors.append("strict_group_shape_mismatch")

    order = statement.args.get("order")
    order_count = len(order.expressions) if order else 0
    if order_count != len(semantic.required_order_by):
        errors.append("strict_order_shape_mismatch")
    return errors


def _flatten_and_predicates(expression: exp.Expression) -> List[exp.Expression]:
    expression = _unwrap(expression)
    if isinstance(expression, exp.And):
        return _flatten_and_predicates(expression.this) + _flatten_and_predicates(
            expression.expression
        )
    return [expression]


def _literal_filter_signature(
    expression: exp.Expression,
) -> Optional[tuple[str, str, str]]:
    operator = "eq" if isinstance(expression, exp.EQ) else "neq"
    if not isinstance(expression, (exp.EQ, exp.NEQ)):
        return None
    left = expression.this
    right = expression.expression
    if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
        return _normalized_name(left.name), operator, str(right.this)
    if isinstance(right, exp.Column) and isinstance(left, exp.Literal):
        return _normalized_name(right.name), operator, str(left.this)
    return None


def _is_current_time_upper_bound(
    expression: exp.Expression, time_column: Optional[str]
) -> bool:
    if not isinstance(expression, (exp.LT, exp.LTE)) or not isinstance(
        expression.this, exp.Column
    ):
        return False
    expected = _normalized_name(time_column)
    return bool(
        expected
        and _normalized_name(expression.this.name) == expected
        and _is_now(expression.expression)
    )


def _is_now(expression: exp.Expression) -> bool:
    expression = _unwrap(expression)
    if isinstance(expression, exp.CurrentTimestamp):
        return True
    return (
        isinstance(expression, exp.Anonymous)
        and _normalized_name(expression.name) == "now"
    )


def _interval_hours(expression: exp.Expression) -> Optional[float]:
    if not isinstance(expression, exp.Interval):
        return None
    literal = _unwrap(expression.this)
    try:
        value = float(literal.this) if isinstance(literal, exp.Literal) else None
    except (TypeError, ValueError):
        value = None
    unit_expression = expression.args.get("unit")
    unit = _normalized_name(
        unit_expression.this
        if isinstance(unit_expression, exp.Expression)
        else unit_expression
    )
    if value is None:
        return None
    factors = {"hour": 1.0, "hours": 1.0, "day": 24.0, "days": 24.0}
    factor = factors.get(unit)
    return value * factor if factor is not None else None


def _aggregate_and_scale(expression: exp.Expression) -> tuple[str, float]:
    expression = _unwrap(expression)
    if isinstance(expression, exp.Mul):
        left_column = _aggregate_column(expression.this)
        right_scale = _literal_number(expression.expression)
        if left_column and right_scale is not None:
            return left_column, right_scale
        right_column = _aggregate_column(expression.expression)
        left_scale = _literal_number(expression.this)
        if right_column and left_scale is not None:
            return right_column, left_scale
    return _aggregate_column(expression), 1.0


def _aggregate_column(expression: exp.Expression) -> str:
    expression = _unwrap(expression)
    if isinstance(expression, exp.Nullif):
        expression = _unwrap(expression.this)
    if isinstance(expression, exp.Sum):
        column = expression.this
        if isinstance(column, exp.Column):
            return _normalized_name(column.name)
    return ""


def _unwrap(expression: exp.Expression) -> exp.Expression:
    while isinstance(expression, (exp.Paren, exp.Cast, exp.TryCast)):
        expression = expression.this
    return expression


def _literal_filters(statement: exp.Select) -> Dict[str, Dict[str, List[str]]]:
    filters: Dict[str, Dict[str, List[str]]] = {}
    where = statement.args.get("where")
    if where is None:
        return filters
    for operator, expression_type in (("eq", exp.EQ), ("neq", exp.NEQ)):
        for comparison in where.find_all(expression_type):
            left = comparison.this
            right = comparison.expression
            if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                filters.setdefault(_normalized_name(left.name), {}).setdefault(
                    operator, []
                ).append(str(right.this))
            elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
                filters.setdefault(_normalized_name(right.name), {}).setdefault(
                    operator, []
                ).append(str(left.this))
    return filters


def _expression_name(expression: exp.Expression) -> str:
    if isinstance(expression, exp.Column):
        return _normalized_name(expression.name)
    return _normalized_name(expression.alias_or_name or expression.sql())


def _literal_number(expression: Optional[exp.Expression]) -> Optional[float]:
    expression = _unwrap(expression) if expression is not None else None
    if not isinstance(expression, exp.Literal) or not expression.is_number:
        return None
    try:
        return float(expression.this)
    except (TypeError, ValueError):
        return None


def _normalized_name(value: Any) -> str:
    return str(value or "").strip().strip('`"').lower()
