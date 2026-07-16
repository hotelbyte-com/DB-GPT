"""Deterministic compiler for governed HotelByte data-query contracts.

The registry is keyed only by typed contract identity. Business semantics stay
in the caller-owned contract and are compiled into an unqualified query for the
physical datasource selected by DB-GPT.
"""

import re
from typing import Callable, Dict, List, Literal, Optional, Tuple

from dbgpt._private.pydantic import BaseModel, Field
from dbgpt_app.scene.chat_db.query_contract import (
    DATA_QUERY_CONTRACT_VERSION,
    DataQueryContract,
    SQLSemanticEvidence,
    validate_sql_semantics,
)

ISSUE_21706_CONTRACT_ID = "supplier_hotelrates_failure_rate_24h"
DETERMINISTIC_CONTRACT_STRATEGY = "deterministic_contract"
RAW_FAILURE_RATE_COMPILER = "hotelbyte.raw_failure_rate/v1"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CompiledDataQuery(BaseModel):
    contract_id: str
    contract_version: str
    strategy: Literal[DETERMINISTIC_CONTRACT_STRATEGY]
    compiler: str
    dialect: str = "tdengine"
    sql: str
    sql_semantics: SQLSemanticEvidence


class ContractCompilation(BaseModel):
    status: Literal["compiled", "unsupported", "invalid"]
    query: Optional[CompiledDataQuery] = None
    gap_kind: str = ""
    reason: str = ""
    evidence: Dict[str, object] = Field(default_factory=dict)


Compiler = Callable[[DataQueryContract], Tuple[Optional[str], List[str]]]


def compile_data_query_contract(contract: DataQueryContract) -> ContractCompilation:
    """Compile a registered contract or return an explicit typed gap."""

    key = (contract.contract_id, contract.version)
    compiler = _COMPILERS.get(key)
    if compiler is None:
        return ContractCompilation(
            status="unsupported",
            gap_kind="contract_compiler_unsupported",
            reason="no deterministic compiler is registered for the contract",
            evidence={
                "contract_id": contract.contract_id,
                "contract_version": contract.version,
            },
        )

    sql, compile_errors = compiler(contract)
    if sql is None:
        return ContractCompilation(
            status="invalid",
            gap_kind="contract_compile_invalid",
            reason="the declared contract cannot be compiled safely",
            evidence={"errors": compile_errors},
        )

    semantic_evidence = validate_sql_semantics(sql, contract)
    if not semantic_evidence.valid:
        return ContractCompilation(
            status="invalid",
            gap_kind="contract_compile_invalid",
            reason="compiled SQL does not satisfy the declared contract",
            evidence={"sql_semantics": semantic_evidence.model_dump()},
        )

    return ContractCompilation(
        status="compiled",
        query=CompiledDataQuery(
            contract_id=contract.contract_id,
            contract_version=contract.version,
            strategy=DETERMINISTIC_CONTRACT_STRATEGY,
            compiler=RAW_FAILURE_RATE_COMPILER,
            sql=sql,
            sql_semantics=semantic_evidence,
        ),
    )


def _compile_raw_failure_rate(
    contract: DataQueryContract,
) -> Tuple[Optional[str], List[str]]:
    semantic = contract.semantic
    errors: List[str] = []
    if len(semantic.allowed_tables) != 1:
        errors.append("exactly_one_allowed_table_required")
    if len(semantic.required_dimensions) != 1:
        errors.append("exactly_one_dimension_required")
    if not semantic.strict_predicates:
        errors.append("strict_predicates_required")
    if not semantic.required_filters:
        errors.append("required_filters_missing")
    if not semantic.required_order_by:
        errors.append("required_order_missing")
    if semantic.raw_failure_rate is None:
        errors.append("raw_failure_rate_required")
    if semantic.ratio_metrics:
        errors.append("additional_ratio_metrics_unsupported")
    if semantic.rolling_window is None or semantic.rolling_window.hours <= 0:
        errors.append("positive_rolling_window_required")
    if not 0 < semantic.limit_max <= 100:
        errors.append("bounded_limit_required")
    if errors:
        return None, errors

    table = semantic.allowed_tables[0]
    dimension = semantic.required_dimensions[0]
    failure = semantic.raw_failure_rate
    rolling = semantic.rolling_window
    assert failure is not None and rolling is not None

    identifiers = [
        table,
        dimension.alias,
        dimension.column,
        failure.rate_alias,
        failure.failed_count_alias,
        failure.total_count_alias,
        failure.business_error_column,
        failure.http_status_column,
        rolling.column,
        *semantic.required_group_by,
        *(order.expression for order in semantic.required_order_by),
        *(item.column for item in semantic.required_filters),
        *semantic.required_result_columns,
    ]
    invalid_identifiers = sorted(
        {
            identifier
            for identifier in identifiers
            if not _IDENTIFIER.fullmatch(identifier)
        }
    )
    if invalid_identifiers:
        return None, [f"unsafe_identifier:{value}" for value in invalid_identifiers]
    if semantic.required_group_by != [dimension.column]:
        errors.append("dimension_group_by_mismatch")
    expected_columns = {
        dimension.alias,
        failure.failed_count_alias,
        failure.total_count_alias,
        failure.rate_alias,
    }
    if set(semantic.required_result_columns) != expected_columns:
        errors.append("raw_failure_result_shape_mismatch")
    if any(
        order.expression not in expected_columns for order in semantic.required_order_by
    ):
        errors.append("order_expression_not_projected")
    if errors:
        return None, errors

    failure_predicate = (
        f"{failure.business_error_column} != '' OR "
        f"{failure.http_status_column} >= {failure.http_failure_min}"
    )
    failed_sum = f"SUM(CASE WHEN {failure_predicate} THEN 1 ELSE 0 END)"
    filters = [
        f"{item.column} {'=' if item.operator == 'eq' else '!='} "
        f"'{_escape_literal(item.value)}'"
        for item in semantic.required_filters
    ]
    filters.append(f"{rolling.column} >= NOW - {rolling.hours}h")
    order_by = ", ".join(
        f"{item.expression} {item.direction.upper()}"
        for item in semantic.required_order_by
    )
    sql = "\n".join(
        [
            "SELECT",
            f"  {dimension.column} AS {dimension.alias},",
            f"  {failed_sum} AS {failure.failed_count_alias},",
            f"  COUNT(*) AS {failure.total_count_alias},",
            f"  {failed_sum} * {float(failure.scale)} "
            f"/ COUNT(*) AS {failure.rate_alias}",
            f"FROM {table}",
            "WHERE " + "\n  AND ".join(filters),
            "GROUP BY " + ", ".join(semantic.required_group_by),
            "ORDER BY " + order_by,
            f"LIMIT {semantic.limit_max}",
        ]
    )
    return sql, []


def _escape_literal(value: str) -> str:
    return str(value).replace("'", "''")


_COMPILERS: Dict[Tuple[str, str], Compiler] = {
    (ISSUE_21706_CONTRACT_ID, DATA_QUERY_CONTRACT_VERSION): _compile_raw_failure_rate,
}
