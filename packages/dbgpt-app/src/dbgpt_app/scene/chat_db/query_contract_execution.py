"""Execute a compiled data query and produce validated product evidence."""

import json
from typing import Any, Dict, List, Literal, Optional

from dbgpt._private.pydantic import BaseModel, Field
from dbgpt_app.scene.chat_db.query_contract import (
    DataQueryContract,
    validate_result_semantics,
    validate_sql_semantics,
)
from dbgpt_app.scene.chat_db.query_contract_compiler import ContractCompilation


class GovernedQueryOutcome(BaseModel):
    status: Literal["executed", "gap"]
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    artifact: Optional[Dict[str, Any]] = None
    gap: Optional[Dict[str, Any]] = None
    final_content: str


def execute_compiled_data_query(
    compilation: ContractCompilation,
    contract: DataQueryContract,
    connector: Any,
    *,
    physical_source_name: str,
    physical_source_type: str,
    source_resolution: Optional[Dict[str, Any]],
    schema: Dict[str, Any],
) -> GovernedQueryOutcome:
    """Execute on the already-resolved connector and validate all returned rows."""

    if compilation.status != "compiled" or compilation.query is None:
        return _gap(
            compilation.gap_kind or "contract_compile_invalid",
            compilation.reason or "the query contract was not compiled",
            compilation.evidence,
        )
    if connector is None:
        return _gap(
            "source_unavailable",
            "no schema-healthy physical datasource was resolved",
            {"source_resolution": source_resolution, "schema": schema},
        )

    compiled = compilation.query
    normalized_source_type = str(physical_source_type or "").strip().lower()
    if normalized_source_type != compiled.dialect:
        return _gap(
            "source_dialect_mismatch",
            "compiled SQL dialect does not match the selected physical datasource",
            {
                "compiled_dialect": compiled.dialect,
                "physical_source_type": physical_source_type,
                "source_resolution": source_resolution,
            },
        )
    sql_evidence = validate_sql_semantics(compiled.sql, contract)
    if not sql_evidence.valid:
        return _gap(
            "sql_semantic_invalid",
            "compiled SQL no longer satisfies the declared business contract",
            sql_evidence.model_dump(),
        )
    try:
        result = connector.run(compiled.sql)
    except Exception as exc:
        return _gap(
            "sql_execution_failed",
            f"database execution failed with {type(exc).__name__}",
            {
                "source_resolution": source_resolution,
                "schema": schema,
                "sql_semantics": sql_evidence.model_dump(),
            },
        )
    if not result:
        return _gap(
            "result_evidence_missing",
            "database returned neither column metadata nor rows",
            sql_evidence.model_dump(),
        )

    columns = result[0]
    col_names = [
        str(item[0]) if isinstance(item, tuple) else str(item) for item in columns
    ]
    rows = list(result[1:])
    result_evidence = validate_result_semantics(col_names, rows, contract)
    if not result_evidence.valid:
        return _gap(
            "result_semantic_invalid",
            "executed rows are not equivalent to the declared formula or order",
            result_evidence.model_dump(),
        )

    visible_rows = [_row_dict(col_names, row) for row in rows[:50]]
    artifact = {
        "display_type": "response_table",
        "title": "Supplier hotelRates failure rate (rolling 24 hours)",
        "sql": compiled.sql,
        "columns": col_names,
        "rows": visible_rows,
        "row_count": len(rows),
        "provenance": {
            "contract": {
                "id": contract.contract_id,
                "version": contract.version,
                "logical_group": contract.logical_group,
                "required_capabilities": contract.required_capabilities,
            },
            "query_plan": {
                "strategy": compiled.strategy,
                "compiler": compiled.compiler,
                "dialect": compiled.dialect,
                "contract_id": compiled.contract_id,
                "contract_version": compiled.contract_version,
            },
            "physical_source": {
                "name": physical_source_name,
                "type": physical_source_type,
            },
            "source_resolution": source_resolution,
            "schema": schema,
            "sql_semantics": sql_evidence.model_dump(),
            "result_semantics": result_evidence.model_dump(),
        },
    }
    table = _markdown_table(col_names, visible_rows)
    if result_evidence.status == "not_evaluable":
        table += (
            "\n\nNo evaluable supplier traffic matched the declared rolling window; "
            "this is not a 0% failure rate."
        )
    if len(rows) > 50:
        table += f"\n\n(Showing 50 of {len(rows)} rows.)"
    artifact_json = json.dumps(artifact, default=str, ensure_ascii=False)
    return GovernedQueryOutcome(
        status="executed",
        artifact=artifact,
        chunks=[
            {"output_type": "markdown", "content": table},
            {
                "output_type": "markdown",
                "content": f"```response_table\n{artifact_json}\n```",
            },
        ],
        final_content=(
            f"Executed governed contract {contract.contract_id} on the selected "
            f"DB-GPT datasource; result status={result_evidence.status}, "
            f"row_count={len(rows)}."
        ),
    )


def _row_dict(columns: List[str], row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return {column: row.get(column) for column in columns}
    if hasattr(row, "_mapping"):
        return {column: row._mapping.get(column) for column in columns}
    values = list(row) if isinstance(row, (list, tuple)) else [row]
    return {
        column: values[index] if index < len(values) else None
        for index, column in enumerate(columns)
    }


def _markdown_table(columns: List[str], rows: List[Dict[str, Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _gap(kind: str, reason: str, evidence: Any) -> GovernedQueryOutcome:
    gap = {"kind": kind, "reason": reason, "evidence": evidence}
    return GovernedQueryOutcome(
        status="gap",
        gap=gap,
        chunks=[
            {
                "output_type": "text",
                "content": f"Data query gap [{kind}]: {reason}",
            }
        ],
        final_content=f"Data query gap [{kind}]: {reason}",
    )
