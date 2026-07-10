"""Resolve logical chat_data datasource names to concrete DB-GPT datasources.

Legacy callers retain their historical best-effort resolver. Callers that
declare a :class:`DataQueryContract` use capability-based, fail-closed routing
and receive evidence describing the physical source decision.
"""

import json
import os
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from dbgpt._private.pydantic import BaseModel, Field
from dbgpt.component import SystemApp
from dbgpt_app.scene.chat_db.query_contract import DataQueryContract
from dbgpt_serve.datasource.manages.connect_config_db import ConnectConfigDao

CHAT_DATA_GROUPS_ENV = "DBGPT_CHAT_DATA_GROUPS"


class SourceCandidateSpec(BaseModel):
    """Deployment-owned physical source capability declaration."""

    name: str
    capabilities: List[str] = Field(default_factory=list)
    priority: int = 0


class DatasourceCandidateEvidence(BaseModel):
    """Non-secret evidence collected while resolving one candidate."""

    name: str
    datasource_type: str = ""
    capabilities: List[str] = Field(default_factory=list)
    priority: int = 0
    tables: List[str] = Field(default_factory=list)
    columns: Dict[str, List[str]] = Field(default_factory=dict)
    healthy: bool = False
    missing_capabilities: List[str] = Field(default_factory=list)
    missing_tables: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    error: str = ""


class DatasourceResolution(BaseModel):
    """Typed source decision returned to the agentic execution boundary."""

    logical_group: str
    status: Literal[
        "selected",
        "group_unconfigured",
        "capability_unavailable",
        "schema_unavailable",
    ]
    selected: Optional[str] = None
    selected_type: str = ""
    required_capabilities: List[str] = Field(default_factory=list)
    required_columns: List[str] = Field(default_factory=list)
    candidates: List[DatasourceCandidateEvidence] = Field(default_factory=list)
    gap_kind: str = ""
    reason: str = ""


def resolve_chat_data_source_with_evidence(
    select_param: Optional[str],
    user_input: str,
    system_app: SystemApp,
    *,
    contract: DataQueryContract,
    dao: Optional[ConnectConfigDao] = None,
    connector_manager: Optional[Any] = None,
) -> DatasourceResolution:
    """Resolve a declared logical group by required capabilities.

    The question text is deliberately not used. Product routing is based on a
    versioned contract plus deployment-owned capability metadata. Candidate
    schema discovery is real connector evidence; an empty or failed schema is
    never reported as a healthy source.

    Deployment example::

        {
            "hotel-be": {
                "candidates": [
                    {
                        "name": "hblog-shared",
                        "capabilities": [
                            "operational_logs",
                            "supplier_reliability",
                            "time_series",
                        ],
                        "priority": 100,
                    }
                ]
            }
        }
    """

    del user_input
    dao = dao or ConnectConfigDao()
    logical_group = str(select_param or contract.logical_group).strip()
    required_capabilities = _normalized_unique(contract.required_capabilities)
    required_tables = list(
        dict.fromkeys(
            _normalized_table_name(table)
            for table in contract.semantic.allowed_tables
            if _normalized_table_name(table)
        )
    )
    required_columns = _required_source_columns(contract)

    if logical_group != contract.logical_group:
        return DatasourceResolution(
            logical_group=logical_group,
            status="group_unconfigured",
            required_capabilities=required_capabilities,
            required_columns=required_columns,
            gap_kind="source_unavailable",
            reason="logical_group_mismatch",
        )

    groups = _load_group_specs(os.environ.get(CHAT_DATA_GROUPS_ENV, ""))
    specs = groups.get(logical_group, [])
    if not specs:
        return DatasourceResolution(
            logical_group=logical_group,
            status="group_unconfigured",
            required_capabilities=required_capabilities,
            required_columns=required_columns,
            gap_kind="source_unavailable",
            reason="logical_group_not_configured",
        )

    if connector_manager is None:
        from dbgpt_serve.datasource.manages import ConnectorManager

        connector_manager = ConnectorManager.get_instance(system_app)

    evidence: List[DatasourceCandidateEvidence] = []
    eligible: List[DatasourceCandidateEvidence] = []
    for spec in sorted(specs, key=lambda item: (-item.priority, item.name)):
        configured_capabilities = _normalized_unique(spec.capabilities)
        missing = [
            capability
            for capability in required_capabilities
            if capability not in configured_capabilities
        ]
        candidate = DatasourceCandidateEvidence(
            name=spec.name,
            datasource_type=_datasource_type(spec.name, dao),
            capabilities=configured_capabilities,
            priority=spec.priority,
            missing_capabilities=missing,
        )
        evidence.append(candidate)
        if missing:
            candidate.error = "capability_mismatch"
            continue
        if not dao.get_by_names(spec.name):
            candidate.error = "datasource_not_registered"
            eligible.append(candidate)
            continue
        try:
            connector = connector_manager.get_connector(spec.name)
            candidate.tables = sorted(
                str(table).strip()
                for table in connector.get_table_names()
                if str(table).strip()
            )
        except Exception as exc:
            candidate.error = f"schema_introspection_failed:{type(exc).__name__}"
            eligible.append(candidate)
            continue
        if not candidate.tables:
            candidate.error = "schema_empty"
            eligible.append(candidate)
            continue
        available_tables = {
            _normalized_table_name(table): table for table in candidate.tables
        }
        candidate.missing_tables = [
            table for table in required_tables if table not in available_tables
        ]
        if candidate.missing_tables:
            candidate.error = "required_tables_missing"
            eligible.append(candidate)
            continue
        available_columns = set()
        for table in required_tables:
            actual_table = available_tables[table]
            try:
                table_columns = _column_names(connector.get_columns(actual_table))
            except Exception as exc:
                candidate.error = f"column_introspection_failed:{type(exc).__name__}"
                break
            candidate.columns[table] = table_columns
            available_columns.update(table_columns)
            if not table_columns:
                candidate.error = "schema_columns_empty"
                break
        if candidate.error:
            eligible.append(candidate)
            continue
        candidate.missing_columns = [
            column for column in required_columns if column not in available_columns
        ]
        if candidate.missing_columns:
            candidate.error = "required_columns_missing"
            eligible.append(candidate)
            continue
        candidate.healthy = True
        eligible.append(candidate)

    if not eligible:
        return DatasourceResolution(
            logical_group=logical_group,
            status="capability_unavailable",
            required_capabilities=required_capabilities,
            required_columns=required_columns,
            candidates=evidence,
            gap_kind="source_unavailable",
            reason="required_capability_not_declared",
        )

    selected = next((candidate for candidate in eligible if candidate.healthy), None)
    if selected is None:
        return DatasourceResolution(
            logical_group=logical_group,
            status="schema_unavailable",
            required_capabilities=required_capabilities,
            required_columns=required_columns,
            candidates=evidence,
            gap_kind="source_unavailable",
            reason="no_schema_healthy_candidate",
        )

    return DatasourceResolution(
        logical_group=logical_group,
        status="selected",
        selected=selected.name,
        selected_type=selected.datasource_type,
        required_capabilities=required_capabilities,
        required_columns=required_columns,
        candidates=evidence,
        reason="highest_priority_healthy_capability_match",
    )


def resolve_chat_data_source(
    select_param: Optional[str], user_input: str, system_app: SystemApp
) -> str:
    """Return the concrete datasource for a chat_data request.

    ``chat_param`` historically names exactly one DB-GPT datasource. For
    deployments like hotel-be it is useful to expose one stable logical name
    while DB-GPT chooses from several physical datasources. Groups can be
    configured through ``DBGPT_CHAT_DATA_GROUPS`` as JSON, for example::

        {"hotel-be": ["hotel", "hotel_user", "hotel_trade"]}

    A semicolon form is also accepted: ``hotel-be=hotel,hotel_user``. Passing
    ``chat_param`` as a comma separated list works as an inline group.
    """

    if not select_param:
        return select_param

    dao = ConnectConfigDao()
    candidates = _resolve_candidates(select_param, dao)
    if len(candidates) <= 1:
        return candidates[0] if candidates else select_param
    return _choose_candidate(candidates, user_input, system_app, dao)


def _resolve_candidates(select_param: str, dao: ConnectConfigDao) -> List[str]:
    inline_candidates = _split_candidates(select_param)
    if len(inline_candidates) > 1:
        return _existing_candidates(inline_candidates, dao) or inline_candidates

    groups = _load_groups(os.environ.get(CHAT_DATA_GROUPS_ENV, ""))
    configured_candidates = groups.get(select_param, [])
    if configured_candidates:
        return _existing_candidates(configured_candidates, dao) or configured_candidates

    # HotelByte convention: expose one product-level chat_param while concrete
    # DB-GPT datasource names keep their physical DB names.
    if select_param == "hotel-be":
        hotel_candidates = _hotel_be_default_candidates(dao)
        if hotel_candidates:
            return hotel_candidates

    if dao.get_by_names(select_param):
        return [select_param]
    return [select_param]


def _load_groups(raw: str) -> Dict[str, List[str]]:
    return {
        group: [candidate.name for candidate in candidates]
        for group, candidates in _load_group_specs(raw).items()
    }


def _load_group_specs(raw: str) -> Dict[str, List[SourceCandidateSpec]]:
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            group: [
                SourceCandidateSpec(name=name, priority=-index)
                for index, name in enumerate(names)
            ]
            for group, names in _parse_semicolon_groups(raw).items()
        }
    groups: Dict[str, List[SourceCandidateSpec]] = {}
    if not isinstance(parsed, dict):
        return groups
    for name, value in parsed.items():
        if isinstance(value, dict):
            value = value.get("candidates", [])
        if isinstance(value, str):
            value = _split_candidates(value)
        elif isinstance(value, list):
            pass
        else:
            continue
        candidates: List[SourceCandidateSpec] = []
        for index, item in enumerate(value):
            if isinstance(item, dict):
                try:
                    candidate = SourceCandidateSpec.model_validate(item)
                except (TypeError, ValueError):
                    continue
            else:
                candidate_name = str(item).strip()
                if not candidate_name:
                    continue
                candidate = SourceCandidateSpec(name=candidate_name, priority=-index)
            if candidate.name.strip():
                candidate.name = candidate.name.strip()
                candidates.append(candidate)
        groups[str(name)] = candidates
    return groups


def _datasource_type(candidate: str, dao: ConnectConfigDao) -> str:
    try:
        rows = dao.get_db_list(db_name=candidate)
    except Exception:
        return ""
    if not rows:
        return ""
    row = rows[0]
    if isinstance(row, dict):
        return str(row.get("db_type") or row.get("type") or "")
    return str(getattr(row, "db_type", "") or getattr(row, "type", ""))


def _normalized_unique(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        normalized = str(value).strip().lower()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _required_source_columns(contract: DataQueryContract) -> List[str]:
    semantic = contract.semantic
    columns = [item.column for item in semantic.required_filters]
    columns.extend(item.column for item in semantic.required_dimensions)
    columns.extend(semantic.required_group_by)
    columns.extend(
        column
        for metric in semantic.ratio_metrics
        for column in (metric.numerator_column, metric.denominator_column)
    )
    if semantic.raw_failure_rate is not None:
        columns.extend(
            [
                semantic.raw_failure_rate.business_error_column,
                semantic.raw_failure_rate.http_status_column,
            ]
        )
    if semantic.rolling_window is not None:
        columns.append(semantic.rolling_window.column)
    if semantic.time_column:
        columns.append(semantic.time_column)
    return _normalized_unique(columns)


def _column_names(columns: Any) -> List[str]:
    names: List[str] = []
    for column in columns or []:
        if isinstance(column, dict):
            value = column.get("name") or column.get("column_name")
        elif isinstance(column, (list, tuple)):
            value = column[0] if column else ""
        else:
            value = getattr(column, "name", column)
        normalized = _normalized_table_name(value)
        if normalized and normalized not in names:
            names.append(normalized)
    return sorted(names)


def _normalized_table_name(value: str) -> str:
    return str(value).strip().strip('`"').split(".")[-1].lower()


def _parse_semicolon_groups(raw: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        group_name = name.strip()
        if group_name:
            groups[group_name] = _split_candidates(value)
    return groups


def _split_candidates(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _existing_candidates(candidates: Sequence[str], dao: ConnectConfigDao) -> List[str]:
    return [candidate for candidate in candidates if dao.get_by_names(candidate)]


def _hotel_be_default_candidates(dao: ConnectConfigDao) -> List[str]:
    datasources = dao.get_db_list()
    names = [str(item.get("db_name", "")) for item in datasources]
    hotel_names = [
        name for name in names if name == "hotel" or name.startswith("hotel_")
    ]
    return hotel_names or [name for name in names if name.startswith("hotel")]


def _choose_candidate(
    candidates: Sequence[str],
    user_input: str,
    system_app: SystemApp,
    dao: ConnectConfigDao,
) -> str:
    from dbgpt_serve.datasource.manages import ConnectorManager

    connector_manager = ConnectorManager.get_instance(system_app)
    question = user_input.lower()

    def safe_tables(candidate: str) -> Iterable[str]:
        try:
            connector = connector_manager.get_connector(candidate)
            return connector.get_table_names()
        except Exception:
            return []

    scored = []
    for index, candidate in enumerate(candidates):
        metadata = _datasource_metadata(candidate, dao)
        tables = list(safe_tables(candidate))
        score = _score_candidate(candidate, metadata, tables, question)
        scored.append((score, -index, candidate))
    scored.sort(reverse=True)
    return scored[0][2]


def _datasource_metadata(candidate: str, dao: ConnectConfigDao) -> str:
    try:
        rows = dao.get_db_list(db_name=candidate)
    except Exception:
        return candidate
    if not rows:
        return candidate
    row = rows[0]
    values = [candidate, str(row.get("comment") or ""), str(row.get("db_type") or "")]
    return " ".join(values).lower()


def _score_candidate(
    candidate: str, metadata: str, tables: Sequence[str], question: str
) -> int:
    score = _token_score(candidate, question) * 4 + _token_score(metadata, question)
    for table in tables:
        table_lower = str(table).lower()
        if table_lower and table_lower in question:
            score += 40
        score += _token_score(table_lower, question) * 3
    return score


def _token_score(text: str, question: str) -> int:
    score = 0
    for token in _tokens(text):
        if len(token) >= 3 and token in question:
            score += 1
    return score


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())
