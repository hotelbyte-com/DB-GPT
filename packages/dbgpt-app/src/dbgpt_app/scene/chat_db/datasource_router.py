"""Resolve logical chat_data datasource names to concrete DB-GPT datasources."""

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

from dbgpt.component import SystemApp
from dbgpt_serve.datasource.manages.connect_config_db import ConnectConfigDao


CHAT_DATA_GROUPS_ENV = "DBGPT_CHAT_DATA_GROUPS"


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
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _parse_semicolon_groups(raw)
    groups: Dict[str, List[str]] = {}
    if not isinstance(parsed, dict):
        return groups
    for name, value in parsed.items():
        if isinstance(value, str):
            groups[str(name)] = _split_candidates(value)
        elif isinstance(value, list):
            groups[str(name)] = [str(item).strip() for item in value if str(item).strip()]
    return groups


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
    hotel_names = [name for name in names if name == "hotel" or name.startswith("hotel_")]
    return hotel_names or [name for name in names if name.startswith("hotel")]


def _choose_candidate(
    candidates: Sequence[str], user_input: str, system_app: SystemApp, dao: ConnectConfigDao
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
