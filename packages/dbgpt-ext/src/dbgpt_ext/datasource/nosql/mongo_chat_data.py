"""Config-driven MongoDB chat_data runtime.

This module is intentionally project-neutral. Each project binds `chat_param` to
a Mongo database/collection through configuration; the DB-GPT model plans the
read-only aggregation, while this runtime validates, guards, executes, and
hands the resulting facts back to the model for interpretation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Tuple

from dbgpt.core import ModelMessageRoleType, ModelRequest

if TYPE_CHECKING:
    from dbgpt.model.cluster.manager_base import WorkerManager

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover - reported as a runtime configuration gap.
    MongoClient = None  # type: ignore


CONFIG_FILE_ENV = "DBGPT_MONGO_CHAT_DATA_CONFIG_FILE"
CONFIG_JSON_ENV = "DBGPT_MONGO_CHAT_DATA_CONFIG"
DATA_PROVENANCE_CONTRACT_VERSION = "data-provenance.v1"

_DATA_PROVENANCE_FIELDS = frozenset(
    {
        "contractVersion",
        "source",
        "schemaFingerprint",
        "compiledPlanFingerprint",
        "resultFingerprint",
        "rowCount",
    }
)
_SHA256_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")

_ALLOWED_STAGES = {"$match", "$sort", "$group", "$addFields", "$project", "$limit"}
_ALLOWED_MATCH_OPERATORS = {"$gte", "$gt", "$lte", "$lt", "$eq", "$in", "$ne"}
_ALLOWED_GROUP_OPERATORS = {
    "$sum",
    "$first",
    "$last",
    "$avg",
    "$min",
    "$max",
    "$stdDevPop",
    "$stdDevSamp",
}
_ALLOWED_EXPRESSION_OPERATORS = _ALLOWED_GROUP_OPERATORS | {
    "$cond",
    "$eq",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$and",
    "$or",
    "$add",
    "$subtract",
    "$multiply",
    "$divide",
    "$ifNull",
    "$ne",
    "$in",
    "$not",
}


@dataclass
class MongoMetricSpec:
    name: str
    op: str
    field: str = ""
    fields: List[str] = dc_field(default_factory=list)


@dataclass
class MongoSummaryMetricSpec:
    name: str
    op: str
    field: str = ""
    weight: str = ""


@dataclass
class MongoQueryPlan:
    pipeline: List[Dict[str, Any]]
    reason: str = ""
    planner_schema: Dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class MongoChatDataApp:
    name: str
    uri: str
    database: str
    collection: str
    source: str
    time_field: str = "timestamp"
    group_field: str = ""
    filter_fields: List[str] = dc_field(default_factory=list)
    row_defaults: Dict[str, Any] = dc_field(default_factory=dict)
    group_label: str = "group"
    metrics: List[MongoMetricSpec] = dc_field(default_factory=list)
    derived: List[MongoMetricSpec] = dc_field(default_factory=list)
    summary_metrics: List[MongoSummaryMetricSpec] = dc_field(default_factory=list)
    default_window_hours: int = 24
    row_limit: int = 10
    prompt: str = ""

    @classmethod
    def from_mapping(cls, name: str, raw: Mapping[str, Any]) -> "MongoChatDataApp":
        return cls(
            name=name,
            uri=_resolve_env(raw.get("uri") or "mongodb://127.0.0.1:27017"),
            database=_resolve_env(raw.get("database") or ""),
            collection=_resolve_env(raw.get("collection") or ""),
            source=_resolve_env(raw.get("source") or ""),
            time_field=str(
                raw.get("timeField") or raw.get("time_field") or "timestamp"
            ),
            group_field=str(raw.get("groupField") or raw.get("group_field") or ""),
            filter_fields=[
                str(v)
                for v in raw.get("filterFields") or raw.get("filter_fields") or []
            ],
            row_defaults=dict(raw.get("rowDefaults") or raw.get("row_defaults") or {}),
            group_label=str(raw.get("groupLabel") or raw.get("group_label") or "group"),
            metrics=[
                MongoMetricSpec(
                    name=str(item.get("name") or ""),
                    op=str(item.get("op") or ""),
                    field=str(item.get("field") or ""),
                    fields=[str(v) for v in item.get("fields") or []],
                )
                for item in raw.get("metrics") or []
            ],
            derived=[
                MongoMetricSpec(
                    name=str(item.get("name") or ""),
                    op=str(item.get("op") or ""),
                    field=str(item.get("field") or ""),
                    fields=[str(v) for v in item.get("fields") or []],
                )
                for item in raw.get("derived") or []
            ],
            summary_metrics=[
                MongoSummaryMetricSpec(
                    name=str(item.get("name") or ""),
                    op=str(item.get("op") or ""),
                    field=str(item.get("field") or ""),
                    weight=str(item.get("weight") or ""),
                )
                for item in raw.get("summaryMetrics")
                or raw.get("summary_metrics")
                or []
            ],
            default_window_hours=int(
                raw.get("defaultWindowHours") or raw.get("default_window_hours") or 24
            ),
            row_limit=int(raw.get("rowLimit") or raw.get("row_limit") or 10),
            prompt=str(raw.get("prompt") or ""),
        )

    def validate(self) -> None:
        missing = []
        for name in ["name", "uri", "database", "collection", "source"]:
            if not getattr(self, name):
                missing.append(name)
        if missing:
            raise ValueError(
                f"Mongo chat_data app {self.name!r} missing config: "
                f"{', '.join(missing)}"
            )
        if not self.metrics:
            raise ValueError(f"Mongo chat_data app {self.name!r} has no metrics")

    async def plan_query(
        self,
        *,
        prompt: str,
        model: str,
        worker_manager: WorkerManager,
        conv_uid: Optional[str],
    ) -> Tuple[MongoQueryPlan, Dict[str, int]]:
        planner_schema = _planner_schema(self)
        model_output = await worker_manager.generate(
            _query_plan_request_dict(
                app=self,
                prompt=prompt,
                model=model,
                conv_uid=conv_uid,
                planner_schema=planner_schema,
            )
        )
        if not model_output.success:
            _raise_model_error(model, model_output.text)
        query_plan = _parse_query_plan(model_output.text)
        query_plan.pipeline = _guard_pipeline(self, query_plan.pipeline, prompt)
        _validate_pipeline(self, query_plan.pipeline)
        query_plan.planner_schema = planner_schema
        return query_plan, _usage_from_model(model_output.usage)

    def query(
        self, query_plan: MongoQueryPlan
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        collection = _collection(self)
        rows = [
            _normalize_row(self, row)
            for row in collection.aggregate(query_plan.pipeline, allowDiskUse=True)
        ]
        return rows, _summarize(self, rows)

    async def interpret(
        self,
        *,
        prompt: str,
        model: str,
        worker_manager: WorkerManager,
        rows: List[Dict[str, Any]],
        summary: Dict[str, Any],
        query_plan: MongoQueryPlan,
        temperature: Optional[float],
        max_new_tokens: Optional[int],
        conv_uid: Optional[str],
    ) -> Tuple[str, Dict[str, int]]:
        model_output = await worker_manager.generate(
            _model_request_dict(
                app=self,
                prompt=prompt,
                rows=rows,
                summary=summary,
                query_plan=query_plan,
                model=model,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                conv_uid=conv_uid,
            )
        )
        if not model_output.success:
            _raise_model_error(model, model_output.text)
        answer = model_output.text.strip()
        if not answer:
            _raise_model_error(model, "empty response")
        return answer, _usage_from_model(model_output.usage)


class MongoChatDataRouter:
    def __init__(self, apps: Mapping[str, MongoChatDataApp]):
        self._apps = dict(apps)

    @classmethod
    def from_env(cls) -> "MongoChatDataRouter":
        raw = _load_config_from_env()
        apps_raw = raw.get("apps") if isinstance(raw, Mapping) else None
        if not isinstance(apps_raw, Mapping):
            apps_raw = {}
        apps = {}
        for name, app_raw in apps_raw.items():
            if isinstance(app_raw, Mapping):
                app = MongoChatDataApp.from_mapping(str(name), app_raw)
                app.validate()
                apps[app.name] = app
        return cls(apps)

    def can_handle(self, chat_param: Optional[str]) -> bool:
        return bool(chat_param and chat_param in self._apps)

    async def answer(
        self,
        *,
        chat_param: str,
        prompt: str,
        model: str,
        worker_manager: WorkerManager,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        conv_uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        app = self._apps[chat_param]
        query_plan, plan_usage = await app.plan_query(
            prompt=prompt,
            model=model,
            worker_manager=worker_manager,
            conv_uid=conv_uid,
        )
        rows, summary = app.query(query_plan)
        answer, usage = await app.interpret(
            prompt=prompt,
            model=model,
            worker_manager=worker_manager,
            rows=rows,
            summary=summary,
            query_plan=query_plan,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            conv_uid=conv_uid,
        )
        combined_usage = _merge_usage(plan_usage, usage)
        query_plan_payload = _query_plan_payload(app, query_plan)
        provenance = _data_provenance(
            app=app,
            planner_schema=query_plan.planner_schema,
            query_plan=query_plan_payload,
            rows=rows,
        )
        return {
            "id": conv_uid or f"mongo-chat-data-{chat_param}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": answer}}
            ],
            "artifact": {
                "type": "mongo.data.result",
                "source": app.source,
                "chat_param": chat_param,
                "summary": summary,
                "queryPlan": query_plan_payload,
                "rows": rows,
            },
            "raw": {
                "source": app.source,
                "chat_param": chat_param,
                "summary": summary,
                "queryPlan": query_plan_payload,
                "rows": rows,
            },
            "provenance": provenance,
            "usage": combined_usage
            or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


def _load_config_from_env() -> Dict[str, Any]:
    config_json = os.getenv(CONFIG_JSON_ENV)
    if config_json:
        return json.loads(config_json)
    config_file = os.getenv(CONFIG_FILE_ENV)
    if config_file:
        return json.loads(Path(config_file).read_text(encoding="utf-8"))
    return {}


def _resolve_env(value: Any) -> str:
    text = str(value or "")
    pattern = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

    def repl(match: re.Match[str]) -> str:
        env_name = match.group(1)
        fallback = match.group(2) or ""
        return os.getenv(env_name, fallback)

    return pattern.sub(repl, text)


def _extract_time_window(prompt: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    dates = []
    for token in re.findall(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?", prompt or ""
    ):
        dates.append(_parse_datetime(token))
    dates = [dt for dt in dates if dt is not None]
    if len(dates) >= 2:
        return dates[0], dates[1]
    return None, None


def _parse_datetime(value: str) -> Optional[datetime]:
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _round(value: Any, digits: int = 3) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(float(value), digits)
    return value


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _sum_numeric_values(values: Iterable[Any]) -> Any:
    total = 0.0
    saw_number = False
    saw_float = False
    for value in values:
        number = _number(value)
        if number is None:
            continue
        total += number
        saw_number = True
        if isinstance(value, float) and not value.is_integer():
            saw_float = True
    if not saw_number:
        return None
    if not saw_float and total.is_integer():
        return int(total)
    return _round(total)


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def _canonical_json(data: Any) -> bytes:
    return json.dumps(
        _jsonable(data),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(data: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(data)).hexdigest()


def _data_provenance(
    *,
    app: MongoChatDataApp,
    planner_schema: Mapping[str, Any],
    query_plan: Mapping[str, Any],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not planner_schema:
        raise ValueError("Mongo chat_data planner schema provenance is unavailable")
    return {
        "contractVersion": DATA_PROVENANCE_CONTRACT_VERSION,
        "source": app.source,
        "schemaFingerprint": _fingerprint(planner_schema),
        "compiledPlanFingerprint": _fingerprint(query_plan),
        "resultFingerprint": _fingerprint({"source": app.source, "rows": rows}),
        "rowCount": len(rows),
    }


def validate_data_provenance(
    provenance: Any,
    *,
    source: str,
    row_count: int,
) -> Dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("Mongo chat_data response provenance is not an object")
    if set(provenance) != _DATA_PROVENANCE_FIELDS:
        raise ValueError("Mongo chat_data response provenance fields are incomplete")
    if provenance.get("contractVersion") != DATA_PROVENANCE_CONTRACT_VERSION:
        raise ValueError("Mongo chat_data response provenance version is unsupported")
    if provenance.get("source") != source:
        raise ValueError("Mongo chat_data response provenance source is unbound")
    receipt_row_count = provenance.get("rowCount")
    if (
        isinstance(receipt_row_count, bool)
        or not isinstance(receipt_row_count, int)
        or receipt_row_count != row_count
    ):
        raise ValueError("Mongo chat_data response provenance row count is unbound")
    for field in (
        "schemaFingerprint",
        "compiledPlanFingerprint",
        "resultFingerprint",
    ):
        value = provenance.get(field)
        if not isinstance(value, str) or not _SHA256_FINGERPRINT.fullmatch(value):
            raise ValueError(f"Mongo chat_data response provenance {field} is invalid")
    return {
        field: provenance[field]
        for field in (
            "contractVersion",
            "source",
            "schemaFingerprint",
            "compiledPlanFingerprint",
            "resultFingerprint",
            "rowCount",
        )
    }


def _find_user_question(prompt: str) -> str:
    text = (prompt or "").strip()
    if not text:
        return ""
    marker = "\n\n"
    if marker in text:
        return text.split(marker, 1)[0].strip()
    return text


def _build_query_plan_prompt(
    app: MongoChatDataApp,
    prompt: str,
    planner_schema: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    schema = dict(planner_schema or _planner_schema(app))
    return [
        {
            "role": ModelMessageRoleType.HUMAN,
            "content": (
                "你是 DB-GPT chat_data 查询规划器。请根据用户问题和 Mongo schema 生成"
                "只读 Mongo aggregation 查询计划。\n\n"
                "硬性规则：\n"
                "1. 只返回 JSON，不要 Markdown，不要解释。\n"
                '2. JSON 结构必须是 {"pipeline": [...], "reason": "..."}。\n'
                "3. pipeline 只能使用 schema.allowedStages 中的 stage，不能使用 $out、"
                "$merge、$lookup、$function、$where 或写入类操作。\n"
                "4. 只能引用 schema.sourceFields 中存在的源字段；如果用户上下文提到"
                " factory/line/asset 但 schema 没有对应字段，不要添加这些过滤条件。\n"
                "5. 如果用户 prompt 中提供时间窗，必须在 timeField 上使用这个时间窗。\n"
                "6. 聚合输出字段名优先使用 schema.metrics.name，"
                "便于下游生成稳定 artifact。\n\n"
                f"Mongo schema（JSON）：{_compact_json(schema)}\n\n"
                f"用户 prompt：{prompt}\n\n"
                "请只返回 JSON。"
            ),
        }
    ]


def _planner_schema(app: MongoChatDataApp) -> Dict[str, Any]:
    return {
        "chatDataApp": app.name,
        "source": app.source,
        "database": app.database,
        "collection": app.collection,
        "timeField": app.time_field,
        "groupField": app.group_field,
        "groupLabel": app.group_label,
        "rowLimit": app.row_limit,
        "sourceFields": _source_fields(app),
        "filterFields": app.filter_fields,
        "metrics": [
            {
                "name": spec.name,
                "op": spec.op,
                "field": spec.field,
                "fields": spec.fields,
            }
            for spec in app.metrics
        ],
        "derived": [
            {
                "name": spec.name,
                "op": spec.op,
                "field": spec.field,
                "fields": spec.fields,
            }
            for spec in app.derived
        ],
        "allowedStages": sorted(_ALLOWED_STAGES),
        "allowedMatchOperators": sorted(_ALLOWED_MATCH_OPERATORS),
        "allowedGroupOperators": sorted(_ALLOWED_GROUP_OPERATORS),
        "allowedExpressionOperators": sorted(_ALLOWED_EXPRESSION_OPERATORS),
    }


def _parse_query_plan(text: str) -> MongoQueryPlan:
    raw = _parse_json_object(text)
    pipeline = raw.get("pipeline")
    if not isinstance(pipeline, list):
        raise ValueError("Mongo chat_data query plan must contain a pipeline list")
    for stage in pipeline:
        if not isinstance(stage, Mapping):
            raise ValueError("Mongo chat_data pipeline stages must be objects")
    reason = raw.get("reason")
    return MongoQueryPlan(
        pipeline=[dict(stage) for stage in pipeline],
        reason=str(reason or ""),
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    candidate = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S | re.I)
    if fenced:
        candidate = fenced.group(1)
    elif not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("Mongo chat_data model did not return valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Mongo chat_data query plan must be a JSON object")
    return parsed


def _guard_pipeline(
    app: MongoChatDataApp, pipeline: List[Dict[str, Any]], prompt: str
) -> List[Dict[str, Any]]:
    guarded = [_copy_stage(stage) for stage in pipeline]
    start, end = _extract_time_window(prompt)
    if start and end:
        time_clause = {app.time_field: {"$gte": start, "$lte": end}}
        if guarded and "$match" in guarded[0]:
            match = dict(guarded[0]["$match"])
            match[app.time_field] = time_clause[app.time_field]
            guarded[0] = {"$match": match}
        else:
            guarded.insert(0, {"$match": time_clause})
    return _cap_pipeline_limit(app, guarded)


def _copy_stage(stage: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(stage, default=str))


def _cap_pipeline_limit(
    app: MongoChatDataApp, pipeline: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    capped: List[Dict[str, Any]] = []
    saw_limit = False
    for stage in pipeline:
        if "$limit" in stage:
            if not saw_limit:
                capped.append(
                    {"$limit": min(_int_limit(stage["$limit"]), app.row_limit)}
                )
                saw_limit = True
            continue
        capped.append(stage)
    if not saw_limit:
        capped.append({"$limit": app.row_limit})
    return capped


def _int_limit(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Mongo chat_data $limit must be an integer")
    limit = int(value)
    if limit < 1:
        raise ValueError("Mongo chat_data $limit must be positive")
    return limit


def _validate_pipeline(app: MongoChatDataApp, pipeline: List[Dict[str, Any]]) -> None:
    if not pipeline:
        raise ValueError("Mongo chat_data query plan pipeline is empty")
    available_fields = set(_source_fields(app))
    for stage in pipeline:
        if not isinstance(stage, Mapping) or len(stage) != 1:
            raise ValueError(
                "Mongo chat_data pipeline stages must be single-key objects"
            )
        stage_name, payload = next(iter(stage.items()))
        if stage_name not in _ALLOWED_STAGES:
            raise ValueError(f"Mongo chat_data stage {stage_name!r} is not allowed")
        if stage_name == "$match":
            _validate_match(app, payload)
        elif stage_name == "$sort":
            _validate_sort(app, payload, available_fields)
        elif stage_name == "$group":
            _validate_group(app, payload, available_fields)
            available_fields = {"_id"} | set(_metric_output_fields(app))
        elif stage_name == "$addFields":
            new_fields = _validate_add_fields(app, payload, available_fields)
            available_fields.update(new_fields)
        elif stage_name == "$project":
            available_fields = _validate_project(app, payload, available_fields)
        elif stage_name == "$limit":
            limit = _int_limit(payload)
            if limit > app.row_limit:
                raise ValueError("Mongo chat_data $limit exceeds configured row limit")


def _validate_match(app: MongoChatDataApp, match: Any) -> None:
    if not isinstance(match, Mapping):
        raise ValueError("Mongo chat_data $match must be an object")
    allowed_fields = set(_source_fields(app))
    for field_name, value in match.items():
        if str(field_name).startswith("$"):
            raise ValueError("Mongo chat_data logical $match operators are not allowed")
        if field_name not in allowed_fields:
            raise ValueError(
                f"Mongo chat_data $match field {field_name!r} is not in schema"
            )
        if isinstance(value, Mapping):
            for operator, operator_value in value.items():
                if operator not in _ALLOWED_MATCH_OPERATORS:
                    raise ValueError(
                        f"Mongo chat_data $match operator {operator!r} is not allowed"
                    )
                _validate_match_literal(operator_value)
        else:
            _validate_match_literal(value)


def _validate_match_literal(value: Any) -> None:
    if isinstance(value, Mapping):
        raise ValueError("Mongo chat_data nested $match values are not allowed")
    if isinstance(value, list):
        for item in value:
            _validate_match_literal(item)


def _validate_sort(
    app: MongoChatDataApp, sort: Any, available_fields: Optional[Iterable[str]] = None
) -> None:
    if not isinstance(sort, Mapping):
        raise ValueError("Mongo chat_data $sort must be an object")
    allowed_fields = set(available_fields or _source_fields(app)) | {"_id"}
    for field_name, direction in sort.items():
        if field_name not in allowed_fields:
            raise ValueError(
                f"Mongo chat_data $sort field {field_name!r} is not in schema"
            )
        if direction not in (1, -1):
            raise ValueError("Mongo chat_data $sort direction must be 1 or -1")


def _validate_group(
    app: MongoChatDataApp, group: Any, available_fields: Optional[Iterable[str]] = None
) -> None:
    if not isinstance(group, Mapping):
        raise ValueError("Mongo chat_data $group must be an object")
    if "_id" not in group:
        raise ValueError("Mongo chat_data $group must include _id")
    _validate_expression(app, group.get("_id"), available_fields)
    metric_fields = set(_metric_output_fields(app))
    for field_name, expression in group.items():
        if field_name == "_id":
            continue
        if field_name not in metric_fields:
            raise ValueError(
                f"Mongo chat_data $group output {field_name!r} is not a "
                "configured metric"
            )
        if not isinstance(expression, Mapping) or len(expression) != 1:
            raise ValueError(
                "Mongo chat_data $group metric expressions must be one-op objects"
            )
        operator = next(iter(expression.keys()))
        if operator not in _ALLOWED_GROUP_OPERATORS:
            raise ValueError(
                f"Mongo chat_data $group operator {operator!r} is not allowed"
            )
        _validate_expression(app, expression, available_fields)


def _validate_add_fields(
    app: MongoChatDataApp, add_fields: Any, available_fields: Iterable[str]
) -> List[str]:
    if not isinstance(add_fields, Mapping):
        raise ValueError("Mongo chat_data $addFields must be an object")
    allowed_outputs = set(_metric_output_fields(app)) | set(_derived_output_fields(app))
    if app.group_label:
        allowed_outputs.add(app.group_label)
    if app.group_field:
        allowed_outputs.add(app.group_field)
    added = []
    for field_name, expression in add_fields.items():
        if str(field_name).startswith("$") or "." in str(field_name):
            raise ValueError(
                f"Mongo chat_data $addFields output {field_name!r} is not allowed"
            )
        if field_name not in allowed_outputs:
            raise ValueError(
                f"Mongo chat_data $addFields output {field_name!r} is not configured"
            )
        _validate_expression(app, expression, available_fields)
        added.append(str(field_name))
    return added


def _validate_project(
    app: MongoChatDataApp, project: Any, available_fields: Iterable[str]
) -> set[str]:
    if not isinstance(project, Mapping) or not project:
        raise ValueError("Mongo chat_data $project must be a non-empty object")
    available = set(available_fields)
    allowed_outputs = (
        available
        | set(_metric_output_fields(app))
        | set(_derived_output_fields(app))
        | {app.group_field, app.group_label, "_id"}
    )
    included: set[str] = set()
    excluded: set[str] = set()
    has_inclusion = False
    has_exclusion = False
    for raw_name, expression in project.items():
        field_name = str(raw_name)
        if (
            field_name.startswith("$")
            or "." in field_name
            or field_name not in allowed_outputs
        ):
            raise ValueError(
                f"Mongo chat_data $project output {field_name!r} is not configured"
            )
        if (
            isinstance(expression, int)
            and not isinstance(expression, bool)
            and expression in (0, 1)
        ):
            if expression == 0:
                excluded.add(field_name)
                if field_name != "_id":
                    has_exclusion = True
                continue
            if field_name not in available:
                raise ValueError(
                    f"Mongo chat_data $project inclusion {field_name!r} "
                    "is not available"
                )
            has_inclusion = True
            included.add(field_name)
            continue
        has_inclusion = True
        _validate_expression(app, expression, available)
        included.add(field_name)
    if has_inclusion and has_exclusion:
        raise ValueError("Mongo chat_data $project cannot mix inclusion and exclusion")
    if has_inclusion and "_id" in available and project.get("_id") != 0:
        included.add("_id")
    retained = included if has_inclusion else available - excluded
    required_metrics = set(_metric_output_fields(app)) & available
    missing_metrics = sorted(required_metrics - retained)
    if missing_metrics:
        raise ValueError(
            "Mongo chat_data $project must retain configured metric fields: "
            + ", ".join(missing_metrics)
        )
    if app.group_field and "_id" in available and "_id" not in retained:
        identity_aliases = {app.group_label, app.group_field}
        if not any(project.get(alias) == "$_id" for alias in identity_aliases):
            raise ValueError(
                "Mongo chat_data $project cannot remove group identity "
                "without a configured identity alias"
            )
    if has_inclusion:
        return included
    return available - excluded


def _validate_expression(
    app: MongoChatDataApp,
    expression: Any,
    available_fields: Optional[Iterable[str]] = None,
) -> None:
    allowed_fields = set(available_fields or _source_fields(app)) | {"_id"}
    if isinstance(expression, str):
        if expression.startswith("$") and expression[1:] not in allowed_fields:
            raise ValueError(
                f"Mongo chat_data field reference {expression!r} is not in schema"
            )
        return
    if isinstance(expression, list):
        for item in expression:
            _validate_expression(app, item, allowed_fields)
        return
    if isinstance(expression, Mapping):
        for key, value in expression.items():
            if str(key).startswith("$") and key not in _ALLOWED_EXPRESSION_OPERATORS:
                raise ValueError(
                    f"Mongo chat_data expression operator {key!r} is not allowed"
                )
            _validate_expression(app, value, allowed_fields)


def _source_fields(app: MongoChatDataApp) -> List[str]:
    fields = {app.time_field}
    if app.group_field:
        fields.add(app.group_field)
    fields.update(app.filter_fields)
    for spec in app.metrics:
        if spec.field:
            fields.add(spec.field)
        fields.update(spec.fields)
    return sorted(field for field in fields if field)


def _metric_output_fields(app: MongoChatDataApp) -> List[str]:
    return [name for name in _iter_metric_names(app.metrics) if name]


def _derived_output_fields(app: MongoChatDataApp) -> List[str]:
    return [name for name in _iter_metric_names(app.derived) if name]


def _query_plan_payload(
    app: MongoChatDataApp, query_plan: MongoQueryPlan
) -> Dict[str, Any]:
    return {
        "source": app.source,
        "database": app.database,
        "collection": app.collection,
        "pipeline": _jsonable(query_plan.pipeline),
        "reason": query_plan.reason,
        "generatedBy": "llm",
    }


def _iter_metric_names(specs: Iterable[MongoMetricSpec]) -> List[str]:
    names = []
    for spec in specs:
        if spec.name:
            names.append(spec.name)
    return names


def _derived_value(spec: MongoMetricSpec, row: Mapping[str, Any]) -> Any:
    if spec.op == "sumFields":
        return _sum_numeric_values(row.get(name) or 0 for name in spec.fields)
    if spec.op == "firstNonEmpty":
        for name in spec.fields:
            value = row.get(name)
            if value not in (None, ""):
                return value
    return None


def _summary_value(spec: MongoSummaryMetricSpec, rows: List[Dict[str, Any]]) -> Any:
    if spec.op == "countRows":
        return len(rows)
    if spec.op == "sum" and spec.field:
        return _sum_numeric_values(row.get(spec.field) for row in rows)
    if spec.op == "weightedAvg" and spec.field and spec.weight:
        total = 0.0
        weight = 0.0
        for row in rows:
            value = _number(row.get(spec.field))
            row_weight = _number(row.get(spec.weight))
            if value is not None and row_weight and row_weight > 0:
                total += value * row_weight
                weight += row_weight
        return _round(total / weight) if weight > 0 else None
    if spec.op == "firstNonEmpty" and spec.field:
        for row in rows:
            value = row.get(spec.field)
            if value not in (None, ""):
                return value
    return None


def _usage_from_model(model_usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    usage = model_usage or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _merge_usage(*usages: Dict[str, int]) -> Dict[str, int]:
    merged = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in usages:
        for key in merged:
            merged[key] += int((usage or {}).get(key) or 0)
    return merged


def _collection(app: MongoChatDataApp):
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed in DB-GPT runtime")
    client = MongoClient(app.uri, serverSelectionTimeoutMS=5000, socketTimeoutMS=15000)
    client.admin.command("ping")
    return client[app.database][app.collection]


def _normalize_row(app: MongoChatDataApp, row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(app.row_defaults)
    if app.group_field:
        identity = row.get("_id")
        for alias in (app.group_label, app.group_field):
            if identity not in (None, ""):
                break
            identity = row.get(alias)
        out[app.group_label] = identity if identity not in (None, "") else "unknown"
    for name in _iter_metric_names(app.metrics):
        out[name] = _jsonable(_round(row.get(name)))
    for spec in app.derived:
        if spec.name:
            out[spec.name] = _jsonable(_round(_derived_value(spec, out)))
    return out


def _summarize(app: MongoChatDataApp, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "source": app.source,
        "database": app.database,
        "collection": app.collection,
        "row_count": len(rows),
    }
    for spec in app.summary_metrics:
        if spec.name:
            summary[spec.name] = _jsonable(_summary_value(spec, rows))
    return summary


def _build_interpret_prompt(
    app: MongoChatDataApp,
    prompt: str,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    query_plan: MongoQueryPlan,
) -> List[Dict[str, str]]:
    user_question = _find_user_question(prompt)
    system_prompt = app.prompt or (
        "你是一个严谨的数据分析助手。只能解释已提供的 Mongo 查询事实，"
        "不要编造没有出现在事实中的字段或结论。涉及执行类动作时，"
        "只能给出建议并明确需要人工确认。"
    )
    facts = {
        "chatDataApp": app.name,
        "source": app.source,
        "summary": summary,
        "executedQuery": _query_plan_payload(app, query_plan),
        "rows": rows,
    }
    return [
        {
            "role": ModelMessageRoleType.HUMAN,
            "content": (
                "任务：直接回答用户问题，不要复述规则，也不要声明已理解规则。\n\n"
                f"上下文约束（只遵守，不要复述）：{system_prompt}\n\n"
                f"用户问题：{user_question}\n\n"
                f"Mongo 查询事实（JSON）：{_compact_json(facts)}\n\n"
                "请用用户问题的语言直接回答，解释关键指标含义、当前数值和治理边界。"
                "如果提到查询、SQL 或 Mongo pipeline，只能引用 facts.executedQuery；"
                "不要编造未执行的过滤字段、SQL 或 Mongo 语句。"
            ),
        },
    ]


def _query_plan_request_dict(
    *,
    app: MongoChatDataApp,
    prompt: str,
    model: str,
    conv_uid: Optional[str],
    planner_schema: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    req = ModelRequest(
        model=model,
        messages=_build_query_plan_prompt(app, prompt, planner_schema),
        temperature=0.0,
        max_new_tokens=1024,
        span_id=f"{conv_uid}:query-plan" if conv_uid else None,
    )
    return req.to_dict()


def _model_request_dict(
    *,
    app: MongoChatDataApp,
    prompt: str,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    query_plan: MongoQueryPlan,
    model: str,
    temperature: Optional[float],
    max_new_tokens: Optional[int],
    conv_uid: Optional[str],
) -> Dict[str, Any]:
    req = ModelRequest(
        model=model,
        messages=_build_interpret_prompt(app, prompt, rows, summary, query_plan),
        temperature=temperature,
        max_new_tokens=max_new_tokens or 1024,
        span_id=conv_uid,
    )
    return req.to_dict()


def _raise_model_error(model: str, text: str) -> None:
    raise RuntimeError(f"Mongo chat_data model {model!r} failed: {text}")
