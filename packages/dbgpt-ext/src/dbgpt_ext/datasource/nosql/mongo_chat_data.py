"""Config-driven MongoDB chat_data runtime.

This module is intentionally project-neutral. Each project binds `chat_param` to
a Mongo database/collection through configuration; the runtime here only knows
how to read Mongo and hand the resulting facts to the requested DB-GPT model for
interpretation.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from dbgpt.core import ModelMessageRoleType, ModelRequest
from dbgpt.model.cluster.manager_base import WorkerManager

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover - reported as a runtime configuration gap.
    MongoClient = None  # type: ignore


CONFIG_FILE_ENV = "DBGPT_MONGO_CHAT_DATA_CONFIG_FILE"
CONFIG_JSON_ENV = "DBGPT_MONGO_CHAT_DATA_CONFIG"


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
class MongoChatDataApp:
    name: str
    uri: str
    database: str
    collection: str
    source: str
    time_field: str = "timestamp"
    group_field: str = ""
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
            time_field=str(raw.get("timeField") or raw.get("time_field") or "timestamp"),
            group_field=str(raw.get("groupField") or raw.get("group_field") or ""),
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
                f"Mongo chat_data app {self.name!r} missing config: {', '.join(missing)}"
            )
        if not self.metrics:
            raise ValueError(f"Mongo chat_data app {self.name!r} has no metrics")

    def query(self, prompt: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        collection = _collection(self)
        match = _match_window(self, prompt, collection)
        sort_metric = (
            "sample_count"
            if any(m.name == "sample_count" for m in self.metrics)
            else None
        )
        pipeline: List[Dict[str, Any]] = [
            {"$match": match},
            {"$sort": {self.time_field: 1}},
            {"$group": _build_group_stage(self)},
        ]
        if sort_metric:
            pipeline.append({"$sort": {sort_metric: -1}})
        pipeline.append({"$limit": self.row_limit})

        rows = [
            _normalize_row(self, row)
            for row in collection.aggregate(pipeline, allowDiskUse=True)
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
        rows, summary = app.query(prompt)
        answer, usage = await app.interpret(
            prompt=prompt,
            model=model,
            worker_manager=worker_manager,
            rows=rows,
            summary=summary,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            conv_uid=conv_uid,
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
                "rows": rows,
            },
            "raw": {
                "source": app.source,
                "chat_param": chat_param,
                "summary": summary,
                "rows": rows,
            },
            "usage": usage or {
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


def _find_user_question(prompt: str) -> str:
    text = (prompt or "").strip()
    if not text:
        return ""
    marker = "\n\n"
    if marker in text:
        return text.split(marker, 1)[0].strip()
    return text


def _iter_metric_names(specs: Iterable[MongoMetricSpec]) -> List[str]:
    names = []
    for spec in specs:
        if spec.name:
            names.append(spec.name)
    return names


def _pipeline_expr(spec: MongoMetricSpec, time_field: str) -> Optional[Dict[str, Any]]:
    if not spec.name:
        return None
    if spec.op == "count":
        return {"$sum": 1}
    if spec.op == "first":
        return {"$first": f"${spec.field or time_field}"}
    if spec.op == "last":
        return {"$last": f"${spec.field or time_field}"}
    if spec.op == "avg" and spec.field:
        return {"$avg": f"${spec.field}"}
    if spec.op == "min" and spec.field:
        return {"$min": f"${spec.field}"}
    if spec.op == "max" and spec.field:
        return {"$max": f"${spec.field}"}
    if spec.op == "stdDevPop" and spec.field:
        return {"$stdDevPop": f"${spec.field}"}
    if spec.op == "sumBoolTrue" and spec.field:
        return {"$sum": {"$cond": [{"$eq": [f"${spec.field}", True]}, 1, 0]}}
    if spec.op == "sum" and spec.field:
        return {"$sum": f"${spec.field}"}
    return None


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


def _collection(app: MongoChatDataApp):
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed in DB-GPT runtime")
    client = MongoClient(app.uri, serverSelectionTimeoutMS=5000, socketTimeoutMS=15000)
    client.admin.command("ping")
    return client[app.database][app.collection]


def _match_window(app: MongoChatDataApp, prompt: str, collection) -> Dict[str, Any]:
    start, end = _extract_time_window(prompt)
    if start and end:
        return {app.time_field: {"$gte": start, "$lte": end}}

    latest = collection.find_one(
        sort=[(app.time_field, -1)], projection={app.time_field: 1}
    )
    latest_ts = latest.get(app.time_field) if latest else None
    if isinstance(latest_ts, datetime):
        return {
            app.time_field: {
                "$gte": latest_ts - timedelta(hours=app.default_window_hours),
                "$lte": latest_ts,
            }
        }
    return {}


def _normalize_row(app: MongoChatDataApp, row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(app.row_defaults)
    if app.group_field:
        out[app.group_label] = row.get("_id") or "unknown"
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


def _build_group_stage(app: MongoChatDataApp) -> Dict[str, Any]:
    group: Dict[str, Any] = {"_id": f"${app.group_field}" if app.group_field else None}
    for spec in app.metrics:
        expr = _pipeline_expr(spec, app.time_field)
        if expr is not None:
            group[spec.name] = expr
    return group


def _build_interpret_prompt(
    app: MongoChatDataApp, prompt: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]
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
            ),
        },
    ]


def _model_request_dict(
    *,
    app: MongoChatDataApp,
    prompt: str,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    model: str,
    temperature: Optional[float],
    max_new_tokens: Optional[int],
    conv_uid: Optional[str],
) -> Dict[str, Any]:
    req = ModelRequest(
        model=model,
        messages=_build_interpret_prompt(app, prompt, rows, summary),
        temperature=temperature,
        max_new_tokens=max_new_tokens or 1024,
        span_id=conv_uid,
    )
    return req.to_dict()


def _raise_model_error(model: str, text: str) -> None:
    raise RuntimeError(f"Mongo chat_data model {model!r} failed: {text}")
