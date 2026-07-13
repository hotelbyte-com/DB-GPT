import asyncio
import hashlib
import json

import pytest

from dbgpt_ext.datasource.nosql import mongo_chat_data


def _fingerprint(value):
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


class FakeModelOutput:
    def __init__(self, text, usage=None, success=True):
        self.text = text
        self.usage = usage or {}
        self.success = success


class FakeWorkerManager:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return self.outputs.pop(0)


def test_router_loads_project_neutral_mongo_apps(monkeypatch):
    monkeypatch.setenv("MONGO_TEST_DB", "runtime_itdu")
    monkeypatch.setenv(
        "DBGPT_MONGO_CHAT_DATA_CONFIG",
        """
        {
          "apps": {
            "manufacturing": {
              "uri": "${env:MONGO_TEST_URI:-mongodb://127.0.0.1:27017}",
              "database": "${env:MONGO_TEST_DB:-ITDU}",
              "collection": "ITDU_PLCData",
              "source": "mongodb.ITDU.ITDU_PLCData",
              "timeField": "timestamp",
              "groupField": "machineId",
              "filterFields": ["factory"],
              "metrics": [{"name": "sample_count", "op": "count"}]
            },
            "hotel": {
              "uri": "mongodb://127.0.0.1:27017",
              "database": "hotel",
              "collection": "orders",
              "source": "mongodb.hotel.orders",
              "metrics": [{"name": "sample_count", "op": "count"}]
            }
          }
        }
        """,
    )

    router = mongo_chat_data.MongoChatDataRouter.from_env()

    assert router.can_handle("manufacturing")
    assert router.can_handle("hotel")
    assert not router.can_handle("unknown")
    assert router._apps["manufacturing"].database == "runtime_itdu"
    assert router._apps["manufacturing"].filter_fields == ["factory"]


def test_configured_app_executes_llm_query_plan_with_runtime_guards(monkeypatch):
    app = mongo_chat_data.MongoChatDataApp.from_mapping(
        "manufacturing",
        {
            "uri": "mongodb://127.0.0.1:27017",
            "database": "ITDU",
            "collection": "ITDU_PLCData",
            "source": "mongodb.ITDU.ITDU_PLCData",
            "timeField": "timestamp",
            "groupField": "machineId",
            "groupLabel": "station",
            "rowDefaults": {"line": "ITDU"},
            "metrics": [
                {"name": "sample_count", "op": "count"},
                {"name": "avg_thickness", "op": "avg", "field": "AvgThk"},
                {"name": "std_thickness", "op": "stdDevPop", "field": "AvgThk"},
                {
                    "name": "spa000_bool_count",
                    "op": "sumBoolTrue",
                    "field": "Spa000_BOOL",
                },
            ],
            "derived": [
                {
                    "name": "alarm_count",
                    "op": "sumFields",
                    "fields": ["spa000_bool_count"],
                }
            ],
            "summaryMetrics": [
                {"name": "sample_count", "op": "sum", "field": "sample_count"},
                {
                    "name": "avg_thickness",
                    "op": "weightedAvg",
                    "field": "avg_thickness",
                    "weight": "sample_count",
                },
            ],
        },
    )

    class FakeCollection:
        def __init__(self):
            self.pipeline = None

        def aggregate(self, pipeline, allowDiskUse):
            self.pipeline = pipeline
            assert allowDiskUse is True
            return [
                {
                    "_id": "Pur_Aoi",
                    "sample_count": 438758,
                    "avg_thickness": 25.308,
                    "std_thickness": 6.484,
                    "spa000_bool_count": 0,
                }
            ]

    fake = FakeCollection()
    monkeypatch.setattr(mongo_chat_data, "_collection", lambda _: fake)
    worker_manager = FakeWorkerManager(
        [
            FakeModelOutput(
                """
                {
                  "pipeline": [
                    {"$match": {"machineId": {"$in": ["Pur_Aoi"]}}},
                    {"$sort": {"timestamp": 1}},
                    {
                      "$group": {
                        "_id": "$machineId",
                        "sample_count": {"$sum": 1},
                        "avg_thickness": {"$avg": "$AvgThk"},
                        "std_thickness": {"$stdDevPop": "$AvgThk"},
                        "spa000_bool_count": {
                          "$sum": {"$cond": [{"$eq": ["$Spa000_BOOL", true]}, 1, 0]}
                        }
                      }
                    },
                    {
                      "$addFields": {
                        "machineId": "$_id",
                        "alarm_count": {"$add": ["$spa000_bool_count", 0]}
                      }
                    },
                    {"$sort": {"alarm_count": -1}},
                    {"$limit": 999}
                  ],
                  "reason": "Group requested equipment readings by machineId."
                }
                """,
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ]
    )

    query_plan, usage = asyncio.run(
        app.plan_query(
            prompt="timeWindow=2026-06-11T05:38:07Z..2026-06-16T08:51:12Z",
            model="MiniMax-M3",
            worker_manager=worker_manager,
            conv_uid="run-1",
        )
    )
    rows, summary = app.query(query_plan)

    assert fake.pipeline[0] == {
        "$match": {
            "machineId": {"$in": ["Pur_Aoi"]},
            "timestamp": {
                "$gte": mongo_chat_data._parse_datetime("2026-06-11T05:38:07Z"),
                "$lte": mongo_chat_data._parse_datetime("2026-06-16T08:51:12Z"),
            },
        }
    }
    assert fake.pipeline[2]["$group"]["avg_thickness"] == {"$avg": "$AvgThk"}
    assert fake.pipeline[3]["$addFields"] == {
        "machineId": "$_id",
        "alarm_count": {"$add": ["$spa000_bool_count", 0]},
    }
    assert fake.pipeline[4] == {"$sort": {"alarm_count": -1}}
    assert fake.pipeline[-1] == {"$limit": 10}
    assert rows == [
        {
            "line": "ITDU",
            "station": "Pur_Aoi",
            "sample_count": 438758,
            "avg_thickness": 25.308,
            "std_thickness": 6.484,
            "spa000_bool_count": 0,
            "alarm_count": 0,
        }
    ]
    assert summary["source"] == "mongodb.ITDU.ITDU_PLCData"
    assert summary["sample_count"] == 438758
    assert summary["avg_thickness"] == 25.308
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert "Group requested equipment readings" in query_plan.reason
    assert len(worker_manager.requests) == 1


def test_query_plan_rejects_unconfigured_line_filter():
    app = mongo_chat_data.MongoChatDataApp.from_mapping(
        "manufacturing",
        {
            "uri": "mongodb://127.0.0.1:27017",
            "database": "ITDU",
            "collection": "ITDU_PLCData",
            "source": "mongodb.ITDU.ITDU_PLCData",
            "timeField": "timestamp",
            "groupField": "machineId",
            "metrics": [{"name": "sample_count", "op": "count"}],
        },
    )
    worker_manager = FakeWorkerManager(
        [
            FakeModelOutput(
                """
                {
                  "pipeline": [
                    {"$match": {"line": "L1"}},
                    {"$group": {"_id": "$machineId", "sample_count": {"$sum": 1}}}
                  ],
                  "reason": "Wrongly filtered by line."
                }
                """
            )
        ]
    )

    with pytest.raises(ValueError, match="line"):
        asyncio.run(
            app.plan_query(
                prompt=(
                    "factory=F1 line=L1 "
                    "timeWindow=2026-06-11T05:38:07Z..2026-06-16T08:51:12Z"
                ),
                model="MiniMax-M3",
                worker_manager=worker_manager,
                conv_uid="run-2",
            )
        )


def test_query_plan_rejects_unconfigured_add_fields_output():
    app = mongo_chat_data.MongoChatDataApp.from_mapping(
        "manufacturing",
        {
            "uri": "mongodb://127.0.0.1:27017",
            "database": "ITDU",
            "collection": "ITDU_PLCData",
            "source": "mongodb.ITDU.ITDU_PLCData",
            "timeField": "timestamp",
            "groupField": "machineId",
            "metrics": [{"name": "sample_count", "op": "count"}],
        },
    )
    worker_manager = FakeWorkerManager(
        [
            FakeModelOutput(
                """
                {
                  "pipeline": [
                    {"$group": {"_id": "$machineId", "sample_count": {"$sum": 1}}},
                    {"$addFields": {"line": "L1"}}
                  ],
                  "reason": "Wrongly invented a line field."
                }
                """
            )
        ]
    )

    with pytest.raises(ValueError, match="line"):
        asyncio.run(
            app.plan_query(
                prompt="factory=F1 line=L1",
                model="MiniMax-M3",
                worker_manager=worker_manager,
                conv_uid="run-4",
            )
        )


def test_query_plan_allows_configured_metric_add_fields_output():
    app = mongo_chat_data.MongoChatDataApp.from_mapping(
        "manufacturing",
        {
            "uri": "mongodb://127.0.0.1:27017",
            "database": "ITDU",
            "collection": "ITDU_PLCData",
            "source": "mongodb.ITDU.ITDU_PLCData",
            "timeField": "timestamp",
            "groupField": "machineId",
            "metrics": [
                {"name": "sample_count", "op": "count"},
                {
                    "name": "spa000_bool_count",
                    "op": "sumBoolTrue",
                    "field": "Spa000_BOOL",
                },
            ],
        },
    )
    worker_manager = FakeWorkerManager(
        [
            FakeModelOutput(
                """
                {
                  "pipeline": [
                    {
                      "$addFields": {
                        "spa000_bool_count": {
                          "$cond": [{"$eq": ["$Spa000_BOOL", true]}, 1, 0]
                        }
                      }
                    },
                    {
                      "$group": {
                        "_id": "$machineId",
                        "sample_count": {"$sum": 1},
                        "spa000_bool_count": {"$sum": "$spa000_bool_count"}
                      }
                    }
                  ],
                  "reason": "Count configured PLC boolean alarms by machine."
                }
                """
            )
        ]
    )

    query_plan, _ = asyncio.run(
        app.plan_query(
            prompt="count current PLC boolean alarms",
            model="glm5.2",
            worker_manager=worker_manager,
            conv_uid="run-configured-metric-add-fields",
        )
    )

    assert query_plan.pipeline[0]["$addFields"]["spa000_bool_count"] == {
        "$cond": [{"$eq": ["$Spa000_BOOL", True]}, 1, 0]
    }


def test_router_answer_returns_executed_query_plan_artifact(monkeypatch):
    app = mongo_chat_data.MongoChatDataApp.from_mapping(
        "manufacturing",
        {
            "uri": "mongodb://127.0.0.1:27017",
            "database": "ITDU",
            "collection": "ITDU_PLCData",
            "source": "mongodb.ITDU.ITDU_PLCData",
            "timeField": "timestamp",
            "groupField": "machineId",
            "groupLabel": "station",
            "metrics": [{"name": "sample_count", "op": "count"}],
        },
    )
    router = mongo_chat_data.MongoChatDataRouter({"manufacturing": app})

    class FakeCollection:
        def aggregate(self, pipeline, allowDiskUse):
            assert allowDiskUse is True
            assert pipeline[-1] == {"$limit": 10}
            return [{"_id": "Pur_Aoi", "sample_count": 2}]

    monkeypatch.setattr(mongo_chat_data, "_collection", lambda _: FakeCollection())
    worker_manager = FakeWorkerManager(
        [
            FakeModelOutput(
                """
                {
                  "pipeline": [
                    {
                      "$group": {
                        "_id": "$machineId",
                        "sample_count": {"$sum": 1}
                      }
                    }
                  ],
                  "reason": "Count rows by machine."
                }
                """,
                {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            ),
            FakeModelOutput(
                "Pur_Aoi 有 2 条样本。",
                {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
            ),
        ]
    )

    response = asyncio.run(
        router.answer(
            chat_param="manufacturing",
            prompt="timeWindow=2026-06-11T05:38:07Z..2026-06-16T08:51:12Z",
            model="MiniMax-M3",
            worker_manager=worker_manager,
            conv_uid="run-3",
        )
    )

    query_plan = response["artifact"]["queryPlan"]
    assert response["choices"][0]["message"]["content"] == "Pur_Aoi 有 2 条样本。"
    assert response["raw"]["queryPlan"] == query_plan
    assert query_plan["generatedBy"] == "llm"
    assert query_plan["pipeline"][0]["$match"]["timestamp"] == {
        "$gte": "2026-06-11T05:38:07Z",
        "$lte": "2026-06-16T08:51:12Z",
    }
    assert query_plan["pipeline"][1]["$group"]["sample_count"] == {"$sum": 1}
    assert response["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 10,
        "total_tokens": 18,
    }
    planner_schema = {
        "chatDataApp": "manufacturing",
        "source": "mongodb.ITDU.ITDU_PLCData",
        "database": "ITDU",
        "collection": "ITDU_PLCData",
        "timeField": "timestamp",
        "groupField": "machineId",
        "groupLabel": "station",
        "rowLimit": 10,
        "sourceFields": ["machineId", "timestamp"],
        "filterFields": [],
        "metrics": [{"name": "sample_count", "op": "count", "field": "", "fields": []}],
        "derived": [],
        "allowedStages": sorted(mongo_chat_data._ALLOWED_STAGES),
        "allowedMatchOperators": sorted(mongo_chat_data._ALLOWED_MATCH_OPERATORS),
        "allowedGroupOperators": sorted(mongo_chat_data._ALLOWED_GROUP_OPERATORS),
        "allowedExpressionOperators": sorted(
            mongo_chat_data._ALLOWED_EXPRESSION_OPERATORS
        ),
    }
    expected_result = {
        "source": "mongodb.ITDU.ITDU_PLCData",
        "rows": [{"station": "Pur_Aoi", "sample_count": 2}],
    }
    assert response["provenance"] == {
        "contractVersion": "data-provenance.v1",
        "source": "mongodb.ITDU.ITDU_PLCData",
        "schemaFingerprint": _fingerprint(planner_schema),
        "compiledPlanFingerprint": _fingerprint(query_plan),
        "resultFingerprint": _fingerprint(expected_result),
        "rowCount": 1,
    }
    assert len(worker_manager.requests) == 2


def test_interpret_prompt_keeps_executed_query_as_only_query_source():
    app = mongo_chat_data.MongoChatDataApp.from_mapping(
        "manufacturing",
        {
            "uri": "mongodb://127.0.0.1:27017",
            "database": "ITDU",
            "collection": "ITDU_PLCData",
            "source": "mongodb.ITDU.ITDU_PLCData",
            "metrics": [{"name": "sample_count", "op": "count"}],
            "prompt": "只能解释事实，执行动作需要人工确认。",
        },
    )

    messages = mongo_chat_data._build_interpret_prompt(
        app,
        "冷辊速度是什么意思？",
        [{"station": "Pur_Aoi", "avg_chill_speed": 52.006}],
        {"source": "mongodb.ITDU.ITDU_PLCData", "avg_chill_speed": 52.006},
        mongo_chat_data.MongoQueryPlan(
            pipeline=[
                {"$group": {"_id": "$machineId", "sample_count": {"$sum": 1}}},
                {"$limit": 10},
            ],
            reason="Group by machineId.",
        ),
    )

    assert [message["role"] for message in messages] == [
        mongo_chat_data.ModelMessageRoleType.HUMAN
    ]
    content = messages[0]["content"]
    assert "直接回答用户问题，不要复述规则" in content
    assert "上下文约束（只遵守，不要复述）" in content
    assert "用户问题：冷辊速度是什么意思？" in content
    assert "avg_chill_speed" in content
    assert "executedQuery" in content
    assert "不要编造未执行的过滤字段" in content
