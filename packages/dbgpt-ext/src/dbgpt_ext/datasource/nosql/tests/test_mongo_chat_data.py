from dbgpt_ext.datasource.nosql import mongo_chat_data


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


def test_configured_app_queries_mongo_with_configured_metrics(monkeypatch):
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
                {"name": "spa000_bool_count", "op": "sumBoolTrue", "field": "Spa000_BOOL"},
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

    rows, summary = app.query(
        "timeWindow=2026-06-11T05:38:07Z..2026-06-16T08:51:12Z"
    )

    assert fake.pipeline[0] == {
        "$match": {
            "timestamp": {
                "$gte": mongo_chat_data._parse_datetime("2026-06-11T05:38:07Z"),
                "$lte": mongo_chat_data._parse_datetime("2026-06-16T08:51:12Z"),
            }
        }
    }
    assert fake.pipeline[2]["$group"]["avg_thickness"] == {"$avg": "$AvgThk"}
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


def test_interpret_prompt_keeps_runtime_rules_as_context_not_answer_target():
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
    )

    assert [message["role"] for message in messages] == [
        mongo_chat_data.ModelMessageRoleType.HUMAN
    ]
    content = messages[0]["content"]
    assert "直接回答用户问题，不要复述规则" in content
    assert "上下文约束（只遵守，不要复述）" in content
    assert "用户问题：冷辊速度是什么意思？" in content
    assert "avg_chill_speed" in content
