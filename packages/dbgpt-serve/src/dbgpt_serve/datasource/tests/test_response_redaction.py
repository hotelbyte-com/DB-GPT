import json

from dbgpt_serve.datasource.api.schemas import DatasourceServeResponse
from dbgpt_serve.datasource.service.service import (
    REDACTED_DATASOURCE_VALUE,
    Service,
    redact_datasource_params,
)


def test_datasource_response_redacts_nested_credentials_without_hiding_shape():
    params = {
        "host": "tdengine.internal",
        "port": 6030,
        "user": "readonly",
        "password": "super-secret",
        "db_pwd": "legacy-secret",
        "apiKey": "api-secret",
        "dsn": "taosws://readonly:secret@tdengine.internal:6041/hblog_ns",
        "ssl": {
            "enabled": True,
            "private_key": "private-key-material",
            "certificate_path": "/run/secrets/client.crt",
        },
        "replicas": [{"host": "tdengine-replica", "token": "replica-secret"}],
    }

    redacted = redact_datasource_params(params)

    assert redacted["host"] == "tdengine.internal"
    assert redacted["port"] == 6030
    assert redacted["user"] == "readonly"
    assert redacted["password"] == REDACTED_DATASOURCE_VALUE
    assert redacted["db_pwd"] == REDACTED_DATASOURCE_VALUE
    assert redacted["apiKey"] == REDACTED_DATASOURCE_VALUE
    assert redacted["dsn"] == REDACTED_DATASOURCE_VALUE
    assert redacted["ssl"]["enabled"] is True
    assert redacted["ssl"]["private_key"] == REDACTED_DATASOURCE_VALUE
    assert redacted["ssl"]["certificate_path"] == "/run/secrets/client.crt"
    assert redacted["replicas"][0]["host"] == "tdengine-replica"
    assert redacted["replicas"][0]["token"] == REDACTED_DATASOURCE_VALUE
    assert "super-secret" not in json.dumps(redacted)
    assert "replica-secret" not in json.dumps(redacted)


def test_datasource_response_redacts_credentials_inside_json_ext_config():
    ext_config = json.dumps(
        {
            "timezone": "Asia/Dubai",
            "auth": {"access_token": "token-value", "secret": "secret-value"},
        }
    )

    redacted = redact_datasource_params({"ext_config": ext_config})
    decoded = json.loads(redacted["ext_config"])

    assert decoded["timezone"] == "Asia/Dubai"
    assert decoded["auth"]["access_token"] == REDACTED_DATASOURCE_VALUE
    assert decoded["auth"]["secret"] == REDACTED_DATASOURCE_VALUE
    assert "token-value" not in redacted["ext_config"]
    assert "secret-value" not in redacted["ext_config"]


def test_datasource_query_response_applies_redaction_at_service_boundary():
    class FakeParameters:
        @classmethod
        def from_persisted_state(cls, state):
            instance = cls()
            instance.state = state
            return instance

        def to_dict(self):
            return {
                "host": self.state["db_host"],
                "user": self.state["db_user"],
                "password": self.state["db_pwd"],
            }

    class FakeDatasourceManager:
        @staticmethod
        def _get_param_cls(db_type):
            assert db_type == "tdengine"
            return FakeParameters

    class ProofService(Service):
        @property
        def datasource_manager(self):
            return FakeDatasourceManager()

    service = object.__new__(ProofService)
    response = service._to_query_response(
        DatasourceServeResponse(
            id=1,
            db_type="tdengine",
            db_name="hblog-shared",
            db_host="tdengine.internal",
            db_port=6041,
            db_user="readonly",
            db_pwd="service-boundary-secret",
        )
    )

    assert response.db_name == "hblog-shared"
    assert response.params["host"] == "tdengine.internal"
    assert response.params["user"] == "readonly"
    assert response.params["password"] == REDACTED_DATASOURCE_VALUE
    assert "service-boundary-secret" not in response.model_dump_json()
