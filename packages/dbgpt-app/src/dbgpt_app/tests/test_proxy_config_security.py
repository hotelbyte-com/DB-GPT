from pathlib import Path

import tomllib

CONFIG = Path(__file__).parents[5] / "configs" / "dbgpt-proxy-openai.toml"


def test_proxy_openai_config_is_loopback_and_contains_no_provider_credentials():
    raw = CONFIG.read_text(encoding="utf-8")
    config = tomllib.loads(raw)

    assert config["service"]["web"]["host"] == "127.0.0.1"
    assert "0.0.0.0" not in raw
    assert _is_env_reference(config["system"]["encrypt_key"])

    llms = config["models"]["llms"]
    assert llms
    for model in llms:
        assert _is_env_reference(model["api_key"]), model["name"]


def _is_env_reference(value: object) -> bool:
    return isinstance(value, str) and value.startswith("${env:") and value.endswith("}")
