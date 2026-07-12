from pathlib import Path

import tomllib

from dbgpt.model.proxy.llms.claude import (
    ClaudeDeployModelParameters,
    ClaudeLLMClient,
)

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


def test_proxy_config_matches_hotel_be_manufacturing_provider_contract():
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))

    providers = {
        model["name"]: {
            "provider": model["provider"],
            "api_base": model["api_base"],
        }
        for model in config["models"]["llms"]
    }

    assert providers == {
        "MiniMax-M3": {
            "provider": "proxy/claude",
            "api_base": "https://api.minimax.io/anthropic",
        },
        "kimi2.7": {
            "provider": "proxy/claude",
            "api_base": "https://api.kimi.com/coding/",
        },
        "glm-5.2": {
            "provider": "proxy/claude",
            "api_base": "https://api.z.ai/api/anthropic",
        },
    }


def test_proxy_config_builds_native_anthropic_clients(monkeypatch):
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")

    clients = {}
    for model in config["models"]["llms"]:
        parameters = ClaudeDeployModelParameters(**model)
        client = ClaudeLLMClient.new_client(parameters)
        clients[client.default_model] = client._api_base

    assert clients == {
        "MiniMax-M3": "https://api.minimax.io/anthropic",
        "kimi2.7": "https://api.kimi.com/coding/",
        "glm-5.2": "https://api.z.ai/api/anthropic",
    }


def _is_env_reference(value: object) -> bool:
    return isinstance(value, str) and value.startswith("${env:") and value.endswith("}")
