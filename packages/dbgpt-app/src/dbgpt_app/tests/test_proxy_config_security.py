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


def test_manufacturing_model_aliases_are_versioned_without_credentials():
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    models = {model["name"]: model for model in config["models"]["llms"]}

    assert models["MiniMax-M3"] == {
        "name": "MiniMax-M3",
        "provider": "proxy/claude",
        "api_base": "https://api.minimax.io/anthropic",
        "api_key": "${env:MINIMAX_M3_API_KEY}",
    }
    assert models["kimi2.7"] == {
        "name": "kimi2.7",
        "provider": "proxy/claude",
        "api_base": "https://api.kimi.com/coding/",
        "api_key": "${env:KIMI27_API_KEY}",
    }
    assert models["glm5.2"] == {
        "name": "glm5.2",
        "provider": "proxy/claude",
        "api_base": "https://api.z.ai/api/anthropic",
        "api_key": "${env:ZAI_ANTHROPIC_API_KEY}",
        "backend": "glm-5.2",
    }


def _is_env_reference(value: object) -> bool:
    return isinstance(value, str) and value.startswith("${env:") and value.endswith("}")
