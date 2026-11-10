import importlib
from pathlib import Path

import config


def reload_config():
    return importlib.reload(config)


def test_environment_variable_overrides_default(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "my/custom-model")
    monkeypatch.setenv("LLM_MODEL", "my-llm")
    c = reload_config()
    assert c.EMBED_MODEL == "my/custom-model" and c.LLM_MODEL == "my-llm"
    monkeypatch.undo(); reload_config()


def test_empty_url_means_none(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "")
    assert reload_config().QDRANT_URL is None
    monkeypatch.undo(); reload_config()


def test_defaults_exist():
    assert config.EMBED_MODEL and config.QDRANT_PATH and config.QUERY_PREFIX.startswith("Represent")


def test_secret_is_masked_in_describe(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-supersecretvalue99")
    c = reload_config()
    assert "supersecret" not in str(c.describe())
    monkeypatch.undo(); reload_config()


def test_no_secret_hardcoded_in_source():
    source = Path(config.__file__).read_text()
    assert "sk-" not in source
