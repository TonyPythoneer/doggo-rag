"""Config loading tests."""

from __future__ import annotations

from app.config import load_config


def test_default_config_loads(config):
    assert config.vision.model_id
    assert config.vision.temperature == 0.0
    assert config.retrieval.top_k >= 1
    assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.prompts_dir.name == "prompts"
    assert not hasattr(config.retrieval, "query")
    assert not hasattr(config, "query_template")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DOGGO_TOP_K", "7")
    monkeypatch.setenv("DOGGO_EMBEDDING_MODEL", "some/other-model")
    cfg = load_config()
    assert cfg.retrieval.top_k == 7
    assert cfg.embedding.model_name == "some/other-model"


def test_paths_resolve_against_repo_root(config):
    assert config.knowledge.resolve_dir().is_absolute()
    assert config.dataset.resolve_manifest().name == "manifest.csv"


def test_tunable_defaults_come_from_config():
    from app.config import load_config

    cfg = load_config()
    assert cfg.knowledge.chunk_max_chars == 1200
    assert cfg.retrieval.oversample == 3
    assert (cfg.server.host, cfg.server.port) == ("127.0.0.1", 8888)
    assert cfg.evaluation.sample_seed == 42
