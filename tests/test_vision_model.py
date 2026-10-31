"""Model identity and model-lifecycle tests for the three stable model IDs.

These tests never load a real MLX model: weights load lazily on generate().
"""

from __future__ import annotations

import pytest

from app.config import load_config
from app.vision.model import (
    _MODEL_CACHE as model_cache,
)
from app.vision.model import (
    VisionModel,
    VisionModelError,
    get_vision_model,
    reset_model_cache,
    resolve_cached_snapshot,
)


@pytest.fixture(autouse=True)
def _clean_model_state():
    reset_model_cache()
    yield
    reset_model_cache()


def test_registry_holds_three_stable_ids():
    # The canonical spec registry in app.config lists exactly the three
    # stable model IDs required by the plan.
    from app.config import MODEL_REGISTRY as SPECS

    assert set(SPECS) == {"lfm3-vl-2b", "lfm2-vl-450m", "qwen3-vl-2b"}
    assert SPECS["lfm3-vl-2b"][1] == "LiquidAI/LFM3-VL-2B-MLX-4bit"
    assert SPECS["lfm2-vl-450m"][1] == "mlx-community/LFM2-VL-450M-4bit"
    assert SPECS["qwen3-vl-2b"][1] == "mlx-community/Qwen3-VL-2B-Instruct-4bit"


def test_get_vision_model_builds_requested_id_with_shared_decoding():
    model = get_vision_model("qwen3-vl-2b")
    cfg = load_config()

    assert model.config.vision.model_id == "qwen3-vl-2b"
    assert model.config.vision.model_repo == "mlx-community/Qwen3-VL-2B-Instruct-4bit"
    assert model.config.vision.temperature == cfg.vision.temperature
    assert model.config.vision.max_tokens_second_pass == cfg.vision.max_tokens_second_pass
    assert not model.is_loaded


def test_default_model_id_is_lfm3():
    assert get_vision_model().config.vision.model_id == "lfm3-vl-2b"


def test_same_id_is_cached_and_switching_drops_previous_model():
    first = get_vision_model("lfm3-vl-2b")
    assert get_vision_model("lfm3-vl-2b") is first

    get_vision_model("lfm2-vl-450m")

    assert list(model_cache) == ["lfm2-vl-450m"]
    assert get_vision_model("lfm3-vl-2b") is not first


def test_unknown_model_id_raises_instead_of_substituting():
    with pytest.raises(VisionModelError, match="Unknown vision model ID"):
        get_vision_model("definitely-not-a-model")


def test_cached_snapshot_resolution_is_local_only(monkeypatch, tmp_path):
    calls: dict = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert resolve_cached_snapshot("some/repo") == tmp_path / "snapshot"
    assert calls["repo_id"] == "some/repo"
    assert calls["local_files_only"] is True


def test_missing_local_model_is_blocked_before_mlx_load(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.vision.model_path = str(tmp_path / "missing-checkpoint")
    cfg.vision.model_repo = "some/repo"

    def missing_snapshot(repo):
        return None

    monkeypatch.setattr("app.vision.model.resolve_cached_snapshot", missing_snapshot)
    model = VisionModel(cfg)

    with pytest.raises(VisionModelError, match="not available locally"):
        model._ensure_loaded()
