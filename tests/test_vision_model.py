"""Model identity and model-lifecycle tests for the three stable model IDs.

These tests never load a real MLX model: weights load lazily on generate().
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app.config import load_config
from app.vision.model import (
    _MODEL_CACHE as model_cache,
)
from app.vision.model import (
    ANSWER_MAX_TOKENS,
    BUDGET_NOTE,
    VisionModel,
    VisionModelError,
    get_vision_model,
    reset_model_cache,
    resolve_cached_snapshot,
)
from app.vision.schemas import SecondPassResult


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


@pytest.fixture()
def stub_mlx(monkeypatch):
    """Fake mlx-vlm modules (the real one imports torch, whose OpenMP clashes with faiss's)."""
    stub = SimpleNamespace(replies=[], calls=[], processors=[])

    def generate(model, processor, prompt, image=None, max_tokens=0, **kwargs):
        stub.calls.append((prompt, max_tokens))
        stub.processors.append(kwargs.get("logits_processors"))
        return SimpleNamespace(text=stub.replies.pop(0))

    def template(*args, enable_thinking=False, **kwargs):
        return "PROMPT<think>" if enable_thinking else "PROMPT<think></think>"

    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(generate=generate))
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.prompt_utils", SimpleNamespace(apply_chat_template=template)
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", SimpleNamespace(load_image=lambda p: None))
    monkeypatch.setattr(
        "app.vision.model._schema_logits_processor", lambda tok, schema: schema.__name__
    )
    model = VisionModel(load_config())
    model._model, model._processor = SimpleNamespace(config=None), object()  # never load
    stub.model = model
    return stub


def test_finished_reasoning_is_returned_with_its_opening_tag(stub_mlx):
    stub_mlx.replies = ['Ears erect.</think>{"is_dog": false, "confidence": 0.7}']

    text = stub_mlx.model.generate("img.jpg", "p", 100, thinking=True)

    assert text == '<think>Ears erect.</think>{"is_dog": false, "confidence": 0.7}'
    assert len(stub_mlx.calls) == 1


def test_reasoning_cut_at_the_token_limit_is_closed_and_answered(stub_mlx):
    stub_mlx.replies = ["Ears erect, but", '{"is_dog": false, "confidence": 0.7}']

    text = stub_mlx.model.generate("img.jpg", "p", 100, thinking=True)

    closed = f"Ears erect, but{BUDGET_NOTE}</think>"
    assert text == f'<think>{closed}{{"is_dog": false, "confidence": 0.7}}'
    assert stub_mlx.calls == [("PROMPT<think>", 100), (f"PROMPT<think>{closed}", ANSWER_MAX_TOKENS)]


def test_without_thinking_the_text_is_returned_unchanged(stub_mlx):
    stub_mlx.replies = ['{"is_dog": true, "confidence": 0.9}']

    assert stub_mlx.model.generate("img.jpg", "p", 100) == '{"is_dog": true, "confidence": 0.9}'


def test_a_schema_constrains_the_answer_but_not_free_reasoning(stub_mlx):
    stub_mlx.replies = ["{}", "Ears erect, but", "{}"]

    stub_mlx.model.generate("img.jpg", "p", 100, schema=SecondPassResult)
    stub_mlx.model.generate("img.jpg", "p", 100, thinking=True, schema=SecondPassResult)

    assert stub_mlx.processors == [["SecondPassResult"], None, ["SecondPassResult"]]
