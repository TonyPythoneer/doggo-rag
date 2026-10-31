"""Paired evaluation runner tests using fake local models."""

from __future__ import annotations

import importlib
import json

import pytest

from app.config import ResultsConfig, load_config
from app.pipeline import orchestrator, rag_pipeline
from tests.conftest import FakeVisionModel


def _runner():
    try:
        return importlib.import_module("app.evaluation.runner")
    except ImportError as exc:
        pytest.fail(f"paired evaluation runner is not importable: {exc}")


def result_path(cfg):
    paths = list(cfg.results.resolve_dir().rglob("paired-*.jsonl"))
    assert paths, "paired result file is missing"
    return paths[0]


@pytest.fixture()
def eval_cfg(tmp_path):
    cfg = load_config()
    cfg.results = ResultsConfig(dir=str(tmp_path / "results"))
    return cfg


@pytest.fixture()
def fakes(monkeypatch, tmp_index):
    monkeypatch.setattr(orchestrator, "get_vision_model", lambda *args, **kwargs: FakeVisionModel())
    monkeypatch.setattr(rag_pipeline, "get_retriever", lambda: tmp_index)
    return tmp_index


def test_run_evaluation_writes_one_paired_record_per_image(eval_cfg, fakes):
    module = _runner()
    summary = module.run_evaluation(limit=3, resume=False, cfg=eval_cfg)
    records = module.load_records(result_path(eval_cfg))

    assert summary["dataset_size"] == 3
    assert len(records) == 3
    for record in records:
        assert record["schema_version"] == "paired"
        assert record["status"] == "ok"
        assert record["ground_truth_is_dog"] in (True, False)
        assert record["first_pass"]["is_dog"] is False
        assert record["second_pass"]["is_dog"] is False
        assert record["retrieval_query"] == record["first_pass"]["description"]
        assert record["retrieved_documents"]
        assert record["timings"]["first_pass_s"] >= 0
        assert record["timings"]["second_pass_s"] >= 0
        assert record["step_statuses"] == {
            "first_pass": "completed",
            "retrieval": "completed",
            "second_pass": "completed",
        }


def test_run_evaluation_resumes_only_missing_images(eval_cfg, fakes):
    module = _runner()
    module.run_evaluation(limit=2, resume=False, cfg=eval_cfg)
    summary = module.run_evaluation(limit=3, resume=True, cfg=eval_cfg)

    assert summary["written"] == 1
    assert summary["total_on_disk"] == 3


def test_run_evaluation_no_resume_restarts_new_schema_file(eval_cfg, fakes):
    module = _runner()
    legacy = eval_cfg.results.resolve_dir() / "baseline.jsonl"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"legacy": true}\n', encoding="utf-8")
    module.run_evaluation(limit=3, resume=False, cfg=eval_cfg)
    module.run_evaluation(limit=2, resume=False, cfg=eval_cfg)

    assert len(module.load_records(result_path(eval_cfg))) == 2
    assert json.loads(legacy.read_text(encoding="utf-8")) == {"legacy": True}


def test_second_pass_error_preserves_first_pass_and_retrieval(eval_cfg, fakes, monkeypatch):
    module = _runner()
    monkeypatch.setattr(
        orchestrator,
        "get_vision_model",
        lambda *args, **kwargs: FakeVisionModel(second={"is_dog": False, "confidence": 2.0}),
    )
    module.run_evaluation(limit=1, resume=False, cfg=eval_cfg)
    record = module.load_records(result_path(eval_cfg))[0]

    assert record["status"] == "error"
    assert record["first_pass"] is not None
    assert record["retrieved_documents"]
    assert record["second_pass"] is None
    assert "second_pass" in record["stage_errors"]
    assert record["step_statuses"] == {
        "first_pass": "completed",
        "retrieval": "completed",
        "second_pass": "failed",
    }


def test_retrieval_error_preserves_first_pass_and_skips_second_pass(eval_cfg, fakes, monkeypatch):
    module = _runner()

    class BrokenRetriever:
        def search(self, query, top_k, oversample):
            raise RuntimeError("index unavailable")

    monkeypatch.setattr(rag_pipeline, "get_retriever", lambda: BrokenRetriever())
    module.run_evaluation(limit=1, resume=False, cfg=eval_cfg)
    record = module.load_records(result_path(eval_cfg))[0]

    assert record["status"] == "error"
    assert record["first_pass"] is not None
    assert record["retrieval_query"] == record["first_pass"]["description"]
    assert record["retrieved_documents"] == []
    assert record["second_pass"] is None
    assert record["stage_errors"] == {"retrieval": "index unavailable"}
    assert record["step_statuses"] == {
        "first_pass": "completed",
        "retrieval": "failed",
        "second_pass": "skipped",
    }


def test_checkpoint_records_step_statuses_before_second_pass(eval_cfg, fakes, monkeypatch):
    module = _runner()
    checkpoints: list[dict] = []
    original = module._write_record

    def spy(path, record):
        checkpoints.append(json.loads(json.dumps(record)))
        original(path, record)

    monkeypatch.setattr(module, "_write_record", spy)
    module.run_evaluation(limit=1, resume=False, cfg=eval_cfg)

    in_progress = [record for record in checkpoints if record["status"] == "in_progress"]
    assert [record["step_statuses"] for record in in_progress] == [
        {"first_pass": "completed", "retrieval": "not_started", "second_pass": "not_started"},
        {"first_pass": "completed", "retrieval": "completed", "second_pass": "not_started"},
    ]
    assert in_progress[0]["first_pass"] is not None
    assert in_progress[1]["retrieved_documents"]


def test_unexpected_pipeline_exception_is_recorded_with_unstarted_steps(
    eval_cfg, fakes, monkeypatch
):
    module = _runner()

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "predict", explode)
    module.run_evaluation(limit=1, resume=False, cfg=eval_cfg)
    record = module.load_records(result_path(eval_cfg))[0]

    assert record["status"] == "error"
    assert record["stage_errors"] == {"pipeline": "boom"}
    assert record["step_statuses"] == {
        "first_pass": "not_started",
        "retrieval": "not_started",
        "second_pass": "not_started",
    }


def test_first_pass_error_is_not_scored_as_false(eval_cfg, fakes, monkeypatch):
    module = _runner()
    monkeypatch.setattr(
        orchestrator,
        "get_vision_model",
        lambda *args, **kwargs: FakeVisionModel(
            first={"is_dog": "bad", "confidence": 0.8, "description": "traits"}
        ),
    )
    module.run_evaluation(limit=1, resume=False, cfg=eval_cfg)
    record = module.load_records(result_path(eval_cfg))[0]
    metrics = module.summarize([record])

    assert record["status"] == "error"
    assert record["first_pass"] is None
    assert record["second_pass"] is None
    assert metrics["paired_scored"] == 0
    assert metrics["failed"] == 1
    assert record["step_statuses"] == {
        "first_pass": "failed",
        "retrieval": "skipped",
        "second_pass": "skipped",
    }


def test_summarize_uses_common_paired_cohort_and_reports_required_fields(eval_cfg, fakes):
    module = _runner()
    module.run_evaluation(limit=3, resume=False, cfg=eval_cfg)
    records = module.load_records(result_path(eval_cfg))
    metrics = module.summarize(records)

    comparison = metrics["comparison"]
    assert comparison["paired_images"] == 3
    assert "accuracy_improvement_percentage_points" in comparison
    assert "incorrect_to_correct" in comparison["transitions"]
    assert "per_species_accuracy" in comparison["first_pass"]
    assert comparison["first_pass"]["confusion_matrix"]["predicted_labels"] == [True, False, None]
    assert metrics["inference_time"]["first_pass"]["count"] == 3
    assert metrics["inference_time"]["second_pass"]["count"] == 3
    assert metrics["inference_time"]["retrieval"]["count"] == 3


def test_resume_retries_failed_images(eval_cfg, fakes, monkeypatch):
    module = _runner()
    broken = FakeVisionModel(first={"is_dog": "bad", "confidence": 0.8, "description": "x"})
    monkeypatch.setattr(orchestrator, "get_vision_model", lambda *a, **k: broken)
    module.run_evaluation(limit=1, resume=False, cfg=eval_cfg)
    monkeypatch.setattr(orchestrator, "get_vision_model", lambda *a, **k: FakeVisionModel())

    summary = module.run_evaluation(limit=1, cfg=eval_cfg)

    assert summary["written"] == 1
    assert [r["status"] for r in module.load_records(result_path(eval_cfg))] == ["ok"]


def test_resume_refuses_to_mix_results_from_different_settings(eval_cfg, fakes):
    module = _runner()
    module.run_evaluation(limit=1, resume=False, cfg=eval_cfg)
    eval_cfg.retrieval.top_k = 2

    with pytest.raises(module.SettingsChangedError, match="FRESH=1"):
        module.run_evaluation(limit=2, cfg=eval_cfg)
    assert len(module.load_records(result_path(eval_cfg))) == 1


def test_result_paths_are_isolated_by_model_and_dataset_version(eval_cfg, tmp_path):
    module = _runner()
    first_path = module.result_path(eval_cfg)
    other_model = eval_cfg.model_copy(deep=True)
    other_model.vision.model_id = "qwen3-vl-2b"
    assert module.result_path(other_model) != first_path

    changed_dataset = eval_cfg.model_copy(deep=True)
    changed_manifest = tmp_path / "different.csv"
    changed_manifest.write_text("filename,label,species,license,source_url,author\n", encoding="utf-8")
    changed_dataset.dataset.manifest = str(changed_manifest)
    assert module.result_path(changed_dataset) != first_path


def test_run_evaluation_dogs_samples_dogs_and_keeps_every_non_dog(eval_cfg, fakes):
    module = _runner()
    module.run_evaluation(resume=False, cfg=eval_cfg, dogs=3)
    records = module.load_records(result_path(eval_cfg))

    assert sum(r["ground_truth_is_dog"] for r in records) == 3
    assert sum(not r["ground_truth_is_dog"] for r in records) == 100
