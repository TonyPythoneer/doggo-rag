"""Tests for `make eval` (verify, report) and the model download script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from app.config import load_config
from app.evaluation import metadata, report, results
from app.evaluation.verify import verify

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_shows_paired_comparison_transitions_and_matrix(tmp_path):
    comparison = {
        "paired_images": 100,
        "first_pass": {
            "accuracy": 0.78,
            "confusion_matrix": {"matrix": [[40, 8, 2], [12, 38, 0]]},
            "per_species_accuracy": {"dog": 0.8, "fox": 0.5},
        },
        "second_pass": {"accuracy": 0.86, "per_species_accuracy": {"dog": 0.9, "fox": 0.7}},
        "accuracy_improvement_percentage_points": 8.0,
        "transitions": {
            "incorrect_to_correct": 12,
            "correct_to_incorrect": 4,
            "unchanged_correct": 70,
            "unchanged_incorrect": 14,
        },
    }
    summary = {"experiments": {"lfm3-vl-2b": {"comparison": comparison, "failed": 0}}}

    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    page = report.build_report(path, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "LFM3 VL 2B (Default)" in page
    assert "▲ +8.0 pp better" in page
    assert "Fixed by the second look" in page
    assert page.index("<th>Dogs</th>") < page.index("<th>Foxes</th>")


@pytest.fixture()
def evaluate(tmp_path, monkeypatch):
    module = _load_script("evaluate_models", "scripts/evaluate-models.py")
    monkeypatch.setattr(module, "MODEL_IDS", ["lfm3-vl-2b", "lfm2-vl-450m"])
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    return module


def _write_pair(root: Path, model_id: str, query: str = "visible animal traits") -> None:
    cfg = load_config().for_model(model_id)
    path = results.result_path(cfg, root / model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _record("dog/dog_001", model_id, metadata.dataset_version(cfg), query)
    record["experiment_key"] = metadata.experiment_key(cfg)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    results.metadata_path(cfg, root / model_id).write_text(
        json.dumps(metadata.experiment_metadata(cfg)), encoding="utf-8"
    )


def test_consolidate_summary_keeps_models_separate_and_preserves_legacy(tmp_path, evaluate):
    legacy = tmp_path / "lfm3-vl-2b" / "baseline.jsonl"
    legacy.parent.mkdir()
    legacy.write_text('{"legacy":true}\n', encoding="utf-8")
    for model_id in evaluate.MODEL_IDS:
        _write_pair(tmp_path, model_id)

    summary = evaluate.consolidate_summary()

    assert set(summary["experiments"]) == {"lfm3-vl-2b", "lfm2-vl-450m"}
    for model_id, experiment in summary["experiments"].items():
        assert experiment["metadata"]["vision_model"] == model_id
        assert experiment["comparison"]["paired_images"] == 1
    assert json.loads(legacy.read_text(encoding="utf-8")) == {"legacy": True}


def test_verify_accepts_consistent_paired_records(tmp_path, evaluate):
    for model_id in evaluate.MODEL_IDS:
        _write_pair(tmp_path, model_id, query=model_id)
    evaluate.consolidate_summary()

    assert verify(tmp_path, models=tuple(evaluate.MODEL_IDS), expected_records=1) == []
    assert "1 records, expected 200" in "\n".join(
        verify(tmp_path, models=tuple(evaluate.MODEL_IDS), expected_records=200)
    )


def test_verify_detects_duplicate_and_wrong_model_records(tmp_path):
    file = tmp_path / "lfm3-vl-2b" / "paired-lfm3-vl-2b-dataset.jsonl"
    file.parent.mkdir(parents=True)
    record = _record("dog/dog_001", "lfm3-vl-2b", "dataset")
    record["model_id"] = "qwen3-vl-2b"
    file.write_text("\n".join(json.dumps(record) for _ in range(2)) + "\n", encoding="utf-8")
    (tmp_path / "paired-summary.json").write_text(
        json.dumps({"experiments": {"lfm3-vl-2b": {"dataset_version": "dataset"}}}),
        encoding="utf-8",
    )

    problems = "\n".join(verify(tmp_path, models=("lfm3-vl-2b",), expected_records=1))

    assert "duplicate" in problems.lower()
    assert "model_id" in problems


def test_main_passes_limit_to_every_model_and_checks_that_count(tmp_path, evaluate, monkeypatch):
    limits: list = []
    monkeypatch.setattr("app.rag.ingestion.build_index", lambda cfg: {})
    monkeypatch.setattr(
        evaluate.runner, "run_evaluation", lambda **kwargs: limits.append(kwargs["limit"]) or {}
    )

    assert evaluate.main(["--limit", "2"]) == 1  # nothing written, so verification fails
    assert limits == [2, 2]
    assert (tmp_path / "eval.html").is_file()


def test_start_fresh_only_removes_paired_v2_results(tmp_path, evaluate):
    directory = tmp_path / "lfm3-vl-2b"
    directory.mkdir()
    paired = directory / "paired-lfm3-vl-2b-data.jsonl"
    paired.write_text("{}\n", encoding="utf-8")
    legacy = directory / "baseline.jsonl"
    legacy.write_text("{}\n", encoding="utf-8")

    evaluate.start_fresh()

    assert not paired.exists()
    assert legacy.exists()


def _record(image_id: str, model_id: str, version: str, query: str = "visible traits") -> dict:
    return {
        "schema_version": "paired",
        "experiment_key": f"paired|{model_id}|repo|{version}",
        "image_id": image_id,
        "image_path": f"images/{image_id}.jpg",
        "ground_truth_is_dog": True,
        "species": "dog",
        "model_id": model_id,
        "model_repo": "repo",
        "dataset_version": version,
        "first_pass": {"is_dog": True, "confidence": 0.8, "description": query},
        "retrieval_query": query,
        "retrieved_documents": [{"document_id": "dog", "score": 0.9}],
        "retrieved_document_ids": ["dog"],
        "second_pass": {"is_dog": True, "confidence": 0.9},
        "timings": {"first_pass_s": 1.0, "retrieval_s": 0.1, "second_pass_s": 1.2},
        "stage_errors": {},
        "status": "ok",
        "error": None,
    }


def test_download_models_check_uses_local_cache_only(monkeypatch, tmp_path):
    module = _load_script("download_models_script", "scripts/download-models.py")
    calls: list[dict] = []
    # Real HF layout: snapshot_download returns models--<repo>/snapshots/<commit>.
    snapshot = tmp_path / "models--org--model" / "snapshots" / "commit"
    snapshot.mkdir(parents=True)

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(module, "targets", lambda: [("model", "org/model")])

    assert module.main(["--check"]) == 0
    assert calls == [{"repo_id": "org/model", "local_files_only": True}]


def test_download_models_default_mode_downloads(monkeypatch, tmp_path):
    module = _load_script("download_models_fetch", "scripts/download-models.py")
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(module, "targets", lambda: [("model", "org/model")])

    assert module.main([]) == 0
    assert calls == [{"repo_id": "org/model"}]


def test_download_models_fails_when_cache_is_missing(monkeypatch):
    module = _load_script("download_models_missing", "scripts/download-models.py")

    def missing(**kwargs):
        raise RuntimeError("not cached")

    monkeypatch.setattr("huggingface_hub.snapshot_download", missing)
    monkeypatch.setattr(module, "targets", lambda: [("model", "org/model")])

    assert module.main(["--check"]) == 1


def test_main_dogs_runs_in_fast_directory_and_reports_there(tmp_path, evaluate, monkeypatch):
    seen: list = []
    monkeypatch.setattr("app.rag.ingestion.build_index", lambda cfg: {})
    monkeypatch.setattr(
        evaluate.runner, "run_evaluation", lambda **kwargs: seen.append(kwargs) or {}
    )

    evaluate.main(["--dogs", "10"])

    assert {kwargs["dogs"] for kwargs in seen} == {10}
    assert all(kwargs["output_dir"].parent == tmp_path / "fast" for kwargs in seen)
    assert (tmp_path / "fast" / "eval.html").is_file()


def test_main_model_runs_only_that_model(tmp_path, evaluate, monkeypatch):
    seen: list = []
    monkeypatch.setattr("app.rag.ingestion.build_index", lambda cfg: {})
    monkeypatch.setattr(
        evaluate.runner, "run_evaluation", lambda **kwargs: seen.append(kwargs["cfg"]) or {}
    )

    evaluate.main(["--model", "qwen3-vl-2b"])

    assert [cfg.vision.model_id for cfg in seen] == ["qwen3-vl-2b"]
