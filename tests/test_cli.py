"""CLI parser tests (no inference)."""

from __future__ import annotations

import pytest

from app.cli import build_parser, main


def test_subcommands_exist():
    parser = build_parser()
    for command in ("ingest", "predict", "evaluate", "serve"):
        args = parser.parse_args([command] if command != "predict" else [command, "x.jpg"])
        assert args.command == command
        assert callable(args.func)


def test_predict_model_choices():
    parser = build_parser()
    assert parser.parse_args(["predict", "img.jpg"]).model is None
    assert parser.parse_args(["predict", "img.jpg", "--model", "qwen3-vl-2b"]).model == "qwen3-vl-2b"
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "img.jpg", "--model", "bogus"])
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "img.jpg", "--mode", "rag"])


def test_evaluate_defaults():
    parser = build_parser()
    args = parser.parse_args(["evaluate"])
    assert args.model is None
    assert args.limit is None
    assert args.no_resume is False


def test_predict_runs_paired_pipeline_with_selected_model(monkeypatch, capsys):
    from app.pipeline import orchestrator
    from app.pipeline.schemas import PipelineResponse

    calls: list[dict] = []

    def fake_predict(image, **kwargs):
        calls.append({"image": image, **kwargs})
        return PipelineResponse(image=image, model_id=kwargs["model_id"], status="ok")

    monkeypatch.setattr(orchestrator, "predict", fake_predict)

    assert main(["predict", "img.jpg", "--model", "qwen3-vl-2b"]) == 0
    assert calls == [{"image": "img.jpg", "model_id": "qwen3-vl-2b"}]
    assert '"model_id": "qwen3-vl-2b"' in capsys.readouterr().out


def test_evaluate_uses_selected_model_config(monkeypatch, tmp_path):
    from app.config import MODEL_REGISTRY, ResultsConfig, load_config
    from app.evaluation import runner

    cfg = load_config()
    cfg.results = ResultsConfig(dir=str(tmp_path))
    monkeypatch.setattr("app.config.load_config", lambda: cfg)
    seen: dict = {}

    def fake_run_evaluation(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(runner, "run_evaluation", fake_run_evaluation)

    assert main(["evaluate", "--model", "lfm2-vl-450m", "--limit", "2"]) == 0
    assert seen["cfg"].vision.model_id == "lfm2-vl-450m"
    assert seen["cfg"].vision.model_repo == MODEL_REGISTRY["lfm2-vl-450m"][1]
    assert seen["limit"] == 2
    assert seen["resume"] is True
    assert seen["output_dir"] == tmp_path / "lfm2-vl-450m"
    assert seen["dogs"] is None

    assert main(["evaluate", "--model", "lfm2-vl-450m", "--dogs", "10"]) == 0
    assert seen["dogs"] == 10
    assert seen["output_dir"] == tmp_path / "fast" / "lfm2-vl-450m"


def test_version_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
