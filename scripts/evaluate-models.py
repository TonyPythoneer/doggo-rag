#!/usr/bin/env python3
"""`make eval`: paired First Pass vs RAG Second Pass for every model, then verify and report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import (
    DATASET_SIZE,
    MODEL_REGISTRY,
    NON_DOG_COUNT,
    RESULT_SCHEMA_VERSION,
    load_config,
)
from app.evaluation import metadata, results, runner
from app.evaluation.metrics import summarize
from app.evaluation.report import build_report
from app.evaluation.verify import SUMMARY_FILE, verify


def start_fresh(results_dir: Path, model_ids: list[str]) -> None:
    """Remove only versioned paired outputs; legacy measured results stay."""
    for model_id in model_ids:
        for path in (results_dir / model_id).glob(f"{RESULT_SCHEMA_VERSION}-*"):
            path.unlink()


def consolidate_summary(results_dir: Path, model_ids: list[str]) -> dict:
    """Summarize each model's current-version result file independently."""
    summary: dict = {"schema_version": RESULT_SCHEMA_VERSION, "experiments": {}}
    base = load_config()
    for model_id in model_ids:
        cfg = base.for_model(model_id)
        path = results.result_path(cfg, results_dir / model_id)
        key = metadata.experiment_key(cfg)
        records = [r for r in results.load_records(path) if r.get("experiment_key") == key]
        if not records:
            continue
        meta = results.metadata_path(cfg, results_dir / model_id)
        summary["experiments"][model_id] = {
            **summarize(records),
            "dataset_version": metadata.dataset_version(cfg),
            "result_file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "metadata": json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {},
        }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / SUMMARY_FILE).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_models(model_ids: list[str], results_dir: Path, args: argparse.Namespace) -> bool:
    """Evaluate every selected model; one model's failure does not stop the others."""
    from app.rag.ingestion import build_index

    # Rebuild once per run so the index always matches the knowledge files and embedder.
    base = load_config()
    build_index(base)
    failed = False
    for model_id in model_ids:
        print(f"=== {model_id} ===", flush=True)
        try:
            result = runner.run_evaluation(
                limit=args.limit,
                cfg=base.for_model(model_id),
                output_dir=results_dir / model_id,
                dogs=args.dogs,
            )
            print(json.dumps(result, indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001 - keep other model runs independent
            failed = True
            print(f"Model {model_id} failed: {exc}", flush=True)
    return failed


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh", action="store_true", help=f"delete {RESULT_SCHEMA_VERSION} results first"
    )
    parser.add_argument("--limit", type=int, default=None, help="only the first N images")
    parser.add_argument(
        "--dogs", type=int, default=None, help="fast mode: N sampled dogs + every non-dog"
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY), default=None, help="only this model"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, results_dir: Path | None = None) -> int:
    args = parse_args(argv)
    model_ids = [args.model] if args.model else list(MODEL_REGISTRY)
    target = results_dir or load_config().results.resolve_dir()
    if args.dogs is not None:
        target = target / "fast"  # sampled runs never mix with the full results
    if args.fresh:
        start_fresh(target, model_ids)

    failed = run_models(model_ids, target, args)
    consolidate_summary(target, model_ids)
    expected = DATASET_SIZE if args.dogs is None else NON_DOG_COUNT + args.dogs
    problems = verify(target, models=tuple(model_ids), expected_records=args.limit or expected)
    for problem in problems:
        print(f"- {problem}", flush=True)
    report = build_report(target / SUMMARY_FILE, target / "eval.html")
    print(f"Wrote {report}" + (" (verification problems above)" if problems else ""), flush=True)
    return 1 if failed or problems else 0


if __name__ == "__main__":
    sys.exit(main())
