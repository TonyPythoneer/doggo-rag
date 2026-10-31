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
from app.evaluation import runner
from app.evaluation.report import build_report
from app.evaluation.verify import SUMMARY_FILE, verify

MODEL_IDS = list(MODEL_REGISTRY)
RESULTS_DIR = load_config().results.resolve_dir()


def start_fresh() -> None:
    """Remove only versioned paired outputs; legacy measured results stay."""
    for model_id in MODEL_IDS:
        for path in (RESULTS_DIR / model_id).glob(f"{RESULT_SCHEMA_VERSION}-*"):
            path.unlink()


def consolidate_summary() -> dict:
    """Summarize each model's current-version result file independently."""
    summary: dict = {"schema_version": runner.RESULT_SCHEMA_VERSION, "experiments": {}}
    base = load_config()
    for model_id in MODEL_IDS:
        cfg = base.for_model(model_id)
        path = runner.result_path(cfg, RESULTS_DIR / model_id)
        key = runner.experiment_key(cfg)
        records = [r for r in runner.load_records(path) if r.get("experiment_key") == key]
        if not records:
            continue
        meta = runner.metadata_path(cfg, RESULTS_DIR / model_id)
        summary["experiments"][model_id] = {
            **runner.summarize(records),
            "dataset_version": runner.dataset_version(cfg),
            "result_file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "metadata": json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {},
        }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / SUMMARY_FILE).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help=f"delete {RESULT_SCHEMA_VERSION} results first")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images")
    parser.add_argument(
        "--dogs", type=int, default=None, help="fast mode: N sampled dogs + every non-dog"
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY), default=None, help="only this model"
    )
    args = parser.parse_args(argv)
    global RESULTS_DIR, MODEL_IDS
    if args.model:
        MODEL_IDS = [args.model]
    if args.dogs is not None:
        RESULTS_DIR = RESULTS_DIR / "fast"  # sampled runs never mix with the full results
    if args.fresh:
        start_fresh()

    from app.rag.ingestion import build_index

    # Rebuild once per run so the index always matches the knowledge files and embedder.
    base = load_config()
    build_index(base)
    failed = False
    for model_id in MODEL_IDS:
        print(f"=== {model_id} ===", flush=True)
        try:
            result = runner.run_evaluation(
                limit=args.limit,
                cfg=base.for_model(model_id),
                output_dir=RESULTS_DIR / model_id,
                dogs=args.dogs,
            )
            print(json.dumps(result, indent=2), flush=True)
        except Exception as exc:  # keep other model runs independent
            failed = True
            print(f"Model {model_id} failed: {exc}", flush=True)

    consolidate_summary()
    expected = DATASET_SIZE if args.dogs is None else NON_DOG_COUNT + args.dogs
    problems = verify(RESULTS_DIR, models=tuple(MODEL_IDS), expected_records=args.limit or expected)
    for problem in problems:
        print(f"- {problem}", flush=True)
    report = build_report(RESULTS_DIR / SUMMARY_FILE, RESULTS_DIR / "eval.html")
    print(f"Wrote {report}" + (" (verification problems above)" if problems else ""), flush=True)
    return 1 if failed or problems else 0


if __name__ == "__main__":
    sys.exit(main())
