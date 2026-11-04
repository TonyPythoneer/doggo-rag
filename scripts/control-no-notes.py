#!/usr/bin/env python3
"""No-notes control (spec §13): each model's Second Pass again, without the retrieved notes.

Reads the paired results of `make eval`, reuses each image's recorded First Pass, and runs
the Second Pass with the context an empty retrieval gives. The only difference from the
recorded RAG Second Pass is the retrieved knowledge, so comparing the two credits (or not)
the knowledge. Refuses to run when code, prompts, knowledge, models or packages differ from
the recorded run. Models run one after another, each in its own process. Ground truth is
used only to score, after inference. Output: results/control/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import MODEL_REGISTRY, AppConfig, load_config
from app.dataset import resolve_image
from app.evaluation import metadata, results
from app.pipeline.rag_pipeline import assemble_context
from app.vision.inference import run_second_pass
from app.vision.model import get_vision_model
from app.vision.schemas import FirstPassResult


def mcnemar(better: int, worse: int) -> float:
    """Two-sided exact McNemar p from the two kinds of paired disagreement."""
    n = better + worse
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(better, worse) + 1)) / 2**n)


def run_model(cfg: AppConfig) -> Path:
    """Write results/control/<model>.jsonl; images already in it are skipped (resume)."""
    recorded_meta = json.loads(results.metadata_path(cfg).read_text(encoding="utf-8"))
    current_meta = metadata.experiment_metadata(cfg)
    if not metadata.same_settings(recorded_meta, current_meta):
        raise SystemExit(f"{cfg.vision.model_id}: settings differ from the recorded make eval run")
    key = metadata.experiment_key(cfg)
    recorded = [
        record
        for record in results.load_records(results.result_path(cfg))
        if record.get("experiment_key") == key and record["status"] == "ok"
    ]
    out = cfg.results.resolve_dir() / "control" / f"{cfg.vision.model_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".metadata.json").write_text(
        json.dumps({**current_meta, "condition": "second pass without retrieved notes"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    done = {record["image_id"] for record in results.load_records(out)}
    model = get_vision_model(cfg.vision.model_id)
    no_notes = assemble_context([])
    with out.open("a", encoding="utf-8") as handle:
        for record in recorded:
            if record["image_id"] in done:
                continue
            first = FirstPassResult.model_validate(record["first_pass"])
            image = str(resolve_image(record["image_path"], cfg))
            try:
                second, _, raw = run_second_pass(image, first, no_notes, cfg=cfg, model=model)
                control, error = second.model_dump(mode="json"), None
            except Exception as exc:  # noqa: BLE001 - recorded as a failure, never scored
                control, error, raw = None, str(exc), getattr(exc, "raw_text", None)
            line = {
                "image_id": record["image_id"],
                "species": record["species"],
                "ground_truth_is_dog": record["ground_truth_is_dog"],
                "first_pass_is_dog": record["first_pass"]["is_dog"],
                "rag_second_pass_is_dog": record["second_pass"]["is_dog"],
                "control_second_pass": control,
                "control_error": error,
                "control_raw_output": raw,
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            handle.flush()
    return out


def compare(records: list[dict]) -> dict:
    """First Pass, RAG Second Pass and no-notes Second Pass on the images all three answered."""
    scored = [record for record in records if record["control_second_pass"]]
    truth = [record["ground_truth_is_dog"] for record in scored]
    first = [r["first_pass_is_dog"] is t for r, t in zip(scored, truth)]
    rag = [r["rag_second_pass_is_dog"] is t for r, t in zip(scored, truth)]
    control = [r["control_second_pass"]["is_dog"] is t for r, t in zip(scored, truth)]
    rag_only = sum(a and not b for a, b in zip(rag, control))
    control_only = sum(b and not a for a, b in zip(rag, control))
    control_fixed = sum(b and not a for a, b in zip(first, control))
    control_broke = sum(a and not b for a, b in zip(first, control))
    n = len(scored)
    return {
        "scored": n,
        "control_failed": len(records) - n,
        "first_pass_accuracy": round(sum(first) / n, 4) if n else None,
        "rag_second_pass_accuracy": round(sum(rag) / n, 4) if n else None,
        "control_second_pass_accuracy": round(sum(control) / n, 4) if n else None,
        "rag_right_control_wrong": rag_only,
        "control_right_rag_wrong": control_only,
        "mcnemar_p_rag_vs_control": round(mcnemar(rag_only, control_only), 6),
        "control_vs_first_pass": {"fixed": control_fixed, "broke": control_broke},
        "mcnemar_p_control_vs_first_pass": round(mcnemar(control_fixed, control_broke), 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), help="only this model")
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.model:
        run_model(cfg.for_model(args.model))
    else:
        for model_id in MODEL_REGISTRY:
            # One process per model: three models in one process broke qwen3-vl-2b's output.
            subprocess.run([sys.executable, __file__, "--model", model_id], check=True)
    control_dir = cfg.results.resolve_dir() / "control"
    summary = {
        path.stem: compare(results.load_records(path))
        for path in sorted(control_dir.glob("*.jsonl"))
    }
    (control_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
