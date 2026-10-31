"""Integrity checks on paired results before they are reported."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import MODEL_REGISTRY
from app.evaluation.runner import RESULT_SCHEMA_VERSION

SUMMARY_FILE = f"{RESULT_SCHEMA_VERSION}-summary.json"


def _load(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"_invalid_json": line})
    return records


def _record_problems(model_id: str, version: str, record: dict[str, Any]) -> list[str]:
    if "_invalid_json" in record:
        return [f"{model_id}: malformed JSON record"]
    image = record.get("image_id")
    problems = []
    if record.get("schema_version") != RESULT_SCHEMA_VERSION:
        problems.append(f"{model_id}: unsupported schema_version {record.get('schema_version')!r}")
    if record.get("model_id") != model_id:
        problems.append(f"{model_id}: record model_id is {record.get('model_id')!r}")
    if record.get("dataset_version") != version:
        problems.append(f"{model_id}: mixed dataset versions ({image})")
    if record.get("status") not in {"ok", "error"}:
        problems.append(f"{model_id}: unfinished record for {image}")
    first, second = record.get("first_pass"), record.get("second_pass")
    if record.get("status") == "ok" and (first is None or second is None):
        problems.append(f"{model_id}: successful record lacks both pass outputs ({image})")
    if record.get("status") == "error" and not record.get("stage_errors"):
        problems.append(f"{model_id}: failed record has no stage error ({image})")
    if first is not None and record.get("retrieval_query") != first.get("description"):
        problems.append(f"{model_id}: retrieval query is not the First Pass description ({image})")
    ids = [item.get("document_id") for item in record.get("retrieved_documents") or []]
    if ids != (record.get("retrieved_document_ids") or []):
        problems.append(f"{model_id}: retrieved document IDs/scores are inconsistent ({image})")
    return problems


def verify(
    results_dir: Path,
    models: tuple[str, ...] = tuple(MODEL_REGISTRY),
    expected_records: int | None = None,
) -> list[str]:
    """Return every problem found; an empty list means the results are reportable."""
    summary_path = results_dir / SUMMARY_FILE
    if not summary_path.is_file():
        return [f"missing {summary_path}"]
    experiments = json.loads(summary_path.read_text(encoding="utf-8")).get("experiments") or {}
    problems: list[str] = []
    image_sets: list[set[str]] = []

    for model_id in models:
        experiment = experiments.get(model_id) or {}
        version = str(experiment.get("dataset_version") or "")
        path = results_dir / model_id / f"{RESULT_SCHEMA_VERSION}-{model_id}-{version}.jsonl"
        if not version or not path.is_file():
            problems.append(
                f"{model_id}: missing paired results for dataset version {version!r}"
            )
            continue
        records = _load(path)
        if expected_records is not None and len(records) != expected_records:
            problems.append(f"{model_id}: {len(records)} records, expected {expected_records}")
        keys = Counter((r.get("experiment_key"), r.get("image_id")) for r in records)
        duplicates = [key for key, count in keys.items() if count > 1]
        if duplicates:
            problems.append(f"{model_id}: duplicate image record {duplicates[0]!r}")
        for record in records:
            problems.extend(_record_problems(model_id, version, record))
        if (experiment.get("metadata") or {}).get("vision_model") != model_id:
            problems.append(f"{model_id}: run metadata is missing or names another model")
        if experiment.get("paired_scored", 0) + experiment.get("failed", 0) != len(records):
            problems.append(f"{model_id}: summary counts do not match the {len(records)} records")
        image_sets.append({record.get("image_id") for record in records})

    if any(images != image_sets[0] for images in image_sets[1:]):
        problems.append("model result files do not cover the same image IDs")
    return problems
