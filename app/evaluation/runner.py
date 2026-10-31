"""Paired First Pass and Second Pass evaluation over the dataset manifest."""

from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

from app.config import RESULT_SCHEMA_VERSION, AppConfig, get_config
from app.dataset import DatasetRow, load_manifest
from app.evaluation.metadata import experiment_key, experiment_metadata, same_settings
from app.evaluation.results import load_records, metadata_path, result_path, write_record
from app.pipeline.orchestrator import predict
from app.pipeline.schemas import PipelineResponse

logger = logging.getLogger(__name__)


class SettingsChangedError(RuntimeError):
    """Existing results were produced with different experiment settings."""


def run_evaluation(
    limit: int | None = None,
    resume: bool = True,
    cfg: AppConfig | None = None,
    output_dir: Path | None = None,
    dogs: int | None = None,
) -> dict:
    """Run paired inference and record one versioned result per dataset image.

    Resume skips images with an `ok` record; failed and unfinished images run
    again. Resuming with different settings raises instead of mixing results.
    `dogs` keeps every non-dog image but only that many dogs, sampled with the
    configured `evaluation.sample_seed` so repeated fast runs use the same images.
    """
    cfg = cfg or get_config()
    rows = load_manifest(cfg)[:limit]
    if dogs is not None:
        dog_rows = [row for row in rows if row.label == "dog"]
        sample = random.Random(cfg.evaluation.sample_seed).sample(dog_rows, min(dogs, len(dog_rows)))
        keep = {row.filename for row in sample}
        rows = [row for row in rows if row.label != "dog" or row.filename in keep]
    out_path = result_path(cfg, output_dir)
    meta_path = metadata_path(cfg, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key = experiment_key(cfg)
    metadata = experiment_metadata(cfg)
    if not resume:
        out_path.unlink(missing_ok=True)
    records = [record for record in load_records(out_path) if record.get("experiment_key") == key]
    if records and meta_path.is_file():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if not same_settings(previous, metadata):
            raise SettingsChangedError(
                f"{out_path} was produced with different settings (prompts, knowledge, "
                "retrieval or decoding). Rerun with `make eval FRESH=1`."
            )
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    done = {record["image_id"] for record in records if record.get("status") == "ok"}
    version = metadata["dataset_version"]  # hashing every image per record would be slow

    written = errors = 0
    for row in rows:
        image_id = Path(row.filename).with_suffix("").as_posix()
        if image_id in done:
            continue

        def checkpoint(partial: PipelineResponse, row: DatasetRow = row) -> None:
            write_record(out_path, _record(row, cfg, key, version, partial, "in_progress"))

        try:
            response = predict(row.filename, cfg=cfg, on_stage=checkpoint)
        except Exception as exc:  # keep evaluating the other images
            logger.exception("evaluation failed unexpectedly for %s", image_id)
            response = PipelineResponse(
                image=row.filename,
                model_id=cfg.vision.model_id,
                status="error",
                stage_errors={"pipeline": str(exc)},
            )
        record = _record(row, cfg, key, version, response, response.status)
        write_record(out_path, record)
        written += 1
        errors += record["status"] != "ok"

    return {
        "model_id": cfg.vision.model_id,
        "dataset_version": version,
        "dataset_size": len(rows),
        "written": written,
        "errors": errors,
        "total_on_disk": sum(r.get("experiment_key") == key for r in load_records(out_path)),
        "path": str(out_path),
    }


def _record(
    row: DatasetRow,
    cfg: AppConfig,
    key: str,
    version: str,
    result: PipelineResponse,
    status: str,
) -> dict:
    """One paired result line; the ground truth is attached only here, after inference."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_key": key,
        "image_id": Path(row.filename).with_suffix("").as_posix(),
        "image_path": row.filename,
        "ground_truth_is_dog": row.label == "dog",
        "species": row.species,
        "model_id": cfg.vision.model_id,
        "model_repo": cfg.vision.model_repo,
        "dataset_version": version,
        "run_at": datetime.now(UTC).isoformat(),
        "status": status,
        "first_pass": result.first_pass.model_dump(mode="json") if result.first_pass else None,
        "retrieval_query": result.retrieval_query,
        "retrieved_documents": [
            {"document_id": item.document_id, "score": item.score}
            for item in result.retrieved_documents
        ],
        "retrieved_document_ids": result.retrieved_document_ids,
        "raw_outputs": result.raw_outputs,
        "second_pass": result.second_pass.model_dump(mode="json") if result.second_pass else None,
        "timings": result.timings,
        "stage_errors": result.stage_errors,
        "step_statuses": result.step_statuses.model_dump(mode="json"),
        "error": result.error,
    }
