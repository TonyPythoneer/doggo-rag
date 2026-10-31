"""Paired First Pass and Second Pass evaluation over the dataset manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.config import RESULT_SCHEMA_VERSION, AppConfig, get_config
from app.dataset import DatasetRow, load_manifest
from app.evaluation.metrics import paired_classification_metrics
from app.pipeline.orchestrator import predict
from app.pipeline.schemas import PipelineResponse, RagStage

logger = logging.getLogger(__name__)


class SettingsChangedError(RuntimeError):
    """Existing results were produced with different experiment settings."""


def _digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def dataset_version(cfg: AppConfig) -> str:
    manifest = cfg.dataset.resolve_manifest()
    return _digest([manifest]) if manifest.is_file() else "unknown"


def result_path(cfg: AppConfig, output_dir: Path | None = None) -> Path:
    model_id = cfg.vision.model_id
    root = output_dir or cfg.results.resolve_dir() / model_id
    return root / f"{RESULT_SCHEMA_VERSION}-{model_id}-{dataset_version(cfg)}.jsonl"


def metadata_path(cfg: AppConfig, output_dir: Path | None = None) -> Path:
    return result_path(cfg, output_dir).with_suffix(".metadata.json")


def experiment_key(cfg: AppConfig) -> str:
    return "|".join(
        (
            RESULT_SCHEMA_VERSION,
            cfg.vision.model_id,
            cfg.vision.model_repo or "",
            dataset_version(cfg),
        )
    )


def experiment_metadata(cfg: AppConfig) -> dict:
    """Everything that decides a result, so runs can be reproduced and compared.

    Excluding `generated_at`, two runs with equal metadata are the same experiment.
    """
    vision = cfg.vision
    prompts = [cfg.prompts_dir / f"{name}.txt" for name in ("policy", "first_pass", "second_pass")]
    knowledge = sorted(cfg.knowledge.resolve_dir().glob("*.md"))
    app_dir = Path(__file__).resolve().parents[1]  # chunking, embedding text and context layout
    rag_code = [
        app_dir / "rag" / "chunking.py",
        app_dir / "rag" / "retriever.py",
        app_dir / "pipeline" / "rag_pipeline.py",
    ]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": RESULT_SCHEMA_VERSION,
        "vision_model": vision.model_id,
        "vision_repo": vision.model_repo,
        "vision_backend": vision.backend,
        "decoding": {
            "temperature": vision.temperature,
            "seed": vision.seed,
            "max_tokens_first_pass": vision.max_tokens_first_pass,
            "max_tokens_second_pass": vision.max_tokens_second_pass,
        },
        "prompts_sha256": _digest(prompts),
        "rag_code_sha256": _digest(rag_code),
        "knowledge_sha256": _digest(knowledge),
        "retrieval": {
            "query_source": "first_pass.description",
            "top_k": cfg.retrieval.top_k,
            "oversample": cfg.retrieval.oversample,
            "chunk_max_chars": cfg.knowledge.chunk_max_chars,
            "vector_store": "faiss(IndexFlatIP, cosine)",
            "embedding_model": cfg.embedding.model_name,
        },
        "dataset_manifest": cfg.dataset.manifest,
        "dataset_version": dataset_version(cfg),
    }


def _same_settings(left: dict, right: dict) -> bool:
    return {**left, "generated_at": None} == {**right, "generated_at": None}


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("ignoring malformed result record in %s", path)
    return records


def _write_record(path: Path, record: dict) -> None:
    """Atomically replace this image's record, keeping every other line."""
    key = (record["experiment_key"], record["image_id"])
    lines = [
        json.dumps(current, ensure_ascii=False)
        for current in load_records(path)
        if (current.get("experiment_key"), current.get("image_id")) != key
    ]
    lines.append(json.dumps(record, ensure_ascii=False))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


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
        if not _same_settings(previous, metadata):
            raise SettingsChangedError(
                f"{out_path} was produced with different settings (prompts, knowledge, "
                "retrieval or decoding). Rerun with `make eval FRESH=1`."
            )
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    done = {record["image_id"] for record in records if record.get("status") == "ok"}

    written = errors = 0
    for row in rows:
        image_id = Path(row.filename).with_suffix("").as_posix()
        if image_id in done:
            continue

        def checkpoint(stage: RagStage, row: DatasetRow = row) -> None:
            _write_record(out_path, _record(row, cfg, key, "in_progress", stage, stage.query))

        try:
            response = predict(row.filename, cfg=cfg, on_stage=checkpoint)
        except Exception as exc:
            logger.exception("evaluation failed unexpectedly for %s", image_id)
            record = _record(row, cfg, key, "error", RagStage(stage_errors={"pipeline": str(exc)}))
        else:
            record = _record(row, cfg, key, response.status, response, response.retrieval_query)
        _write_record(out_path, record)
        written += 1
        errors += record["status"] != "ok"

    return {
        "model_id": cfg.vision.model_id,
        "dataset_version": dataset_version(cfg),
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
    status: str,
    result: RagStage | PipelineResponse,
    query: str | None = None,
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
        "dataset_version": dataset_version(cfg),
        "run_at": datetime.now(UTC).isoformat(),
        "status": status,
        "first_pass": result.first_pass.model_dump(mode="json") if result.first_pass else None,
        "retrieval_query": query,
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
        "error": "; ".join(f"{name}: {msg}" for name, msg in result.stage_errors.items()) or None,
    }


def _mean_seconds(records: list[dict], key: str) -> dict:
    values = [
        r["timings"][key]
        for r in records
        if isinstance(r.get("timings", {}).get(key), (int, float))
    ]
    return {
        "count": len(values),
        "mean_seconds": round(sum(values) / len(values), 4) if values else None,
    }


def summarize(records: list[dict]) -> dict:
    """Score both passes on the common cohort of images where both passes succeeded."""
    paired = [
        record
        for record in records
        if record.get("status") == "ok" and record.get("first_pass") and record.get("second_pass")
    ]
    failed = [record for record in records if record.get("status") != "ok"]
    metrics: dict = {
        "images": len(records),
        "paired_scored": len(paired),
        "failed": len(failed),
        "failed_images": [record["image_id"] for record in failed],
        "stage_failures": dict(
            Counter(stage for record in failed for stage in record.get("stage_errors") or {})
        ),
        "inference_time": {
            stage: _mean_seconds(records, f"{stage}_s")
            for stage in ("first_pass", "retrieval", "second_pass")
        },
    }
    if paired:
        metrics["comparison"] = paired_classification_metrics(
            [record["ground_truth_is_dog"] for record in paired],
            [record["first_pass"]["is_dog"] for record in paired],
            [record["second_pass"]["is_dog"] for record in paired],
            [record["species"] for record in paired],
        )
    return metrics
