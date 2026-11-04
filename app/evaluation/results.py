"""Versioned JSONL result files: one record per image, replaced atomically."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import RESULT_SCHEMA_VERSION, AppConfig
from app.evaluation.metadata import dataset_version

logger = logging.getLogger(__name__)


def result_path(cfg: AppConfig, output_dir: Path | None = None) -> Path:
    model_id = cfg.vision.model_id
    root = output_dir or cfg.results.resolve_dir() / model_id
    return root / f"{RESULT_SCHEMA_VERSION}-{model_id}-{dataset_version(cfg)}.jsonl"


def metadata_path(cfg: AppConfig, output_dir: Path | None = None) -> Path:
    return result_path(cfg, output_dir).with_suffix(".metadata.json")


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


def write_record(path: Path, record: dict) -> None:
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
