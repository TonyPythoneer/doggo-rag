#!/usr/bin/env python3
"""Self-test loop for prompt and RAG changes: run the real pipeline on a fixed split and score it.

A development aid, not the reported evaluation (`make eval`). The split is fixed by image
number, so changes are tuned on `dev` (even numbers) and checked on `test` (odd numbers).
Model outputs are cached by model, decoding settings, image, prompt and token limit in
`results/lab/cache/`, so a Second Pass change reruns only the Second Pass. The cache keys
on inputs, not code: delete it after changing how `VisionModel.generate` works. The ground
truth is attached only after inference, exactly as in `make eval`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import MODEL_REGISTRY, load_config
from app.dataset import load_manifest
from app.evaluation.metadata import dataset_version, experiment_key
from app.evaluation.metrics import summarize
from app.evaluation.results import load_records
from app.evaluation.runner import _record
from app.pipeline.orchestrator import predict
from app.rag.ingestion import build_index
from app.vision.model import SCHEMA_OPTIONS, VisionModel

LAB_DIR = ROOT / "results" / "lab"


def in_split(filename: str, split: str) -> bool:
    if split == "all":
        return True
    number = int(Path(filename).stem.rsplit("_", 1)[1])
    return (number % 2 == 0) == (split == "dev")


def cache_generate(cache_file: Path) -> None:
    """Replace VisionModel.generate with a disk-cached version (greedy decoding is deterministic)."""
    cache: dict[str, str] = {}
    if cache_file.is_file():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            cache[item["key"]] = item["text"]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    generate = VisionModel.generate

    def cached(
        self: VisionModel,
        image_path: str,
        prompt: str,
        max_tokens: int,
        thinking: bool = False,
        schema: type[BaseModel] | None = None,
    ) -> str:
        vision = self.config.vision
        parts = [vision.model_id, vision.temperature, vision.seed, max_tokens, prompt]
        if thinking:
            parts.append("thinking")
        if schema is not None:
            parts.append([schema.model_json_schema(), SCHEMA_OPTIONS])
        parts.append(os.path.relpath(image_path, ROOT))
        key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
        if key not in cache:
            cache[key] = generate(
                self, image_path, prompt, max_tokens, thinking=thinking, schema=schema
            )
            with cache_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "text": cache[key]}) + "\n")
        return cache[key]

    VisionModel.generate = cached


def report(name: str, records: list[dict]) -> dict:
    summary = summarize(records)
    comparison = summary.get("comparison", {})
    first = comparison.get("first_pass", {}).get("accuracy")
    second = comparison.get("second_pass", {}).get("accuracy")
    scored = [r for r in records if r["status"] == "ok"]
    own_top1 = sum(r["retrieved_document_ids"][:1] == [r["species"]] for r in scored)
    own_top5 = sum(r["species"] in r["retrieved_document_ids"] for r in scored)
    changed = sum(r["first_pass"]["is_dog"] != r["second_pass"]["is_dog"] for r in scored)
    fixed = [
        r["image_id"] for r in scored if _correct(r, "second_pass") > _correct(r, "first_pass")
    ]
    broken = [
        r["image_id"] for r in scored if _correct(r, "second_pass") < _correct(r, "first_pass")
    ]
    wrong = [r["image_id"] for r in scored if not _correct(r, "second_pass")]
    line = {
        "name": name,
        "images": len(records),
        "failed": summary["failed"],
        "first_acc": first,
        "second_acc": second,
        "delta_pp": comparison.get("accuracy_improvement_percentage_points"),
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "second_wrong": wrong,
        "failed_images": summary["failed_images"],
        "own_species_top1": f"{own_top1}/{len(scored)}",
        "own_species_top5": f"{own_top5}/{len(scored)}",
    }
    print(json.dumps(line, indent=1), flush=True)
    return line


def _correct(record: dict, step: str) -> bool:
    return record[step]["is_dog"] is record["ground_truth_is_dog"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="run name; records go to results/lab/<name>-<split>.jsonl")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="lfm3-vl-2b")
    parser.add_argument("--score", type=Path, help="only score this result JSONL (e.g. make eval)")
    args = parser.parse_args(argv)

    if args.score:
        records = load_records(args.score)
        report(args.name, [r for r in records if in_split(r["image_path"], args.split)])
        return 0
    cfg = load_config().for_model(args.model)
    build_index(cfg)  # the index must match the knowledge files in this checkout
    cache_generate(LAB_DIR / "cache" / f"{args.model}.jsonl")
    key, version = experiment_key(cfg), dataset_version(cfg)
    records = []
    for row in load_manifest(cfg):
        if in_split(row.filename, args.split):
            response = predict(row.filename, cfg=cfg)
            records.append(_record(row, cfg, key, version, response, response.status))
    out = LAB_DIR / f"{args.name}-{args.split}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    report(f"{args.name}-{args.split}", records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
