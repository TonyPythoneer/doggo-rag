"""What identifies an experiment: settings, fingerprints and the environment of a run."""

from __future__ import annotations

import hashlib
import platform
from datetime import UTC, datetime
from importlib import metadata as package_metadata
from pathlib import Path

from app.config import RESULT_SCHEMA_VERSION, AppConfig
from app.dataset import ImageNotFoundError, load_manifest, manifest_image_path
from app.vision.model import resolve_cached_snapshot, resolve_model_source

# Installed packages whose version can change inference, embeddings or retrieval.
PACKAGES = (
    "mlx",
    "mlx-vlm",
    "transformers",
    "torch",
    "sentence-transformers",
    "faiss-cpu",
    "numpy",
    "pillow",
    "pydantic",
)


def _digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def dataset_version(cfg: AppConfig) -> str:
    """Digest of the manifest and every image it lists, so edited pixels change the version."""
    try:
        rows = load_manifest(cfg)
    except FileNotFoundError:  # missing or empty manifest
        return "unknown"
    images = []
    for row in rows:
        try:
            images.append(manifest_image_path(row.filename, cfg))
        except ImageNotFoundError:
            continue  # a missing image drops its bytes, which still changes the digest
    return _digest([cfg.dataset.resolve_manifest(), *images])


def _revision(source: Path | None) -> str | None:
    """Hugging Face cache snapshot directories are named after the commit hash."""
    return source.name if source and source.parent.name == "snapshots" else None


def _package_version(name: str) -> str | None:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


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
    # Model call, output parsing, chunking, embedding, retrieval and context layout.
    app_dir = Path(__file__).resolve().parents[1]
    code = sorted(
        path
        for package in ("rag", "pipeline", "vision")
        for path in (app_dir / package).glob("*.py")
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": RESULT_SCHEMA_VERSION,
        "vision_model": vision.model_id,
        "vision_repo": vision.model_repo,
        "vision_revision": _revision(resolve_model_source(vision)),
        "vision_backend": vision.backend,
        "decoding": {
            "temperature": vision.temperature,
            "seed": vision.seed,
            "max_tokens_first_pass": vision.max_tokens_first_pass,
            "max_tokens_second_pass": vision.max_tokens_second_pass,
        },
        "prompts_sha256": _digest(prompts),
        "code_sha256": _digest(code),
        "knowledge_sha256": _digest(knowledge),
        "retrieval": {
            "query_source": "first_pass.description",
            "top_k": cfg.retrieval.top_k,
            "oversample": cfg.retrieval.oversample,
            "chunk_max_chars": cfg.knowledge.chunk_max_chars,
            "vector_store": "faiss(IndexFlatIP, cosine)",
            "embedding_model": cfg.embedding.model_name,
            "embedding_revision": _revision(resolve_cached_snapshot(cfg.embedding.model_name)),
        },
        "dataset_manifest": cfg.dataset.manifest,
        "dataset_version": dataset_version(cfg),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: _package_version(name) for name in PACKAGES},
        },
    }


def same_settings(left: dict, right: dict) -> bool:
    return {**left, "generated_at": None} == {**right, "generated_at": None}
