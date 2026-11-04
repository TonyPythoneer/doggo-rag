"""Dataset helpers: manifest loading and image path resolution."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig, get_config


class ImageNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class DatasetRow:
    filename: str  # path relative to the dataset root, e.g. images/dog/dog_001.jpg
    label: str  # "dog" | "non_dog"
    species: str
    license: str
    source_url: str
    author: str

    @property
    def image_name(self) -> str:
        return Path(self.filename).name


def load_manifest(cfg: AppConfig | None = None) -> list[DatasetRow]:
    cfg = cfg or get_config()
    manifest = cfg.dataset.resolve_manifest()
    if not manifest.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest}. "
            "The 200-image evaluation dataset is an external dependency "
            "(see README 'Quickstart')."
        )
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            DatasetRow(
                filename=row["filename"],
                label=row["label"],
                species=row["species"],
                license=row["license"],
                source_url=row["source_url"],
                author=row["author"],
            )
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise FileNotFoundError(f"Dataset manifest is empty: {manifest}")
    return rows


def resolve_image(image: str, cfg: AppConfig | None = None) -> Path:
    """Resolve an image reference: absolute path, relative path, or bare file
    name looked up anywhere under the dataset images directory."""
    cfg = cfg or get_config()
    candidate = Path(image)
    checked = []
    for path in (candidate, cfg.dataset.resolve_images_dir() / image):
        checked.append(path)
        if path.is_file():
            return path
    if not candidate.is_absolute():
        images_dir = cfg.dataset.resolve_images_dir()
        if images_dir.is_dir():
            matches = sorted(images_dir.rglob(candidate.name))
            if matches:
                return matches[0]
    raise ImageNotFoundError(
        f"Image not found: {image!r} (checked: {', '.join(str(p) for p in checked)})"
    )


def manifest_image_path(filename: str, cfg: AppConfig) -> Path:
    images_dir = cfg.dataset.resolve_images_dir().resolve()
    relative = Path(filename)
    if relative.parts and relative.parts[0] == "images":
        relative = Path(*relative.parts[1:])
    candidate = (images_dir / relative).resolve()
    if not candidate.is_relative_to(images_dir) or not candidate.is_file():
        raise ImageNotFoundError(f"Dataset image is unavailable: {filename!r}")
    return candidate


def list_dataset_images(cfg: AppConfig | None = None) -> list[dict[str, str]]:
    """Return stable references for the manifest's available images.

    Each entry carries the species only, for grouping in the picker. The binary
    `label` is never returned, and the species is never sent to inference.
    """
    cfg = cfg or get_config()
    images: list[dict[str, str]] = []
    for index, row in enumerate(load_manifest(cfg), start=1):
        try:
            manifest_image_path(row.filename, cfg)
        except ImageNotFoundError:
            continue
        image_id = f"image-{index:04d}"
        images.append(
            {
                "image_id": image_id,
                "name": f"Image {index}",
                "species": row.species,
                "preview_url": f"/api/v1/images/dataset/{image_id}",
            }
        )
    return images


def resolve_dataset_image_id(image_id: str, cfg: AppConfig | None = None) -> Path:
    """Resolve a stable catalog ID without accepting caller-provided paths."""
    cfg = cfg or get_config()
    match = re.fullmatch(r"image-(\d{4})", image_id)
    if not match:
        raise ImageNotFoundError(f"Dataset image not found: {image_id!r}")
    index = int(match.group(1))
    rows = load_manifest(cfg)
    if index < 1 or index > len(rows):
        raise ImageNotFoundError(f"Dataset image not found: {image_id!r}")
    return manifest_image_path(rows[index - 1].filename, cfg)


def resolve_image_reference(image: str, cfg: AppConfig | None = None) -> Path:
    """Resolve UI references, retaining path compatibility for CLI/evaluation."""
    if image.startswith("dataset:"):
        return resolve_dataset_image_id(image.removeprefix("dataset:"), cfg)
    if image.startswith("upload:"):
        from app.api.images import resolve_uploaded_image

        return resolve_uploaded_image(image.removeprefix("upload:"))
    return resolve_image(image, cfg)
