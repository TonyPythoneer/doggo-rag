#!/usr/bin/env python3
"""Download every model needed by the evaluation into the local HF cache.

This is the only project command that is allowed to contact the Hugging Face
Hub. Inference resolves models strictly from the local cache and fails with a
download hint when something is missing.

Usage:
    .venv/bin/python scripts/download-models.py          # download / refresh
    .venv/bin/python scripts/download-models.py --check  # report cache status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import MODEL_REGISTRY, load_config
from app.vision.model import resolve_cached_snapshot


def targets() -> list[tuple[str, str]]:
    """Return (label, repo_id) for the three vision models + the embedder."""
    cfg = load_config()
    items = [(model_id, repo) for model_id, (_path, repo) in MODEL_REGISTRY.items()]
    items.append(("embedding", cfg.embedding.model_name))
    return items


def fetch(repo: str) -> Path | None:
    """Download ``repo`` into the HF cache; return its snapshot or None on failure."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id=repo))
    except Exception as exc:  # noqa: BLE001 - report every failed model, keep going
        print(f"         {type(exc).__name__}: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only inspect the local cache; never contact the Hub",
    )
    args = parser.parse_args(argv)

    missing: list[tuple[str, str]] = []
    for label, repo in targets():
        # --check uses the same local-only resolver as inference, so the check
        # can never disagree with what the model loader will actually find.
        path = resolve_cached_snapshot(repo) if args.check else fetch(repo)
        if path is not None:
            print(f"OK       {label}: {path}")
        else:
            missing.append((label, repo))
            print(f"MISSING  {label}: {repo}")

    if missing:
        if args.check:
            print("\nRun `make download` to fetch the missing models.")
        else:
            print(f"\n{len(missing)} model(s) failed to download.")
        return 1
    print("\nAll models are available locally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
