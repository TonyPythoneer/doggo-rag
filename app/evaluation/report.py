"""Render the paired First Pass / Second Pass summary as an HTML report (spec §9)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Undefined

_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

# Same look and wording as the web app (frontend/src/index.css and App.tsx), so
# the evaluation reads like the demo. Self-contained: no network, no build step.

MODEL_LABELS = {
    "lfm3-vl-2b": "LFM3 VL 2B (Default)",
    "lfm2-vl-450m": "LFM2 VL 450M",
    "qwen3-vl-2b": "Qwen3 VL 2B",
}
SPECIES_HEADINGS = {
    "bear": "Bears",
    "cat": "Cats",
    "coyote": "Coyotes",
    "dingo": "Dingoes",
    "dog": "Dogs",
    "fox": "Foxes",
    "hyena": "Hyenas",
    "koala": "Koalas",
    "raccoon": "Raccoons",
    "red_panda": "Red pandas",
    "wolf": "Wolves",
}
STAGE_LABELS = {
    "first_pass": "First Pass",
    "retrieval": "RAG retrieval",
    "second_pass": "Second Pass",
    "image": "Image loading",
    "model_loading": "Model loading",
    "pipeline": "Pipeline",
}

SCORES = [
    ("Precision", "When it says “dog”, how often it is right", "precision"),
    ("Recall", "Of all dog photos, how many it found", "recall"),
    ("F1", "Balance of precision and recall", "f1"),
]
EMPTY_MATRIX = [[0, 0, 0], [0, 0, 0]]


def _missing(value: Any) -> bool:
    return value is None or isinstance(value, Undefined)


def _fmt(value: Any) -> str:
    if _missing(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _pct(value: Any) -> str:
    return "-" if _missing(value) else f"{value * 100:.1f}%"


def _seconds(value: Any) -> str:
    return "-" if _missing(value) else f"{value:.2f} s"


_ENV.filters.update(fmt=_fmt, pct=_pct, seconds=_seconds)


def build_report(summary_path: Path, output_path: Path) -> Path:
    """Write the HTML report."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_path.write_text(render_html(summary), encoding="utf-8")
    return output_path


def _model_view(model_id: str, experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare one model's numbers for the template; layout lives in the template."""
    comparison = experiment.get("comparison") or {}
    first = comparison.get("first_pass") or {}
    second = comparison.get("second_pass") or {}
    timing = experiment.get("inference_time") or {}
    metadata = experiment.get("metadata") or {}
    decoding = metadata.get("decoding") or {}
    retrieval = metadata.get("retrieval") or {}
    before = first.get("per_species_accuracy") or {}
    after = second.get("per_species_accuracy") or {}
    species = [
        {
            "label": SPECIES_HEADINGS.get(name, name),
            "before": before[name],
            "after": after.get(name),
            "change": None if after.get(name) is None else (after[name] - before[name]) * 100,
        }
        for name in sorted(before, key=lambda n: (n != "dog", SPECIES_HEADINGS.get(n, n)))
    ]
    settings = [
        ("Model repo", metadata.get("vision_repo")),
        ("Decoding", f"temperature {decoding.get('temperature')}, seed {decoding.get('seed')}"),
        ("Embedding model", retrieval.get("embedding_model")),
        ("Retrieved documents", f"top {retrieval.get('top_k')}, query = First Pass description"),
        ("Prompts digest", metadata.get("prompts_sha256")),
        ("Knowledge digest", metadata.get("knowledge_sha256")),
        ("Dataset version", experiment.get("dataset_version")),
        ("Run started", metadata.get("generated_at")),
        ("Result file", experiment.get("result_file")),
    ]
    return {
        "label": MODEL_LABELS.get(model_id, model_id),
        "experiment": experiment,
        "comparison": comparison,
        "first": first,
        "second": second,
        "moves": comparison.get("transitions") or {},
        "gain": comparison.get("accuracy_improvement_percentage_points"),
        "matrices": [
            (m.get("confusion_matrix") or {}).get("matrix") or EMPTY_MATRIX for m in (first, second)
        ],
        "species": species,
        "stage_failures": [
            (STAGE_LABELS.get(stage, stage), count)
            for stage, count in sorted((experiment.get("stage_failures") or {}).items())
        ],
        "speed": [
            (STAGE_LABELS[stage], (timing.get(stage) or {}).get("mean_seconds"))
            for stage in ("first_pass", "retrieval", "second_pass")
        ],
        "settings": [(key, _fmt(value)) for key, value in settings],
    }


def render_html(summary: Mapping[str, Any]) -> str:
    models = [_model_view(m, e) for m, e in (summary.get("experiments") or {}).items()]
    return _ENV.get_template("report.html.j2").render(models=models, scores=SCORES).lstrip()
