"""Runs the paired First Pass and RAG-assisted Second Pass workflow."""

from __future__ import annotations

import threading
from collections.abc import Callable

from app.config import AppConfig, ConfigError, get_config
from app.dataset import resolve_image_reference
from app.pipeline.rag_pipeline import run_rag
from app.pipeline.schemas import PipelineResponse
from app.rag.retriever import Retriever
from app.vision.model import VisionModel, VisionModelError, get_vision_model

# One pipeline at a time: the API runs requests in a thread pool, MLX makes no
# thread-safety promise for a shared model, and switching models must not leave two loaded.
_INFERENCE_LOCK = threading.Lock()


def _setup_failed(image: str, model_id: str, step: str, exc: Exception) -> PipelineResponse:
    """A failure before the First Pass: every pipeline step stays `not_started`."""
    return PipelineResponse(
        image=image, model_id=model_id, status="error", stage_errors={step: str(exc)}
    )


def predict(
    image: str,
    cfg: AppConfig | None = None,
    model: VisionModel | None = None,
    retriever: Retriever | None = None,
    model_id: str | None = None,
    on_stage: Callable[[PipelineResponse], None] | None = None,
) -> PipelineResponse:
    cfg = cfg or get_config()
    selected = model_id or cfg.vision.model_id
    if selected != cfg.vision.model_id:
        try:
            cfg = cfg.for_model(selected)
        except ConfigError as exc:
            return _setup_failed(image, selected, "model_selection", exc)
    try:
        image_path = str(resolve_image_reference(image, cfg))
    except FileNotFoundError as exc:
        return _setup_failed(image, selected, "image", exc)

    with _INFERENCE_LOCK:
        try:
            model = model or get_vision_model(selected)
        except VisionModelError as exc:
            return _setup_failed(image_path, selected, "model_loading", exc)
        response = run_rag(image_path, cfg=cfg, model=model, retriever=retriever, on_stage=on_stage)
    response.timings["total_s"] = round(sum(response.timings.values()), 3)
    return response
