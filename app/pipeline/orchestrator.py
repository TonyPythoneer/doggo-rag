"""Runs the paired First Pass and RAG-assisted Second Pass workflow."""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.config import MODEL_REGISTRY, AppConfig, get_config
from app.dataset import resolve_image_reference
from app.pipeline.rag_pipeline import run_rag
from app.pipeline.schemas import PipelineResponse, RagStage
from app.rag.retriever import Retriever
from app.vision.model import VisionModel, VisionModelError, get_vision_model

logger = logging.getLogger(__name__)


def predict(
    image: str,
    cfg: AppConfig | None = None,
    model: VisionModel | None = None,
    retriever: Retriever | None = None,
    model_id: str | None = None,
    on_stage: Callable[[RagStage], None] | None = None,
) -> PipelineResponse:
    cfg = cfg or get_config()
    selected_model_id = model_id or cfg.vision.model_id
    if selected_model_id not in MODEL_REGISTRY:
        message = f"Unknown vision model ID: {selected_model_id!r}"
        return PipelineResponse(
            image=image,
            model_id=selected_model_id,
            status="error",
            error=message,
            stage_errors={"model_selection": message},
        )
    if selected_model_id != cfg.vision.model_id:
        cfg = cfg.for_model(selected_model_id)
    try:
        image_path = resolve_image_reference(image, cfg)
    except FileNotFoundError as exc:
        return PipelineResponse(
            image=image,
            model_id=cfg.vision.model_id,
            status="error",
            error=str(exc),
            stage_errors={"image": str(exc)},
        )
    try:
        model = model or get_vision_model(cfg.vision.model_id)
    except VisionModelError as exc:
        return PipelineResponse(
            image=str(image_path),
            model_id=cfg.vision.model_id,
            status="error",
            error=str(exc),
            stage_errors={"model_loading": str(exc)},
        )

    stage = run_rag(
        str(image_path), cfg=cfg, model=model, retriever=retriever, on_stage=on_stage
    )
    if stage.stage_errors:
        logger.error("paired inference failed for %s: %s", image, stage.stage_errors)
    error = "; ".join(f"{name}: {message}" for name, message in stage.stage_errors.items())
    timings = dict(stage.timings)
    timings.setdefault("total_s", round(sum(timings.values()), 3))
    return PipelineResponse(
        image=str(image_path),
        model_id=cfg.vision.model_id,
        status=stage.status,
        error=error or None,
        first_pass=stage.first_pass,
        retrieval_query=stage.query,
        retrieved_documents=stage.retrieved_documents,
        retrieved_document_ids=stage.retrieved_document_ids,
        second_pass=stage.second_pass,
        usages=stage.usages,
        raw_outputs=stage.raw_outputs,
        timings=timings,
        stage_errors=stage.stage_errors,
        step_statuses=stage.step_statuses,
    )
