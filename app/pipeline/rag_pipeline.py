"""Image → First Pass → description retrieval → Second Pass."""

from __future__ import annotations

import time
from collections.abc import Callable

from app.config import AppConfig, get_config
from app.pipeline.schemas import RagStage
from app.rag.retriever import Retriever, get_retriever
from app.rag.schemas import RetrievedDocument
from app.vision import inference
from app.vision.model import VisionModel


def assemble_context(documents: list[RetrievedDocument]) -> str:
    """Render retrieved knowledge for the Second Pass prompt."""
    if not documents:
        return "(no relevant knowledge was retrieved)"
    blocks = []
    for position, doc in enumerate(documents, start=1):
        blocks.append(f"[{position}] {doc.title}\n{doc.content}")
    return "\n\n".join(blocks)


def run_rag(
    image: str,
    cfg: AppConfig | None = None,
    model: VisionModel | None = None,
    retriever: Retriever | None = None,
    on_stage: Callable[[RagStage], None] | None = None,
) -> RagStage:
    """Run both passes, retaining all completed stages if a later stage fails."""
    cfg = cfg or get_config()
    stage = RagStage()

    try:
        first, first_usage, first_raw = inference.run_first_pass(image, cfg=cfg, model=model)
    except Exception as exc:
        stage.status = "error"
        stage.step_statuses.first_pass = "failed"
        stage.step_statuses.retrieval = "skipped"
        stage.step_statuses.second_pass = "skipped"
        stage.stage_errors["first_pass"] = str(exc)
        return stage
    stage.step_statuses.first_pass = "completed"
    stage.first_pass = first
    stage.usages["first_pass"] = first_usage
    stage.raw_outputs["first_pass"] = first_raw
    stage.timings["first_pass_s"] = first_usage.latency_s
    if on_stage:
        on_stage(stage)

    # Store the structured First Pass before retrieval; its description is the query.
    stage.query = first.description
    retrieval_started = time.perf_counter()
    try:
        retriever = retriever or get_retriever()
        documents = retriever.search(
            first.description,
            top_k=cfg.retrieval.top_k,
            oversample=cfg.retrieval.oversample,
        )
    except Exception as exc:
        stage.status = "error"
        stage.step_statuses.retrieval = "failed"
        stage.step_statuses.second_pass = "skipped"
        stage.stage_errors["retrieval"] = str(exc)
        stage.timings["retrieval_s"] = round(time.perf_counter() - retrieval_started, 3)
        return stage
    retrieval_s = round(time.perf_counter() - retrieval_started, 3)
    stage.retrieved_documents = documents
    stage.step_statuses.retrieval = "completed"
    stage.retrieved_document_ids = [doc.document_id for doc in documents]
    stage.timings["retrieval_s"] = retrieval_s
    if on_stage:
        on_stage(stage)

    context = assemble_context(documents)
    try:
        final, second_usage, second_raw = inference.run_second_pass(
            image, first, context, cfg=cfg, model=model
        )
    except Exception as exc:
        stage.status = "error"
        stage.step_statuses.second_pass = "failed"
        stage.stage_errors["second_pass"] = str(exc)
        return stage
    stage.step_statuses.second_pass = "completed"
    stage.second_pass = final
    stage.usages["second_pass"] = second_usage
    stage.raw_outputs["second_pass"] = second_raw
    stage.timings["second_pass_s"] = second_usage.latency_s
    stage.timings["total_s"] = round(sum(stage.timings.values()), 3)
    return stage
