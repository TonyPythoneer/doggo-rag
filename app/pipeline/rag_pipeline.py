"""Image → First Pass → description retrieval → Second Pass."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from app.config import AppConfig, get_config
from app.pipeline.schemas import PipelineResponse
from app.rag.retriever import Retriever, get_retriever
from app.rag.schemas import RetrievedDocument
from app.vision import inference
from app.vision.model import VisionModel

logger = logging.getLogger(__name__)

STEPS = ("first_pass", "retrieval", "second_pass")


def assemble_context(documents: list[RetrievedDocument]) -> str:
    """Render retrieved knowledge for the Second Pass prompt."""
    if not documents:
        return "(no relevant knowledge was retrieved)"
    blocks = []
    for position, doc in enumerate(documents, start=1):
        blocks.append(f"[{position}] {doc.title}\n{doc.content}")
    return "\n\n".join(blocks)


def _fail(response: PipelineResponse, step: str, exc: Exception) -> PipelineResponse:
    """Mark `step` failed and the later steps skipped; call from an `except` block."""
    logger.exception("%s failed for %s", step, response.image)
    response.status = "error"
    response.stage_errors[step] = str(exc)
    for later in STEPS[STEPS.index(step) :]:
        setattr(response.step_statuses, later, "failed" if later == step else "skipped")
    return response


def run_rag(
    image: str,
    cfg: AppConfig | None = None,
    model: VisionModel | None = None,
    retriever: Retriever | None = None,
    on_stage: Callable[[PipelineResponse], None] | None = None,
) -> PipelineResponse:
    """Run both passes, retaining all completed steps if a later step fails.

    Each step catches every exception: MLX, the embedder and FAISS raise their own
    types, and a failure must become a recorded step error, never a crash or a guess.
    """
    cfg = cfg or get_config()
    response = PipelineResponse(image=image, model_id=cfg.vision.model_id)

    try:
        first, usage, raw = inference.run_first_pass(image, cfg=cfg, model=model)
    except Exception as exc:
        return _fail(response, "first_pass", exc)
    response.first_pass = first
    response.usages["first_pass"] = usage
    response.raw_outputs["first_pass"] = raw
    response.timings["first_pass_s"] = usage.latency_s
    response.step_statuses.first_pass = "completed"
    if on_stage:
        on_stage(response)

    # Store the structured First Pass before retrieval; its description is the query.
    response.retrieval_query = first.description
    started = time.perf_counter()
    try:
        documents = (retriever or get_retriever()).search(
            first.description, top_k=cfg.retrieval.top_k, oversample=cfg.retrieval.oversample
        )
    except Exception as exc:
        return _fail(response, "retrieval", exc)
    finally:
        response.timings["retrieval_s"] = round(time.perf_counter() - started, 3)
    response.retrieved_documents = documents
    response.step_statuses.retrieval = "completed"
    if on_stage:
        on_stage(response)

    try:
        final, usage, raw = inference.run_second_pass(
            image, first, assemble_context(documents), cfg=cfg, model=model
        )
    except Exception as exc:
        return _fail(response, "second_pass", exc)
    response.second_pass = final
    response.usages["second_pass"] = usage
    response.raw_outputs["second_pass"] = raw
    response.timings["second_pass_s"] = usage.latency_s
    response.step_statuses.second_pass = "completed"
    return response
