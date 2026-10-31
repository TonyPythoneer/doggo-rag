"""Paired E2E pipeline response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.rag.schemas import RetrievedDocument
from app.vision.schemas import FirstPassResult, SecondPassResult, VisionUsage

Status = Literal["ok", "error"]
StepStatus = Literal["not_started", "completed", "failed", "skipped"]


class PipelineStepStatuses(BaseModel):
    first_pass: StepStatus = "not_started"
    retrieval: StepStatus = "not_started"
    second_pass: StepStatus = "not_started"


class RagStage(BaseModel):
    """Partial or complete two-pass run, including errors from each stage."""

    status: Status = "ok"
    first_pass: FirstPassResult | None = None
    query: str | None = None
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    second_pass: SecondPassResult | None = None
    usages: dict[str, VisionUsage] = Field(default_factory=dict)
    raw_outputs: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    stage_errors: dict[str, str] = Field(default_factory=dict)
    step_statuses: PipelineStepStatuses = Field(default_factory=PipelineStepStatuses)


class PipelineResponse(BaseModel):
    image: str
    model_id: str
    status: Status
    error: str | None = None
    first_pass: FirstPassResult | None = None
    retrieval_query: str | None = None
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    second_pass: SecondPassResult | None = None
    usages: dict[str, VisionUsage] = Field(default_factory=dict)
    raw_outputs: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    stage_errors: dict[str, str] = Field(default_factory=dict)
    step_statuses: PipelineStepStatuses = Field(default_factory=PipelineStepStatuses)
