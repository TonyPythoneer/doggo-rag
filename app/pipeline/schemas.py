"""Paired E2E pipeline response schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field

from app.rag.schemas import RetrievedDocument
from app.vision.schemas import FirstPassResult, SecondPassResult, VisionUsage

Status = Literal["ok", "error"]
StepStatus = Literal["not_started", "completed", "failed", "skipped"]


class PipelineStepStatuses(BaseModel):
    first_pass: StepStatus = "not_started"
    retrieval: StepStatus = "not_started"
    second_pass: StepStatus = "not_started"


class PipelineResponse(BaseModel):
    """Partial or complete two-pass run; completed steps survive a later failure."""

    image: str
    model_id: str
    status: Status = "ok"
    first_pass: FirstPassResult | None = None
    retrieval_query: str | None = None
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    second_pass: SecondPassResult | None = None
    usages: dict[str, VisionUsage] = Field(default_factory=dict)
    raw_outputs: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    stage_errors: dict[str, str] = Field(default_factory=dict)
    step_statuses: PipelineStepStatuses = Field(default_factory=PipelineStepStatuses)

    @computed_field
    @property
    def retrieved_document_ids(self) -> list[str]:
        return [document.document_id for document in self.retrieved_documents]

    @computed_field
    @property
    def error(self) -> str | None:
        errors = [f"{step}: {message}" for step, message in self.stage_errors.items()]
        return "; ".join(errors) or None
