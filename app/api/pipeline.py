"""Main user-facing endpoint: the complete E2E pipeline."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.api import require_api_key
from app.api.images import remove_uploaded_image
from app.config import DEFAULT_MODEL_ID, MODEL_REGISTRY
from app.pipeline.orchestrator import predict
from app.pipeline.schemas import PipelineResponse

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


class PredictRequest(BaseModel):
    image: str = Field(min_length=1, description="Dataset image reference or uploaded image token")
    model_id: Literal[*MODEL_REGISTRY] = DEFAULT_MODEL_ID

    @field_validator("image")
    @classmethod
    def require_managed_image_reference(cls, value: str) -> str:
        if not value.startswith(("dataset:", "upload:")):
            raise ValueError("image must be a dataset or upload reference")
        return value


@router.post("/predict", response_model=PipelineResponse, dependencies=[Depends(require_api_key)])
def run_pipeline(request: PredictRequest) -> PipelineResponse:
    """Run both inference passes over the same image using the selected VLM."""
    try:
        response = predict(request.image, model_id=request.model_id)
    finally:
        if request.image.startswith("upload:"):
            remove_uploaded_image(request.image.removeprefix("upload:"))
    return response.model_copy(update={"image": request.image})
