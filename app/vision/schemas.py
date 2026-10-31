"""Structured schemas for the Vision Model's two inference passes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator


class FirstPassResult(BaseModel):
    """Output of the First Pass: classification and visual description."""

    model_config = ConfigDict(extra="forbid", strict=True)

    is_dog: StrictBool | None
    confidence: float = Field(ge=0.0, le=1.0, strict=True)
    description: StrictStr = Field(min_length=1)

    @field_validator("description")
    @classmethod
    def description_must_contain_visible_details(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("description must contain visible physical characteristics")
        return value


class SecondPassResult(BaseModel):
    """Output of the Second Pass: final dog classification only."""

    model_config = ConfigDict(extra="forbid", strict=True)

    is_dog: StrictBool | None
    confidence: float = Field(ge=0.0, le=1.0, strict=True)


class VisionUsage(BaseModel):
    """Model identity and decoding settings recorded with every inference."""

    model_id: str
    backend: str
    temperature: float
    seed: int
    prompt_name: str
    latency_s: float
