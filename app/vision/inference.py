"""Two-pass Vision Model inference: First Pass and Second Pass.

Both passes use the same VisionModel instance and decoding settings.
Outputs are parsed into strict structured schemas; parsing failures raise
`VisionParseError` so callers can record the failure instead of guessing.
"""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, ValidationError

from app.config import AppConfig, get_config
from app.vision import prompts
from app.vision.model import THINK_CLOSE, THINK_OPEN, VisionModel, get_vision_model
from app.vision.schemas import FirstPassResult, SecondPassResult, VisionUsage


class VisionParseError(ValueError):
    """The model did not return the required structured JSON."""

    def __init__(self, message: str, raw_text: str | None = None):
        super().__init__(message)
        self.raw_text = raw_text


def extract_json_object(text: str) -> dict:
    """Extract the first decodable JSON object from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned
        cleaned = cleaned.removeprefix("json").strip()
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            start = cleaned.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = cleaned.find("{", start + 1)
    raise VisionParseError(f"No JSON object found in model output: {text[:200]!r}", text)


def _run_pass[Result: BaseModel](
    model: VisionModel,
    image: str,
    prompt_name: str,
    prompt: str,
    max_tokens: int,
    schema: type[Result],
    thinking: bool = False,
) -> tuple[Result, VisionUsage, str]:
    """Generate, then parse and validate; malformed output raises instead of being guessed."""
    started = time.perf_counter()
    text = model.generate(image, prompt, max_tokens, thinking=thinking, schema=schema)
    latency = time.perf_counter() - started
    answer = text
    if text.startswith(THINK_OPEN):
        # Only the text after the reasoning is the answer; a draft inside it never counts.
        _, closed, answer = text.partition(THINK_CLOSE)
        if not closed:
            raise VisionParseError(
                f"{prompt_name} reasoning did not end within the token limit", text
            )
    try:
        raw = extract_json_object(answer)
    except VisionParseError as exc:
        exc.raw_text = text  # keep the reasoning for debugging
        raise
    try:
        result = schema.model_validate(raw)
    except ValidationError as exc:
        raise VisionParseError(f"{prompt_name} output failed validation: {raw}", text) from exc
    return result, model.usage(prompt_name, latency), text


def run_first_pass(
    image: str,
    cfg: AppConfig | None = None,
    model: VisionModel | None = None,
) -> tuple[FirstPassResult, VisionUsage, str]:
    cfg = cfg or get_config()
    model = model or get_vision_model(cfg.vision.model_id)
    prompt = prompts.render(cfg, "first_pass")
    return _run_pass(
        model, image, "first_pass", prompt, cfg.vision.max_tokens_first_pass, FirstPassResult
    )


def run_second_pass(
    image: str,
    first: FirstPassResult,
    retrieved_context: str,
    cfg: AppConfig | None = None,
    model: VisionModel | None = None,
) -> tuple[SecondPassResult, VisionUsage, str]:
    cfg = cfg or get_config()
    model = model or get_vision_model(cfg.vision.model_id)
    prompt = prompts.render(
        cfg,
        "second_pass",
        is_dog=json.dumps(first.is_dog),
        confidence=str(first.confidence),
        description=first.description,
        retrieved_context=retrieved_context,
    )
    return _run_pass(
        model,
        image,
        "second_pass",
        prompt,
        cfg.vision.max_tokens_second_pass,
        SecondPassResult,
        thinking=cfg.vision.second_pass_thinking,
    )
