"""Strict parsing and pass contracts for the specified VLM outputs."""

from __future__ import annotations

import pytest

import app.vision.schemas as vision_schemas
from app.vision.inference import (
    VisionParseError,
    extract_json_object,
    run_first_pass,
    run_second_pass,
)


def _schema_type(name):
    schema = getattr(vision_schemas, name, None)
    assert schema is not None, f"schema {name!r} is missing"
    return schema


@pytest.mark.parametrize("text", ['{"a": 1}', '```json\n{"a": 1}\n```'])
def test_extract_json_object(text):
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_object_rejects_non_json():
    with pytest.raises(VisionParseError):
        extract_json_object("there is no json here")


def test_first_pass_accepts_null_as_an_explicit_abstention():
    result = _schema_type("FirstPassResult").model_validate(
        {"is_dog": None, "confidence": 0.42, "description": "Pointed ears and long muzzle."}
    )

    assert result.is_dog is None


@pytest.mark.parametrize(
    "payload",
    [
        {"is_dog": "false", "confidence": 0.7, "description": "Visible traits."},
        {"is_dog": False, "confidence": 1.1, "description": "Visible traits."},
        {"is_dog": False, "confidence": 0.7, "description": " "},
        {"is_dog": False, "confidence": 0.7},
    ],
)
def test_first_pass_rejects_malformed_fields(payload):
    with pytest.raises(ValueError):
        _schema_type("FirstPassResult").model_validate(payload)


def test_second_pass_schema_has_only_the_required_fields():
    schema = _schema_type("SecondPassResult")
    result = schema.model_validate({"is_dog": True, "confidence": 0.97})

    assert result.is_dog is True
    with pytest.raises(ValueError):
        schema.model_validate(
            {"is_dog": True, "confidence": 0.97, "description": "extra"}
        )


def test_first_and_second_pass_use_specified_prompts_and_fields(config, fake_vision):
    first, _, first_raw = run_first_pass("animal.jpg", cfg=config, model=fake_vision)
    second, _, second_raw = run_second_pass(
        "animal.jpg", first, "retrieved fox facts", cfg=config, model=fake_vision
    )

    assert extract_json_object(first_raw) == first.model_dump()
    assert extract_json_object(second_raw) == second.model_dump()

    assert first.is_dog is False
    assert first.confidence == 0.8
    assert "pointed muzzle" in first.description
    assert second.is_dog is False
    assert second.confidence == 0.9
    assert len(fake_vision.calls) == 2
    first_prompt, second_prompt = [call["prompt"] for call in fake_vision.calls]
    assert "visible" in first_prompt.lower()
    assert "retrieved fox facts" in second_prompt
    assert "pointed muzzle" in second_prompt
    assert "is_dog = false, confidence = 0.8" in second_prompt
    assert "dingo" in first_prompt.lower() and "dingo" in second_prompt.lower()
    assert all(call["image"] == "animal.jpg" for call in fake_vision.calls)


def test_invalid_boolean_is_not_coerced_to_a_classification(config, fake_vision):
    fake_vision.first = {"is_dog": "false", "confidence": 0.8, "description": "Visible traits."}

    with pytest.raises(VisionParseError):
        run_first_pass("animal.jpg", cfg=config, model=fake_vision)


def test_invalid_confidence_is_not_clamped(config, fake_vision):
    fake_vision.second = {"is_dog": False, "confidence": 150}
    first, _, _ = run_first_pass("animal.jpg", cfg=config, model=fake_vision)

    with pytest.raises(VisionParseError):
        run_second_pass("animal.jpg", first, "facts", cfg=config, model=fake_vision)


@pytest.mark.parametrize("enabled", [True, False])
def test_only_the_second_pass_reasons_and_only_when_configured(config, fake_vision, enabled):
    config.vision.second_pass_thinking = enabled
    first, _, _ = run_first_pass("animal.jpg", cfg=config, model=fake_vision)
    run_second_pass("animal.jpg", first, "facts", cfg=config, model=fake_vision)

    assert [call["thinking"] for call in fake_vision.calls] == [False, enabled]


def test_each_pass_decodes_with_its_own_schema(config, fake_vision):
    first, _, _ = run_first_pass("animal.jpg", cfg=config, model=fake_vision)
    run_second_pass("animal.jpg", first, "facts", cfg=config, model=fake_vision)

    schemas = [call["schema"] for call in fake_vision.calls]
    assert schemas == [vision_schemas.FirstPassResult, vision_schemas.SecondPassResult]
    # Decoding follows the field order: the First Pass describes before it classifies.
    assert list(vision_schemas.FirstPassResult.model_json_schema()["properties"]) == [
        "description",
        "is_dog",
        "confidence",
    ]


class _TextModel:
    """Returns fixed raw text, as a model that reasons before answering would."""

    def __init__(self, text):
        self.text = text

    def generate(self, image, prompt, max_tokens, thinking=False, schema=None):
        return self.text

    def usage(self, prompt_name, latency_s):
        return None


def test_reasoning_is_skipped_and_the_answer_after_it_is_parsed(config):
    text = (
        '<think>A draft {"is_dog": true, "confidence": 0.1} first.</think>'
        '{"is_dog": false, "confidence": 0.9}'
    )
    first = _schema_type("FirstPassResult").model_validate(
        {"is_dog": True, "confidence": 0.8, "description": "Pointed ears."}
    )

    model = _TextModel(text)
    second, _, raw = run_second_pass("animal.jpg", first, "facts", cfg=config, model=model)

    assert (second.is_dog, second.confidence) == (False, 0.9)
    assert raw == text


def test_unfinished_reasoning_fails_instead_of_using_a_draft_answer(config):
    text = '<think>Maybe {"is_dog": true, "confidence": 0.9, "description": "Ears."} but'

    with pytest.raises(VisionParseError, match="reasoning did not end") as caught:
        run_first_pass("animal.jpg", cfg=config, model=_TextModel(text))
    assert caught.value.raw_text == text


def test_a_parse_failure_after_reasoning_keeps_the_reasoning(config):
    text = "<think>The ears are erect.</think>not json"

    with pytest.raises(VisionParseError) as caught:
        run_first_pass("animal.jpg", cfg=config, model=_TextModel(text))
    assert caught.value.raw_text == text
