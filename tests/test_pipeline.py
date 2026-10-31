"""Paired two-pass pipeline tests."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.pipeline.orchestrator import predict
from app.pipeline.rag_pipeline import assemble_context
from tests.conftest import FakeVisionModel

IMAGE = "doggo-dataset/images/non_dog/fox/fox_001.jpg"


class RecordingRetriever:
    def __init__(self, retriever, fail=False):
        self.retriever = retriever
        self.queries = []
        self.fail = fail

    def search(self, query, top_k, oversample=3):
        self.queries.append((query, top_k))
        if self.fail:
            raise RuntimeError("retrieval exploded")
        return self.retriever.search(query, top_k, oversample)


def test_pipeline_uses_first_pass_description_and_same_image_and_model(
    config, fake_vision, tmp_index
):
    retriever = RecordingRetriever(tmp_index)

    response = predict(IMAGE, cfg=config, model=fake_vision, retriever=retriever)

    assert response.status == "ok"
    assert response.step_statuses.model_dump() == {
        "first_pass": "completed",
        "retrieval": "completed",
        "second_pass": "completed",
    }
    assert response.first_pass.description
    assert retriever.queries == [(response.first_pass.description, config.retrieval.top_k)]
    assert response.second_pass.is_dog is False
    assert response.retrieved_document_ids
    assert response.retrieved_documents
    second_prompt = fake_vision.calls[1]["prompt"]
    assert response.first_pass.description in second_prompt
    assert response.retrieved_documents[0].title in second_prompt
    assert len(fake_vision.calls) == 2
    assert all(call["image"] == str(IMAGE) for call in fake_vision.calls)
    assert set(response.raw_outputs) == {"first_pass", "second_pass"}
    assert all(response.raw_outputs.values())


def test_first_pass_failure_skips_retrieval_and_second_pass(config, fake_vision, tmp_index):
    fake_vision.first = {"is_dog": "invalid", "confidence": 0.8, "description": "traits"}
    retriever = RecordingRetriever(tmp_index)

    response = predict(IMAGE, cfg=config, model=fake_vision, retriever=retriever)

    assert response.status == "error"
    assert response.step_statuses.model_dump() == {
        "first_pass": "failed",
        "retrieval": "skipped",
        "second_pass": "skipped",
    }
    assert response.first_pass is None
    assert response.stage_errors["first_pass"]
    assert retriever.queries == []
    assert len(fake_vision.calls) == 1


def test_retrieval_failure_preserves_first_pass_and_skips_second_pass(
    config, fake_vision, tmp_index
):
    response = predict(
        IMAGE, cfg=config, model=fake_vision, retriever=RecordingRetriever(tmp_index, fail=True)
    )

    assert response.status == "error"
    assert response.step_statuses.model_dump() == {
        "first_pass": "completed",
        "retrieval": "failed",
        "second_pass": "skipped",
    }
    assert response.first_pass is not None
    assert response.stage_errors["retrieval"] == "retrieval exploded"
    assert response.second_pass is None
    assert len(fake_vision.calls) == 1


def test_second_pass_failure_preserves_first_pass_and_retrieval(config, fake_vision, tmp_index):
    fake_vision.second = {"is_dog": False, "confidence": 2.0}

    response = predict(IMAGE, cfg=config, model=fake_vision, retriever=tmp_index)

    assert response.status == "error"
    assert response.step_statuses.model_dump() == {
        "first_pass": "completed",
        "retrieval": "completed",
        "second_pass": "failed",
    }
    assert response.first_pass is not None
    assert response.retrieved_documents
    assert response.retrieved_document_ids
    assert response.stage_errors["second_pass"]
    assert response.second_pass is None
    assert len(fake_vision.calls) == 2


def test_missing_image_reports_error(config, fake_vision, tmp_index):
    response = predict("nope/missing.jpg", cfg=config, model=fake_vision, retriever=tmp_index)

    assert response.status == "error"
    assert "not found" in (response.error or "").lower()


def test_assemble_context_handles_empty():
    assert "no relevant knowledge" in assemble_context([])


def test_concurrent_predictions_run_one_at_a_time(config, tmp_index):
    active = peak = 0
    guard = threading.Lock()

    class SlowModel(FakeVisionModel):
        def generate(self, image, prompt, max_tokens):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return super().generate(image, prompt, max_tokens)

    def run(_):
        return predict(IMAGE, cfg=config, model=SlowModel(), retriever=tmp_index)

    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(run, range(3)))

    assert peak == 1
    assert [response.status for response in responses] == ["ok"] * 3


def test_unknown_model_is_a_setup_error_with_no_step_started(config, fake_vision, tmp_index):
    response = predict(IMAGE, cfg=config, model=fake_vision, retriever=tmp_index, model_id="nope")

    assert response.status == "error"
    assert list(response.stage_errors) == ["model_selection"]
    assert set(response.step_statuses.model_dump().values()) == {"not_started"}
    assert fake_vision.calls == []


def test_step_failure_logs_the_traceback(config, fake_vision, tmp_index, caplog):
    response = predict(
        IMAGE, cfg=config, model=fake_vision, retriever=RecordingRetriever(tmp_index, fail=True)
    )

    assert response.error == "retrieval: retrieval exploded"
    failure = next(record for record in caplog.records if record.levelname == "ERROR")
    assert failure.exc_info and "retrieval exploded" in str(failure.exc_info[1])
