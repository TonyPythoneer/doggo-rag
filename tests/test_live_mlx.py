"""Real MLX inference through the full paired pipeline, one test per supported model.

Excluded by default; run with `uv run pytest -m live` after `make download` and
`python -m app.cli ingest` on Apple Silicon.
"""

from __future__ import annotations

import pytest

from app.config import MODEL_REGISTRY
from app.pipeline.orchestrator import predict

pytestmark = pytest.mark.live

# Known failure with the current prompts (2026-10-31): lfm2-vl-450m answers
# {"is_dog": ..., "confidence": ...} without "description" (6 of 6 sampled images).
LFM2_DROPS_DESCRIPTION = pytest.mark.xfail(
    reason="lfm2-vl-450m omits the First Pass description with the current prompt", strict=True
)


@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param(model_id, marks=LFM2_DROPS_DESCRIPTION if model_id == "lfm2-vl-450m" else ())
        for model_id in MODEL_REGISTRY
    ],
)
def test_paired_pipeline_runs_on_the_real_model(model_id):
    response = predict("doggo-dataset/images/dog/dog_033.jpg", model_id=model_id)

    assert response.stage_errors == {}
    assert response.status == "ok"
    assert response.step_statuses.model_dump() == {
        "first_pass": "completed",
        "retrieval": "completed",
        "second_pass": "completed",
    }
    assert response.retrieval_query == response.first_pass.description
    assert response.retrieved_document_ids
