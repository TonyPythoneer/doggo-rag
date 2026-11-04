"""Metrics for paired First Pass and Second Pass outputs."""

from __future__ import annotations

from app.evaluation import metrics


def _paired_metrics(*args, **kwargs):
    function = getattr(metrics, "paired_classification_metrics", None)
    assert function is not None, "paired_classification_metrics is missing"
    return function(*args, **kwargs)


def test_paired_metrics_score_null_as_wrong_and_report_transitions():
    result = _paired_metrics(
        y_true=[True, False, True, False, True, False],
        first=[False, False, None, True, True, False],
        second=[True, True, False, False, None, False],
        species=["dog", "fox", "wolf", "dingo", "cat", "koala"],
    )

    assert result["first_pass"]["accuracy"] == 0.5
    assert result["second_pass"]["accuracy"] == 0.5
    assert result["accuracy_improvement_percentage_points"] == 0.0
    assert result["transitions"] == {
        "incorrect_to_correct": 2,
        "correct_to_incorrect": 2,
        "unchanged_correct": 1,
        "unchanged_incorrect": 1,
    }
    assert result["first_pass"]["abstentions"] == 1
    assert result["second_pass"]["abstentions"] == 1
    assert result["first_pass"]["confusion_matrix"] == {
        "true_labels": [True, False],
        "predicted_labels": [True, False, None],
        "matrix": [[1, 1, 1], [1, 2, 0]],
    }
    assert result["second_pass"]["per_species_accuracy"] == {
        "cat": 0.0,
        "dingo": 1.0,
        "dog": 1.0,
        "fox": 0.0,
        "koala": 1.0,
        "wolf": 0.0,
    }


def test_paired_metrics_reject_misaligned_inputs():
    try:
        _paired_metrics([True], [True], [True, False], ["dog"])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected misaligned inputs to be rejected")
