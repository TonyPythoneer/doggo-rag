"""Paired First Pass / Second Pass classification metrics (pure Python)."""

from __future__ import annotations

from collections import defaultdict


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _pass_metrics(y_true: list[bool], predicted: list[bool | None], species: list[str]) -> dict:
    correct = [guess is truth for truth, guess in zip(y_true, predicted)]
    tp = sum(truth is True and guess is True for truth, guess in zip(y_true, predicted))
    fp = sum(truth is False and guess is True for truth, guess in zip(y_true, predicted))
    fn = sum(truth is True and guess is not True for truth, guess in zip(y_true, predicted))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    matrix = [
        [
            sum(truth is True and guess is True for truth, guess in zip(y_true, predicted)),
            sum(truth is True and guess is False for truth, guess in zip(y_true, predicted)),
            sum(truth is True and guess is None for truth, guess in zip(y_true, predicted)),
        ],
        [
            sum(truth is False and guess is True for truth, guess in zip(y_true, predicted)),
            sum(truth is False and guess is False for truth, guess in zip(y_true, predicted)),
            sum(truth is False and guess is None for truth, guess in zip(y_true, predicted)),
        ],
    ]
    by_species: dict[str, list[bool]] = defaultdict(list)
    for name, is_correct in zip(species, correct):
        by_species[name].append(is_correct)
    return {
        "count": len(y_true),
        "accuracy": round(_safe_div(sum(correct), len(correct)), 4),
        "abstentions": sum(value is None for value in predicted),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "true_labels": [True, False],
            "predicted_labels": [True, False, None],
            "matrix": matrix,
        },
        "per_species_accuracy": {
            name: round(_safe_div(sum(values), len(values)), 4)
            for name, values in sorted(by_species.items())
        },
    }


def paired_classification_metrics(
    y_true: list[bool],
    first: list[bool | None],
    second: list[bool | None],
    species: list[str],
) -> dict:
    """Score both passes on the same successful image cohort."""
    if not (len(y_true) == len(first) == len(second) == len(species)):
        raise ValueError("all paired metric inputs must have the same length")
    if not y_true:
        raise ValueError("No paired predictions to score")
    first_correct = [guess is truth for truth, guess in zip(y_true, first)]
    second_correct = [guess is truth for truth, guess in zip(y_true, second)]
    first_accuracy = _safe_div(sum(first_correct), len(first_correct))
    second_accuracy = _safe_div(sum(second_correct), len(second_correct))
    transitions = {
        "incorrect_to_correct": sum(not a and b for a, b in zip(first_correct, second_correct)),
        "correct_to_incorrect": sum(a and not b for a, b in zip(first_correct, second_correct)),
        "unchanged_correct": sum(a and b for a, b in zip(first_correct, second_correct)),
        "unchanged_incorrect": sum(not a and not b for a, b in zip(first_correct, second_correct)),
    }
    return {
        "paired_images": len(y_true),
        "first_pass": _pass_metrics(y_true, first, species),
        "second_pass": _pass_metrics(y_true, second, species),
        "accuracy_improvement_percentage_points": round(
            (second_accuracy - first_accuracy) * 100, 4
        ),
        "transitions": transitions,
    }
