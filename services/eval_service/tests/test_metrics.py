from __future__ import annotations

from services.eval_service.app.metrics import brier_score, expected_calibration_error


def test_brier_score() -> None:
    confidences = [0.9, 0.2, 0.7]
    outcomes = [1, 0, 1]
    score = brier_score(confidences, outcomes)
    assert 0.0 <= score <= 1.0


def test_expected_calibration_error() -> None:
    confidences = [0.1, 0.2, 0.8, 0.9]
    outcomes = [0, 0, 1, 1]
    ece = expected_calibration_error(confidences, outcomes, bins=5)
    assert 0.0 <= ece <= 1.0
