from __future__ import annotations

from .schemas import CalibrationMetrics


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    if not confidences:
        return 0.0
    return sum((confidence - outcome) ** 2 for confidence, outcome in zip(confidences, outcomes)) / len(
        confidences
    )


def expected_calibration_error(confidences: list[float], outcomes: list[int], bins: int = 10) -> float:
    if not confidences:
        return 0.0

    n = len(confidences)
    ece = 0.0
    for i in range(bins):
        lower = i / bins
        upper = (i + 1) / bins
        in_bin = [idx for idx, conf in enumerate(confidences) if lower <= conf < upper or (i == bins - 1 and conf == 1.0)]
        if not in_bin:
            continue

        bin_conf = sum(confidences[idx] for idx in in_bin) / len(in_bin)
        bin_acc = sum(outcomes[idx] for idx in in_bin) / len(in_bin)
        ece += abs(bin_acc - bin_conf) * (len(in_bin) / n)
    return ece


def calibration_metrics(confidences: list[float], outcomes: list[int]) -> CalibrationMetrics:
    return CalibrationMetrics(
        brier_score=brier_score(confidences, outcomes),
        ece=expected_calibration_error(confidences, outcomes),
    )
