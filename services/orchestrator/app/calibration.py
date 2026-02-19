from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CalibrationFeatures:
    support_coverage: float
    unsupported_claims: int
    contradiction_flags: int
    provider_config_ratio: float
    ensemble_agreement: float
    evidence_count: int


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def calibrate_answer(features: CalibrationFeatures, base_confidences: dict[str, float]) -> tuple[float, dict[str, float]]:
    raw = (
        0.25
        + 0.45 * features.support_coverage
        + 0.15 * features.provider_config_ratio
        + 0.10 * features.ensemble_agreement
        + 0.05 * min(1.0, features.evidence_count / 10.0)
        - 0.07 * features.unsupported_claims
        - 0.05 * features.contradiction_flags
    )
    answer_confidence = _clamp(raw, 0.05, 0.95)

    calibrated_claims: dict[str, float] = {}
    for claim_id, base in base_confidences.items():
        calibrated_claims[claim_id] = _clamp(0.5 * base + 0.5 * answer_confidence, 0.05, 0.95)

    return answer_confidence, calibrated_claims


def features_to_dict(features: CalibrationFeatures) -> dict[str, float | int]:
    return asdict(features)
