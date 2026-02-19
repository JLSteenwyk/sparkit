from __future__ import annotations

from services.orchestrator.app.calibration import CalibrationFeatures, calibrate_answer


def test_calibration_bounds_and_adjusts_claims() -> None:
    features = CalibrationFeatures(
        support_coverage=0.8,
        unsupported_claims=1,
        contradiction_flags=1,
        provider_config_ratio=1.0,
        ensemble_agreement=0.6,
        evidence_count=10,
    )
    answer_conf, claim_conf = calibrate_answer(features, {"clm_1": 0.9, "clm_2": 0.4})
    assert 0.05 <= answer_conf <= 0.95
    assert 0.05 <= claim_conf["clm_1"] <= 0.95
    assert 0.05 <= claim_conf["clm_2"] <= 0.95
