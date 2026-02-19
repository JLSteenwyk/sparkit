from __future__ import annotations

from services.eval_service.app.drift import check_manifest_drift, check_metrics, report_to_metrics


def _thresholds() -> dict:
    return {
        "absolute": {
            "min_average_rubric_score": 0.55,
            "max_brier_score": 0.35,
            "max_ece": 0.30,
            "min_avg_citation_coverage": 0.50,
            "max_avg_unsupported_claims": 1.5,
            "max_avg_contradiction_flags": 2.0,
        },
        "regression": {
            "max_rubric_drop": 0.05,
            "max_brier_increase": 0.03,
            "max_ece_increase": 0.03,
            "max_citation_coverage_drop": 0.05,
            "max_avg_unsupported_increase": 0.3,
            "max_avg_contradiction_increase": 0.3,
        },
    }


def test_check_metrics_passes_for_healthy_values() -> None:
    metrics = {
        "average_rubric_score": 0.70,
        "brier_score": 0.20,
        "ece": 0.12,
        "avg_citation_coverage": 0.85,
        "avg_unsupported_claims": 0.2,
        "avg_contradiction_flags": 0.1,
    }
    assert check_metrics(metrics, _thresholds()) == []


def test_check_metrics_flags_absolute_and_regression_failures() -> None:
    candidate = {
        "average_rubric_score": 0.50,
        "brier_score": 0.40,
        "ece": 0.35,
        "avg_citation_coverage": 0.40,
        "avg_unsupported_claims": 2.0,
        "avg_contradiction_flags": 2.4,
    }
    baseline = {
        "average_rubric_score": 0.70,
        "brier_score": 0.20,
        "ece": 0.10,
        "avg_citation_coverage": 0.85,
        "avg_unsupported_claims": 0.1,
        "avg_contradiction_flags": 0.2,
    }
    violations = check_metrics(candidate, _thresholds(), baseline=baseline)
    assert len(violations) >= 6


def test_check_manifest_drift_uses_completed_configs_only() -> None:
    candidate = {
        "configs": [
            {
                "name": "single_openai",
                "status": "completed",
                "average_rubric_score": 0.68,
                "brier_score": 0.23,
                "ece": 0.11,
                "avg_citation_coverage": 0.80,
                "avg_unsupported_claims": 0.4,
                "avg_contradiction_flags": 0.3,
            },
            {"name": "single_kimi", "status": "skipped_missing_keys"},
        ]
    }
    baseline = {
        "configs": [
            {
                "name": "single_openai",
                "status": "completed",
                "average_rubric_score": 0.70,
                "brier_score": 0.22,
                "ece": 0.10,
                "avg_citation_coverage": 0.82,
                "avg_unsupported_claims": 0.3,
                "avg_contradiction_flags": 0.2,
            }
        ]
    }
    result = check_manifest_drift(candidate, _thresholds(), baseline_manifest=baseline)
    assert result["num_configs_checked"] == 1
    assert result["passed"] is True


def test_report_to_metrics_defaults_quality_fields() -> None:
    metrics = report_to_metrics(
        {
            "average_rubric_score": 0.66,
            "calibration": {"brier_score": 0.24, "ece": 0.09},
        }
    )
    assert metrics["avg_citation_coverage"] == 1.0
    assert metrics["avg_unsupported_claims"] == 0.0
    assert metrics["avg_contradiction_flags"] == 0.0
