from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_thresholds(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _absolute_violations(metrics: dict[str, float], absolute: dict[str, float]) -> list[str]:
    violations: list[str] = []
    min_map = {
        "average_rubric_score": "min_average_rubric_score",
        "avg_citation_coverage": "min_avg_citation_coverage",
    }
    max_map = {
        "brier_score": "max_brier_score",
        "ece": "max_ece",
        "avg_unsupported_claims": "max_avg_unsupported_claims",
        "avg_contradiction_flags": "max_avg_contradiction_flags",
    }

    for metric, key in min_map.items():
        if key in absolute and metrics.get(metric, 0.0) < float(absolute[key]):
            violations.append(f"{metric}={metrics.get(metric, 0.0):.4f} below {key}={float(absolute[key]):.4f}")
    for metric, key in max_map.items():
        if key in absolute and metrics.get(metric, 0.0) > float(absolute[key]):
            violations.append(f"{metric}={metrics.get(metric, 0.0):.4f} above {key}={float(absolute[key]):.4f}")
    return violations


def _regression_violations(
    candidate: dict[str, float], baseline: dict[str, float], regression: dict[str, float]
) -> list[str]:
    violations: list[str] = []
    drop_map = {
        "average_rubric_score": "max_rubric_drop",
        "avg_citation_coverage": "max_citation_coverage_drop",
    }
    increase_map = {
        "brier_score": "max_brier_increase",
        "ece": "max_ece_increase",
        "avg_unsupported_claims": "max_avg_unsupported_increase",
        "avg_contradiction_flags": "max_avg_contradiction_increase",
    }

    for metric, key in drop_map.items():
        if key in regression:
            delta = baseline.get(metric, 0.0) - candidate.get(metric, 0.0)
            if delta > float(regression[key]):
                violations.append(f"{metric} dropped by {delta:.4f}, allowed {float(regression[key]):.4f}")
    for metric, key in increase_map.items():
        if key in regression:
            delta = candidate.get(metric, 0.0) - baseline.get(metric, 0.0)
            if delta > float(regression[key]):
                violations.append(f"{metric} increased by {delta:.4f}, allowed {float(regression[key]):.4f}")
    return violations


def check_metrics(
    candidate: dict[str, float],
    thresholds: dict[str, Any],
    baseline: dict[str, float] | None = None,
) -> list[str]:
    violations: list[str] = []
    violations.extend(_absolute_violations(candidate, thresholds.get("absolute", {})))
    if baseline is not None:
        violations.extend(_regression_violations(candidate, baseline, thresholds.get("regression", {})))
    return violations


def manifest_to_metric_map(manifest: dict[str, Any]) -> dict[str, dict[str, float]]:
    metric_map: dict[str, dict[str, float]] = {}
    for config in manifest.get("configs", []):
        if config.get("status") != "completed":
            continue
        name = str(config.get("name"))
        metric_map[name] = {
            "average_rubric_score": float(config.get("average_rubric_score", 0.0)),
            "brier_score": float(config.get("brier_score", 0.0)),
            "ece": float(config.get("ece", 0.0)),
            "avg_citation_coverage": float(config.get("avg_citation_coverage", 0.0)),
            "avg_unsupported_claims": float(config.get("avg_unsupported_claims", 0.0)),
            "avg_contradiction_flags": float(config.get("avg_contradiction_flags", 0.0)),
        }
    return metric_map


def check_manifest_drift(
    candidate_manifest: dict[str, Any],
    thresholds: dict[str, Any],
    baseline_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_map = manifest_to_metric_map(candidate_manifest)
    baseline_map = manifest_to_metric_map(baseline_manifest or {})
    by_config: dict[str, Any] = {}
    total_violations = 0

    for name, candidate_metrics in candidate_map.items():
        baseline_metrics = baseline_map.get(name)
        violations = check_metrics(candidate=candidate_metrics, thresholds=thresholds, baseline=baseline_metrics)
        by_config[name] = {
            "violations": violations,
            "candidate_metrics": candidate_metrics,
            "baseline_metrics": baseline_metrics,
        }
        total_violations += len(violations)

    return {
        "passed": total_violations == 0,
        "num_configs_checked": len(candidate_map),
        "num_violations": total_violations,
        "by_config": by_config,
    }


def report_to_metrics(report: dict[str, Any], quality_summary: dict[str, Any] | None = None) -> dict[str, float]:
    quality = quality_summary or {}
    calibration = report.get("calibration", {})
    return {
        "average_rubric_score": float(report.get("average_rubric_score", 0.0)),
        "brier_score": float(calibration.get("brier_score", 0.0)),
        "ece": float(calibration.get("ece", 0.0)),
        "avg_citation_coverage": float(quality.get("avg_citation_coverage", 1.0)),
        "avg_unsupported_claims": float(quality.get("avg_unsupported_claims", 0.0)),
        "avg_contradiction_flags": float(quality.get("avg_contradiction_flags", 0.0)),
    }
