from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from services.eval_service.app.drift import (
    check_manifest_drift,
    check_metrics,
    load_thresholds,
    report_to_metrics,
)


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check benchmark drift against thresholds and optional baseline.")
    parser.add_argument("--thresholds", default="benchmarks/drift_thresholds.json")

    parser.add_argument("--candidate-manifest", default="")
    parser.add_argument("--baseline-manifest", default="")

    parser.add_argument("--report", default="")
    parser.add_argument("--quality-summary", default="")
    parser.add_argument("--baseline-report", default="")
    parser.add_argument("--baseline-quality-summary", default="")

    parser.add_argument("--output", default="")
    args = parser.parse_args()

    thresholds = load_thresholds(args.thresholds)

    if args.candidate_manifest:
        candidate_manifest = _load_json(args.candidate_manifest)
        baseline_manifest = _load_json(args.baseline_manifest) if args.baseline_manifest else None
        result = check_manifest_drift(candidate_manifest, thresholds, baseline_manifest=baseline_manifest)
    elif args.report:
        report = _load_json(args.report)
        quality = _load_json(args.quality_summary) if args.quality_summary else None
        baseline_metrics = None
        if args.baseline_report:
            baseline_report = _load_json(args.baseline_report)
            baseline_quality = _load_json(args.baseline_quality_summary) if args.baseline_quality_summary else None
            baseline_metrics = report_to_metrics(baseline_report, baseline_quality)

        metrics = report_to_metrics(report, quality)
        violations = check_metrics(metrics, thresholds, baseline=baseline_metrics)
        result = {
            "passed": len(violations) == 0,
            "num_violations": len(violations),
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "violations": violations,
        }
    else:
        raise SystemExit("Provide either --candidate-manifest or --report")

    rendered = json.dumps(result, indent=2)
    print(rendered)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")

    if not result.get("passed", False):
        sys.exit(2)


if __name__ == "__main__":
    main()
