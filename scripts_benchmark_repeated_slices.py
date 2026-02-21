from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from pathlib import Path
from statistics import mean, stdev

from services.eval_service.app.baseline_capture import parse_configs
from services.eval_service.app.runner import run_benchmark_with_predictions


def _ci95(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_half_width": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "ci95_half_width": 0.0}
    m = float(mean(values))
    s = float(stdev(values))
    hw = 1.96 * s / math.sqrt(len(values))
    return {"mean": m, "std": s, "ci95_half_width": float(hw)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeated-slice benchmark with confidence intervals for SPARKIT configs."
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument("--configs", default="single_openai,single_anthropic,routed_frontier")
    parser.add_argument("--slice-size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--output", default="benchmarks/results/repeated_slices_report.json")
    args = parser.parse_args()

    raw_questions = json.loads(Path(args.questions).read_text())
    if args.slice_size <= 0:
        raise ValueError("slice-size must be > 0")
    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")
    if len(raw_questions) < args.slice_size:
        raise ValueError(f"Need at least {args.slice_size} questions, found {len(raw_questions)}")

    rng = random.Random(args.seed)
    configs = parse_configs(args.configs)

    metrics: dict[str, dict[str, list[float]]] = {
        cfg.name: {
            "average_rubric_score": [],
            "brier_score": [],
            "ece": [],
            "total_cost_usd": [],
            "total_latency_s": [],
        }
        for cfg in configs
    }

    for repeat_idx in range(args.repeats):
        sample = rng.sample(raw_questions, args.slice_size)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as handle:
            json.dump(sample, handle)
            handle.flush()

            for cfg in configs:
                result = run_benchmark_with_predictions(
                    questions_path=handle.name,
                    mode=cfg.mode,
                    providers=cfg.providers,
                    max_questions=None,
                    min_sources=None,
                    max_latency_s=None,
                    max_cost_usd=3.0,
                    parallel_workers=max(1, args.parallel_workers),
                )
                report = result["report"]
                usage = result.get("usage_summary", {})
                metrics[cfg.name]["average_rubric_score"].append(float(report.get("average_rubric_score", 0.0)))
                metrics[cfg.name]["brier_score"].append(float(report.get("calibration", {}).get("brier_score", 0.0)))
                metrics[cfg.name]["ece"].append(float(report.get("calibration", {}).get("ece", 0.0)))
                metrics[cfg.name]["total_cost_usd"].append(float(usage.get("total_cost_usd", 0.0)))
                metrics[cfg.name]["total_latency_s"].append(float(usage.get("total_latency_s", 0.0)))

        print(f"Completed repeat {repeat_idx + 1}/{args.repeats}")

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for cfg in configs:
        row = metrics[cfg.name]
        summary[cfg.name] = {metric: _ci95(values) for metric, values in row.items()}

    output = {
        "questions_path": args.questions,
        "slice_size": args.slice_size,
        "repeats": args.repeats,
        "seed": args.seed,
        "parallel_workers": max(1, args.parallel_workers),
        "configs": [cfg.name for cfg in configs],
        "summary": summary,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
