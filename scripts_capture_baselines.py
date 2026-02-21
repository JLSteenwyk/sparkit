from __future__ import annotations

import argparse
import json

from services.eval_service.app.baseline_capture import capture_baselines, parse_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture benchmark baselines for configured provider/mode runs.")
    parser.add_argument("--questions", default="benchmarks/stem_exam_200/questions.json")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--label", default="baseline")
    parser.add_argument(
        "--configs",
        default="",
        help="Comma-separated config names. Defaults to all presets.",
    )
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--min-sources", type=int, default=0)
    parser.add_argument("--max-latency-s", type=int, default=0)
    parser.add_argument("--max-cost-usd", type=float, default=3.0)
    parser.add_argument("--parallel-workers", type=int, default=1, help="Parallel question workers per config.")
    parser.add_argument("--parallel-configs", type=int, default=1, help="Parallel configs in baseline capture.")
    parser.add_argument(
        "--skip-missing-keys",
        action="store_true",
        help="Skip runs when provider credentials are unavailable.",
    )
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    max_questions = args.max_questions if args.max_questions > 0 else None
    min_sources = args.min_sources if args.min_sources > 0 else None
    max_latency_s = args.max_latency_s if args.max_latency_s > 0 else None

    manifest = capture_baselines(
        questions_path=args.questions,
        output_dir=args.output_dir,
        label=args.label,
        configs=configs,
        max_questions=max_questions,
        skip_missing_keys=args.skip_missing_keys,
        min_sources=min_sources,
        max_latency_s=max_latency_s,
        max_cost_usd=args.max_cost_usd,
        parallel_workers=max(1, args.parallel_workers),
        parallel_configs=max(1, args.parallel_configs),
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
