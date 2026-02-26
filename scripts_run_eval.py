from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.eval_service.app.evaluator import evaluate, load_predictions, load_questions
from services.eval_service.app.runner import run_benchmark_with_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="SPARKIT benchmark and calibration evaluator")
    parser.add_argument("--questions", default="benchmarks/stem_exam_200_sample/questions.json")
    parser.add_argument("--predictions", default="")
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "simple_rag", "routed", "ensemble", "option_graph_v2"],
    )
    parser.add_argument("--providers", default="openai")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--save-predictions", default="")
    parser.add_argument("--save-report", default="")
    parser.add_argument("--save-quality-summary", default="")
    args = parser.parse_args()

    if args.predictions:
        questions = load_questions(args.questions)
        if args.max_questions > 0:
            questions = questions[: args.max_questions]
        predictions = load_predictions(args.predictions)
        report = evaluate(questions, predictions).model_dump(mode="json")
    else:
        providers = [provider.strip() for provider in args.providers.split(",") if provider.strip()]
        max_questions = args.max_questions if args.max_questions > 0 else None
        result = run_benchmark_with_predictions(
            args.questions,
            mode=args.mode,
            providers=providers,
            max_questions=max_questions,
        )
        report = result["report"]
        if args.save_predictions:
            destination = Path(args.save_predictions)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result["predictions"], indent=2))
        if args.save_quality_summary:
            destination = Path(args.save_quality_summary)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result.get("quality_summary", {}), indent=2))

    if args.save_report:
        destination = Path(args.save_report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
