from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from services.eval_service.app.direct_call_runner import run_direct_single_call_benchmark_with_predictions


KEY_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "kimi": ["KIMI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "grok": ["GROK_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
}


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _provider_has_keys(provider: str) -> bool:
    return any(bool(os.getenv(var)) for var in KEY_VARS.get(provider, []))


def _key_status() -> dict[str, bool]:
    return {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "KIMI_API_KEY": bool(os.getenv("KIMI_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.getenv("DEEPSEEK_API_KEY")),
        "GROK_API_KEY": bool(os.getenv("GROK_API_KEY")),
        "MISTRAL_API_KEY": bool(os.getenv("MISTRAL_API_KEY")),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
        "EXA_API_KEY": bool(os.getenv("EXA_API_KEY")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture direct single-call model baselines.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--label", default="direct_calls")
    parser.add_argument("--providers", default="openai,anthropic,gemini,kimi,deepseek,grok,mistral")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--skip-missing-keys", action="store_true")
    args = parser.parse_args()

    providers = [item.strip().lower() for item in args.providers.split(",") if item.strip()]
    max_questions = args.max_questions if args.max_questions > 0 else None
    run_slug = f"{args.label}_{_timestamp_slug()}"
    destination = Path(args.output_dir) / run_slug
    destination.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "label": args.label,
        "run_slug": run_slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "questions_path": args.questions,
        "max_questions": max_questions,
        "max_tokens": args.max_tokens,
        "kind": "direct_single_call",
        "key_status": _key_status(),
        "providers": [],
    }

    for provider in providers:
        record: dict[str, object] = {
            "provider": provider,
            "required_key_vars": KEY_VARS.get(provider, []),
        }
        if args.skip_missing_keys and not _provider_has_keys(provider):
            record["status"] = "skipped_missing_keys"
            cast = manifest["providers"]
            assert isinstance(cast, list)
            cast.append(record)
            continue

        result = run_direct_single_call_benchmark_with_predictions(
            questions_path=args.questions,
            provider=provider,
            max_questions=max_questions,
            max_tokens=args.max_tokens,
        )
        (destination / f"predictions_direct_{provider}.json").write_text(json.dumps(result["predictions"], indent=2))
        (destination / f"report_direct_{provider}.json").write_text(json.dumps(result["report"], indent=2))
        record.update(
            {
                "status": "completed",
                "report_file": f"report_direct_{provider}.json",
                "predictions_file": f"predictions_direct_{provider}.json",
                "num_predictions": len(result["predictions"]),
                "average_rubric_score": result["report"].get("average_rubric_score", 0.0),
                "brier_score": result["report"].get("calibration", {}).get("brier_score", 0.0),
                "ece": result["report"].get("calibration", {}).get("ece", 0.0),
                "total_cost_usd": result.get("usage_summary", {}).get("total_cost_usd", 0.0),
                "avg_cost_usd": result.get("usage_summary", {}).get("avg_cost_usd", 0.0),
                "total_tokens_input": result.get("usage_summary", {}).get("total_tokens_input", 0),
                "total_tokens_output": result.get("usage_summary", {}).get("total_tokens_output", 0),
                "failure_count": len(result.get("failures", [])),
                "failed_question_ids": [item.get("id") for item in result.get("failures", []) if item.get("id")],
            }
        )
        cast = manifest["providers"]
        assert isinstance(cast, list)
        cast.append(record)

    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
