from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.eval_service.app.evaluator import evaluate, load_questions
from services.eval_service.app.runner import _summarize_usage
from services.eval_service.app.schemas import Prediction
from services.orchestrator.app.policy import estimate_generation_cost, has_exact_pricing
from services.orchestrator.app.providers import generate_text


def _ts_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_answer_prompt(question_text: str, answer_type: str) -> str:
    if (answer_type or "").lower() == "multiplechoice":
        return (
            "You are answering a STEM multiple-choice question.\n"
            "Return ONLY a single XML tag containing the chosen option letter.\n"
            "Format exactly: <answer>X</answer> where X is one letter.\n"
            "No extra text.\n\n"
            f"Question: {question_text}"
        )
    return (
        "You are answering a STEM literature-style question.\n"
        "Provide a high-level academic answer only; do not provide actionable lab protocols.\n"
        "Return strict JSON only with this schema:\n"
        '{"answer_text":"<final answer>","answer_confidence":<float 0..1>}.\n'
        "No markdown, no code fences, no extra keys.\n\n"
        f"Question: {question_text}"
    )


def _build_rephrase_prompt(question_text: str, answer_type: str) -> str:
    return (
        "Rewrite the question into a cleaner, higher-signal STEM prompt for an expert model.\n"
        "Keep meaning unchanged. Do not answer. Preserve all constraints and answer choices exactly.\n"
        "For multiple-choice: keep the same option letters and texts.\n"
        "Return only the rewritten question text.\n\n"
        f"Answer type: {answer_type}\n\n"
        f"Original question:\n{question_text}"
    )


def _parse_answer(text: str) -> tuple[str, float]:
    raw = (text or "").strip()
    if not raw:
        return "", 0.0
    try:
        payload = json.loads(raw)
        answer = str(payload.get("answer_text", "")).strip()
        conf = float(payload.get("answer_confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        if answer:
            return answer, conf
    except Exception:
        pass
    return raw, 0.5


def _usage_row(result: Any) -> dict[str, float | int | bool]:
    return {
        "cost_exact": bool(has_exact_pricing(provider=result.provider, model=result.model)),
        "cost_usd": float(
            estimate_generation_cost(
                provider=result.provider,
                model=result.model,
                tokens_input=result.tokens_input,
                tokens_input_cached=result.tokens_input_cached,
                tokens_output=result.tokens_output,
            )
        ),
        "latency_s": float(result.latency_s),
        "tokens_input": int(result.tokens_input),
        "tokens_output": int(result.tokens_output),
    }


def run_experiment(questions_path: str, max_questions: int, output_dir: str, label: str) -> dict[str, Any]:
    questions = load_questions(questions_path)[:max(0, max_questions)]
    destination = Path(output_dir) / f"{label}_{_ts_slug()}"
    destination.mkdir(parents=True, exist_ok=True)

    baseline_predictions: list[Prediction] = []
    rephrase_predictions: list[Prediction] = []
    baseline_usage_rows: list[dict[str, float | int | bool]] = []
    rephrase_usage_rows: list[dict[str, float | int | bool]] = []
    rephrase_records: list[dict[str, str]] = []

    for question in questions:
        baseline_prompt = _build_answer_prompt(question.question, question.answer_type)
        baseline_result = generate_text("openai", baseline_prompt, max_tokens=700)
        baseline_answer, baseline_conf = _parse_answer(baseline_result.text)
        baseline_predictions.append(
            Prediction(
                id=question.id,
                answer_text=baseline_answer,
                answer_confidence=baseline_conf,
                citation_count=0,
            )
        )
        baseline_usage_rows.append(_usage_row(baseline_result))

        rephrase_prompt = _build_rephrase_prompt(question.question, question.answer_type)
        rephrase_result = generate_text("openai", rephrase_prompt, max_tokens=900)
        rewritten_question = (rephrase_result.text or "").strip() or question.question
        rephrase_usage_rows.append(_usage_row(rephrase_result))

        rewritten_answer_prompt = _build_answer_prompt(rewritten_question, question.answer_type)
        rewritten_answer_result = generate_text("openai", rewritten_answer_prompt, max_tokens=700)
        rewritten_answer, rewritten_conf = _parse_answer(rewritten_answer_result.text)
        rephrase_predictions.append(
            Prediction(
                id=question.id,
                answer_text=rewritten_answer,
                answer_confidence=rewritten_conf,
                citation_count=0,
            )
        )
        rephrase_usage_rows.append(_usage_row(rewritten_answer_result))

        rephrase_records.append(
            {
                "id": question.id,
                "original_question": question.question,
                "rewritten_question": rewritten_question,
            }
        )

    baseline_report = evaluate(questions, baseline_predictions).model_dump(mode="json")
    rephrase_report = evaluate(questions, rephrase_predictions).model_dump(mode="json")
    baseline_report["usage_summary"] = _summarize_usage(baseline_usage_rows)
    rephrase_report["usage_summary"] = _summarize_usage(rephrase_usage_rows)

    baseline_pred_json = [item.model_dump(mode="json") for item in baseline_predictions]
    rephrase_pred_json = [item.model_dump(mode="json") for item in rephrase_predictions]

    (destination / "predictions_baseline_openai.json").write_text(json.dumps(baseline_pred_json, indent=2))
    (destination / "report_baseline_openai.json").write_text(json.dumps(baseline_report, indent=2))
    (destination / "predictions_rephrase_then_answer_openai.json").write_text(json.dumps(rephrase_pred_json, indent=2))
    (destination / "report_rephrase_then_answer_openai.json").write_text(json.dumps(rephrase_report, indent=2))
    (destination / "rephrased_questions.json").write_text(json.dumps(rephrase_records, indent=2))

    summary = {
        "label": label,
        "run_slug": destination.name,
        "questions_path": questions_path,
        "max_questions": max_questions,
        "baseline": {
            "average_rubric_score": baseline_report.get("average_rubric_score"),
            "brier_score": (baseline_report.get("calibration") or {}).get("brier_score"),
            "ece": (baseline_report.get("calibration") or {}).get("ece"),
            "usage_summary": baseline_report.get("usage_summary"),
        },
        "rephrase_then_answer": {
            "average_rubric_score": rephrase_report.get("average_rubric_score"),
            "brier_score": (rephrase_report.get("calibration") or {}).get("brier_score"),
            "ece": (rephrase_report.get("calibration") or {}).get("ece"),
            "usage_summary": rephrase_report.get("usage_summary"),
        },
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B: direct OpenAI vs rephrase-then-answer on HLE questions.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--label", default="hle_rephrase_ab_openai")
    parser.add_argument("--max-questions", type=int, default=10)
    args = parser.parse_args()

    summary = run_experiment(
        questions_path=args.questions,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        label=args.label,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
