from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from services.orchestrator.app.policy import estimate_generation_cost
from services.orchestrator.app.providers import generate_text

from .evaluator import evaluate, load_questions
from .schemas import Prediction


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_direct_prompt(question: str) -> str:
    return (
        "You are answering a STEM literature-style question.\n"
        "Return strict JSON only with this schema:\n"
        '{"answer_text":"<final answer>","answer_confidence":<float 0..1>}.\n'
        "No markdown, no code fences, no extra keys.\n\n"
        f"Question: {question}"
    )


def _parse_direct_response(text: str) -> tuple[str, float]:
    if not text.strip():
        return "", 0.0

    candidate = text.strip()
    match = _JSON_OBJECT_RE.search(candidate)
    if match:
        candidate = match.group(0)

    try:
        obj = json.loads(candidate)
        answer = str(obj.get("answer_text", "")).strip()
        conf = float(obj.get("answer_confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        if answer:
            return answer, conf
    except Exception:  # noqa: BLE001
        pass

    # Fallback: treat raw output as answer text with neutral confidence.
    return text.strip(), 0.5


def _summarize_usage(usage_rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    count = max(1, len(usage_rows))
    total_cost = sum(float(row.get("cost_usd", 0.0)) for row in usage_rows)
    total_latency = sum(float(row.get("latency_s", 0.0)) for row in usage_rows)
    total_input = sum(int(row.get("tokens_input", 0)) for row in usage_rows)
    total_output = sum(int(row.get("tokens_output", 0)) for row in usage_rows)
    return {
        "num_runs": len(usage_rows),
        "total_cost_usd": total_cost,
        "avg_cost_usd": total_cost / count,
        "total_latency_s": total_latency,
        "avg_latency_s": total_latency / count,
        "total_tokens_input": total_input,
        "avg_tokens_input": total_input / count,
        "total_tokens_output": total_output,
        "avg_tokens_output": total_output / count,
        "cost_estimated": True,
        "token_usage_partial": True,
        "token_usage_notes": (
            "Direct-call baseline uses provider token usage where available. "
            "Cost estimates are exact only for priced models configured in policy pricing map."
        ),
    }


def run_direct_single_call_benchmark_with_predictions(
    questions_path: str,
    provider: str,
    max_questions: int | None = None,
    max_tokens: int = 700,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    if max_questions is not None:
        questions = questions[: max(0, max_questions)]

    predictions: list[Prediction] = []
    usage_rows: list[dict[str, float | int]] = []
    failures: list[dict[str, str]] = []

    for question in questions:
        prompt = _build_direct_prompt(question.question)
        result = generate_text(provider, prompt, max_tokens=max_tokens)

        if result.success:
            answer_text, answer_conf = _parse_direct_response(result.text)
            if not answer_text.strip():
                failures.append(
                    {
                        "id": question.id,
                        "provider": provider,
                        "error": "empty_answer_text",
                    }
                )
                answer_conf = 0.0
        else:
            answer_text, answer_conf = "", 0.0
            failures.append(
                {
                    "id": question.id,
                    "provider": provider,
                    "error": result.error or "unknown_error",
                }
            )

        predictions.append(
            Prediction(
                id=question.id,
                answer_text=answer_text,
                answer_confidence=answer_conf,
                citation_count=0,
            )
        )

        usage_rows.append(
            {
                "cost_usd": float(
                    estimate_generation_cost(
                        provider=provider,
                        model=result.model,
                        tokens_input=result.tokens_input,
                        tokens_input_cached=result.tokens_input_cached,
                        tokens_output=result.tokens_output,
                    )
                ),
                "latency_s": 0.0,
                "tokens_input": int(result.tokens_input),
                "tokens_output": int(result.tokens_output),
            }
        )

    report = evaluate(questions, predictions).model_dump(mode="json")
    usage_summary = _summarize_usage(usage_rows)
    report["usage_summary"] = usage_summary
    report["direct_call"] = {
        "provider": provider,
        "failures": failures,
        "failure_count": len(failures),
    }

    return {
        "report": report,
        "predictions": [prediction.model_dump(mode="json") for prediction in predictions],
        "usage_summary": usage_summary,
        "failures": failures,
    }


def write_direct_single_call_report(
    *,
    questions_path: str,
    provider: str,
    output_dir: str | Path,
    max_questions: int | None = None,
    max_tokens: int = 700,
) -> dict[str, Any]:
    result = run_direct_single_call_benchmark_with_predictions(
        questions_path=questions_path,
        provider=provider,
        max_questions=max_questions,
        max_tokens=max_tokens,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"predictions_direct_{provider}.json").write_text(json.dumps(result["predictions"], indent=2))
    (destination / f"report_direct_{provider}.json").write_text(json.dumps(result["report"], indent=2))
    return result
