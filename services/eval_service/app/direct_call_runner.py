from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from services.orchestrator.app.policy import estimate_generation_cost, has_exact_pricing
from services.orchestrator.app.providers import generate_text

from .evaluator import evaluate, load_questions
from .schemas import BenchmarkQuestion
from .schemas import Prediction


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_direct_prompt(question: BenchmarkQuestion) -> str:
    if (question.answer_type or "").lower() == "multiplechoice":
        return (
            "You are answering a STEM multiple-choice question.\n"
            "Return ONLY a single XML tag containing the chosen option letter.\n"
            "Format exactly: <answer>X</answer> where X is one letter.\n"
            "No extra text.\n\n"
            f"Question: {question.question}"
        )
    return (
        "You are answering a STEM literature-style question.\n"
        "Provide a high-level academic answer only; do not provide actionable lab protocols.\n"
        "Return strict JSON only with this schema:\n"
        '{"answer_text":"<final answer>","answer_confidence":<float 0..1>}.\n'
        "No markdown, no code fences, no extra keys.\n\n"
        f"Question: {question.question}"
    )


def _build_direct_fallback_prompt(question: str) -> str:
    return (
        "Answer the STEM question concisely in plain text (1-4 sentences), "
        "with a high-level academic explanation only (no actionable protocols). "
        "Do not return JSON.\n\n"
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


def _retry_attempts() -> int:
    raw = os.getenv("DIRECT_CALL_MAX_ATTEMPTS", "3")
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _retry_backoff_s() -> float:
    raw = os.getenv("DIRECT_CALL_RETRY_BACKOFF_S", "0.8")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.8


def _should_retry(provider: str, error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return True
    if "empty_content" in lowered:
        return True
    if provider.lower() == "deepseek" and "empty_content with reasoning_content" in lowered:
        return True
    if "too many requests" in lowered or "rate limit" in lowered:
        return True
    return False


def _direct_prompt_retry_variant(provider: str, prompt: str, attempt: int, error: str | None) -> str:
    if provider.lower() == "deepseek" and error and "reasoning_content" in error.lower() and attempt > 1:
        return (
            f"{prompt}\n\n"
            "IMPORTANT: Provide final answer content in message.content only. "
            "Do not leave content empty. Return strict JSON only."
        )
    return prompt


def _generate_with_retries(provider: str, prompt: str, max_tokens: int) -> Any:
    attempts = _retry_attempts()
    backoff = _retry_backoff_s()
    last = None
    for attempt in range(1, attempts + 1):
        retry_prompt = _direct_prompt_retry_variant(provider, prompt, attempt, getattr(last, "error", None))
        result = generate_text(provider, retry_prompt, max_tokens=max_tokens)
        last = result
        if result.success:
            return result
        if attempt >= attempts or not _should_retry(provider, result.error):
            return result
        sleep_s = backoff * (2 ** (attempt - 1))
        if sleep_s > 0:
            time.sleep(sleep_s)
    return last


def _recover_empty_answer(provider: str, prompt: str, max_tokens: int) -> Any:
    retry_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: You must provide a non-empty answer_text value. "
        "If uncertain, provide the best concise hypothesis. "
        "Return strict JSON only."
    )
    return _generate_with_retries(provider, retry_prompt, max_tokens=max_tokens)


def _summarize_usage(usage_rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    count = max(1, len(usage_rows))
    total_cost = sum(float(row.get("cost_usd", 0.0)) for row in usage_rows)
    total_latency = sum(float(row.get("latency_s", 0.0)) for row in usage_rows)
    total_input = sum(int(row.get("tokens_input", 0)) for row in usage_rows)
    total_output = sum(int(row.get("tokens_output", 0)) for row in usage_rows)
    exact_count = sum(1 for row in usage_rows if bool(row.get("cost_exact", False)))
    estimated_count = max(0, len(usage_rows) - exact_count)
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
        "cost_estimated": estimated_count > 0 or len(usage_rows) == 0,
        "cost_exact_runs": exact_count,
        "cost_estimated_runs": estimated_count,
        "token_usage_partial": True,
        "token_usage_notes": (
            "Direct-call baseline uses provider token usage where available. "
            "Cost estimates are exact only for models present in pricing config "
            "(defaults + SPARKIT_MODEL_PRICING_JSON)."
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
        prompt = _build_direct_prompt(question)
        result = _generate_with_retries(provider, prompt, max_tokens=max_tokens)
        failed = False
        failed_error = ""

        if result.success:
            answer_text, answer_conf = _parse_direct_response(result.text)
            if not answer_text.strip():
                recovered = _recover_empty_answer(provider, prompt, max_tokens=max_tokens)
                if recovered.success:
                    recovered_text, recovered_conf = _parse_direct_response(recovered.text)
                    if recovered_text.strip():
                        result = recovered
                        answer_text, answer_conf = recovered_text, recovered_conf
                    else:
                        failures.append(
                            {"id": question.id, "provider": provider, "error": "empty_answer_text"}
                        )
                        failed = True
                        failed_error = "empty_answer_text"
                        answer_conf = 0.0
                else:
                    failures.append(
                        {"id": question.id, "provider": provider, "error": "empty_answer_text"}
                    )
                    failed = True
                    failed_error = "empty_answer_text"
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
            failed = True
            failed_error = result.error or "unknown_error"

        # Last-chance salvage so isolated prompt-format failures do not drop the question.
        if failed and failed_error in {"empty_answer_text", "empty_content"}:
            fallback_prompt = _build_direct_fallback_prompt(question.question)
            fallback = _generate_with_retries(provider, fallback_prompt, max_tokens=max(1200, max_tokens))
            if fallback.success:
                fallback_text, fallback_conf = _parse_direct_response(fallback.text)
                if fallback_text.strip():
                    # Replace previous failure with recovered answer.
                    failures = [
                        item for item in failures if not (item.get("id") == question.id and item.get("provider") == provider)
                    ]
                    result = fallback
                    answer_text, answer_conf = fallback_text, min(0.5, max(0.2, fallback_conf))

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
                "cost_exact": bool(has_exact_pricing(provider=provider, model=result.model)),
                "cost_usd": float(
                    estimate_generation_cost(
                        provider=provider,
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
