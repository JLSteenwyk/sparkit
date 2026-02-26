from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Any

from fastapi import HTTPException

from services.api_gateway.app.main import ask as gateway_ask
from services.api_gateway.app.main import get_run as gateway_get_run
from services.api_gateway.app.main import get_trace as gateway_get_trace
from services.orchestrator.app.policy import has_exact_pricing
from shared.schemas.api import AskRequest
from shared.schemas.domain import Constraints, Mode

from .evaluator import evaluate, load_questions
from .schemas import Prediction
from .usage_summary import summarize_usage


def _extract_answer_letter(text: str) -> str | None:
    match = re.search(r"<answer>\s*([A-N])\s*</answer>", text or "", flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _extract_mcq_provenance(trace_payload: dict[str, Any], answer_text: str) -> dict[str, Any] | None:
    stages = trace_payload.get("stages") or []
    if not isinstance(stages, list):
        return None
    selected = _extract_answer_letter(answer_text or "")

    scorer = None
    judge = None
    fallback = None
    rescue = None
    gate_final = None
    parse_failures: list[dict[str, str]] = []
    hard_block = None
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        name = stage.get("name")
        artifacts = stage.get("artifacts") or {}
        if not isinstance(artifacts, dict):
            artifacts = {}
        if name == "mcq_option_scorer":
            scorer = artifacts
        elif name == "mcq_option_judge":
            judge = artifacts
        elif name == "mcq_format_fallback":
            fallback = artifacts
        elif name == "mcq_option_rescue":
            rescue = artifacts
        elif name == "mcq_parse_failure":
            parse_failures.append(
                {
                    "source": str(artifacts.get("source", "")),
                    "reason": str(artifacts.get("reason", "")),
                }
            )
        elif name == "mcq_evidence_gate" and artifacts.get("phase") == "finalization":
            gate_final = artifacts
        elif name == "mcq_hard_block":
            hard_block = artifacts

    is_mcq = any(
        isinstance(stage, dict) and str(stage.get("name", "")).startswith("mcq_")
        for stage in stages
    )
    if not is_mcq:
        return None

    provenance_source = "unknown"
    if hard_block:
        provenance_source = "hard_block"
    elif rescue and rescue.get("rescue_applied"):
        provenance_source = "rescue_override"
    elif fallback:
        provenance_source = "format_fallback"
    elif judge:
        provenance_source = "judge"
    elif scorer:
        provenance_source = "scorer"

    return {
        "selected_option": selected,
        "selected_via": provenance_source,
        "eligible_labels": (scorer or {}).get("eligible_labels"),
        "allowed_labels": (scorer or {}).get("allowed_labels"),
        "evidence_gate_passed": bool((gate_final or {}).get("passed", False)),
        "evidence_gate_reason": (gate_final or {}).get("reason"),
        "rescue_applied": bool((rescue or {}).get("rescue_applied", False)),
        "fallback_used": bool(fallback),
        "hard_block_applied": bool(hard_block),
        "parse_failures": parse_failures,
    }


def _summarize_usage(usage_rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    return summarize_usage(
        usage_rows,
        token_usage_partial=False,
        token_usage_notes=(
            "Synthesis provider tokens are tracked from API usage when available; "
            "cost is exact when all generation models in a run have configured pricing."
        ),
    )


def run_benchmark(
    questions_path: str,
    mode: str = "single",
    providers: list[str] | None = None,
    max_questions: int | None = None,
    min_sources: int | None = None,
    max_latency_s: int | None = None,
    max_cost_usd: float = 3.0,
    parallel_workers: int = 1,
) -> dict:
    result = run_benchmark_with_predictions(
        questions_path=questions_path,
        mode=mode,
        providers=providers,
        max_questions=max_questions,
        min_sources=min_sources,
        max_latency_s=max_latency_s,
        max_cost_usd=max_cost_usd,
        parallel_workers=parallel_workers,
    )
    return result["report"]


def _run_question(
    idx: int,
    question: Any,
    mode: str,
    providers: list[str],
    min_sources: int | None,
    max_latency_s: int | None,
    max_cost_usd: float,
) -> dict[str, Any]:
    constraints = {
        "min_sources": min_sources if min_sources is not None else question.must_have_citations,
        "max_cost_usd": max_cost_usd,
    }
    if max_latency_s is not None:
        constraints["max_latency_s"] = max_latency_s
    ask_payload = AskRequest(
        question=question.question,
        mode=Mode(mode),
        providers=providers,
        constraints=Constraints.model_validate(constraints),
    )
    try:
        ask_response = gateway_ask(ask_payload)
    except HTTPException as exc:
        return {
            "idx": idx,
            "question_id": question.id,
            "run_id": "n/a",
            "run_status": "failed",
            "failure_reason": f"ask_http_{exc.status_code}",
            "usage": {
                "cost_usd": 0.0,
                "latency_s": 0.0,
                "tokens_input": 0,
                "tokens_output": 0,
                "cost_exact": True,
            },
            "quality": {
                "citation_coverage": 0.0,
                "unsupported_claims": 1,
                "contradiction_flags": 0,
            },
            "prediction": Prediction(
                id=question.id,
                answer_text="",
                answer_confidence=0.0,
                citation_count=0,
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "idx": idx,
            "question_id": question.id,
            "run_id": "n/a",
            "run_status": "failed",
            "failure_reason": f"ask_error_{type(exc).__name__}",
            "usage": {
                "cost_usd": 0.0,
                "latency_s": 0.0,
                "tokens_input": 0,
                "tokens_output": 0,
                "cost_exact": True,
            },
            "quality": {
                "citation_coverage": 0.0,
                "unsupported_claims": 1,
                "contradiction_flags": 0,
            },
            "prediction": Prediction(
                id=question.id,
                answer_text="",
                answer_confidence=0.0,
                citation_count=0,
            ),
        }
    run_id = ask_response.run_id
    run_response = gateway_get_run(run_id).model_dump(mode="json")
    run_status = str(run_response.get("status", ""))
    usage = run_response.get("usage") or {}
    trace_payload = gateway_get_trace(run_id).model_dump(mode="json") or {}
    quality = trace_payload.get("quality_gates", {})
    provider_usage = trace_payload.get("provider_usage") or []
    cost_exact = True
    for usage_item in provider_usage:
        provider = str(usage_item.get("provider", ""))
        model = str(usage_item.get("model", ""))
        cost_usd = float(usage_item.get("cost_usd", 0.0))
        if cost_usd <= 0:
            continue
        if not has_exact_pricing(provider=provider, model=model):
            cost_exact = False
            break
    answer = run_response.get("answer") or {}
    answer_text = str(answer.get("final_text", "") or "")
    citations = run_response.get("citations") or []
    prediction = Prediction(
        id=question.id,
        answer_text=answer_text,
        answer_confidence=answer.get("answer_confidence", 0.0),
        citation_count=len(citations),
        mcq_decision=_extract_mcq_provenance(trace_payload, answer_text),
    )
    failure_reason: str | None = None
    if run_status != "completed":
        failure_reason = f"run_status_{run_status or 'unknown'}"
    elif not answer_text.strip():
        failure_reason = "empty_answer_text"
    return {
        "idx": idx,
        "question_id": question.id,
        "run_id": run_id,
        "run_status": run_status,
        "failure_reason": failure_reason,
        "usage": {
            "cost_usd": float(usage.get("cost_usd", 0.0)),
            "latency_s": float(usage.get("latency_s", 0.0)),
            "tokens_input": int(usage.get("tokens_input", 0)),
            "tokens_output": int(usage.get("tokens_output", 0)),
            "cost_exact": cost_exact,
        },
        "quality": {
            "citation_coverage": float(quality.get("citation_coverage", 0.0)),
            "unsupported_claims": int(quality.get("unsupported_claims", 0)),
            "contradiction_flags": int(quality.get("contradiction_flags", 0)),
        },
        "prediction": prediction,
    }


def run_benchmark_with_predictions(
    questions_path: str,
    mode: str = "single",
    providers: list[str] | None = None,
    max_questions: int | None = None,
    min_sources: int | None = None,
    max_latency_s: int | None = None,
    max_cost_usd: float = 3.0,
    parallel_workers: int = 1,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    if max_questions is not None:
        questions = questions[: max(0, max_questions)]

    predictions: list[Prediction] = []
    run_ids: list[str] = []
    failures: list[dict[str, str]] = []
    quality_gates: list[dict[str, float | int]] = []
    usage_rows: list[dict[str, float | int]] = []
    provider_list = providers or ["openai"]
    workers = max(1, parallel_workers)
    if workers == 1:
        rows = [
            _run_question(
                idx=idx,
                question=question,
                mode=mode,
                providers=provider_list,
                min_sources=min_sources,
                max_latency_s=max_latency_s,
                max_cost_usd=max_cost_usd,
            )
            for idx, question in enumerate(questions)
        ]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_question,
                    idx,
                    question,
                    mode,
                    provider_list,
                    min_sources,
                    max_latency_s,
                    max_cost_usd,
                )
                for idx, question in enumerate(questions)
            ]
            for future in as_completed(futures):
                rows.append(future.result())

    for row in sorted(rows, key=lambda item: int(item["idx"])):
        run_ids.append(str(row["run_id"]))
        if row.get("failure_reason"):
            failures.append(
                {
                    "id": str(row.get("question_id", "")),
                    "run_id": str(row.get("run_id", "")),
                    "status": str(row.get("run_status", "")),
                    "error": str(row.get("failure_reason", "")),
                }
            )
        usage_rows.append(dict(row["usage"]))
        quality_gates.append(dict(row["quality"]))
        predictions.append(row["prediction"])

    report = evaluate(questions, predictions)
    gate_count = max(1, len(quality_gates))
    quality_summary = {
        "avg_citation_coverage": sum(float(item["citation_coverage"]) for item in quality_gates) / gate_count,
        "avg_unsupported_claims": sum(int(item["unsupported_claims"]) for item in quality_gates) / gate_count,
        "avg_contradiction_flags": sum(int(item["contradiction_flags"]) for item in quality_gates) / gate_count,
        "max_unsupported_claims": max((int(item["unsupported_claims"]) for item in quality_gates), default=0),
        "max_contradiction_flags": max((int(item["contradiction_flags"]) for item in quality_gates), default=0),
    }
    usage_summary = _summarize_usage(usage_rows)
    report_dict = report.model_dump(mode="json")
    report_dict["quality_summary"] = quality_summary
    report_dict["usage_summary"] = usage_summary
    report_dict["orchestrated_run"] = {
        "failures": failures,
        "failure_count": len(failures),
    }
    return {
        "report": report_dict,
        "predictions": [prediction.model_dump(mode="json") for prediction in predictions],
        "run_ids": run_ids,
        "failures": failures,
        "quality_summary": quality_summary,
        "usage_summary": usage_summary,
    }
