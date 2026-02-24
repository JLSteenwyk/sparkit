from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from fastapi.testclient import TestClient

from services.api_gateway.app.main import app as gateway_app
from services.orchestrator.app.policy import has_exact_pricing

from .evaluator import evaluate, load_questions
from .schemas import Prediction
from .usage_summary import summarize_usage


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
    client = TestClient(gateway_app)
    if max_latency_s is not None:
        ask_response = client.post(
            "/v1/ask",
            json={
                "question": question.question,
                "mode": mode,
                "providers": providers,
                "constraints": {
                    "min_sources": min_sources if min_sources is not None else question.must_have_citations,
                    "max_latency_s": max_latency_s,
                    "max_cost_usd": max_cost_usd,
                },
            },
        )
    else:
        ask_response = client.post(
            "/v1/ask",
            json={
                "question": question.question,
                "mode": mode,
                "providers": providers,
                "constraints": {
                    "min_sources": min_sources if min_sources is not None else question.must_have_citations,
                    "max_cost_usd": max_cost_usd,
                },
            },
        )
    ask_response.raise_for_status()
    run_id = ask_response.json()["run_id"]
    run_response = client.get(f"/v1/runs/{run_id}").json()
    run_status = str(run_response.get("status", ""))
    usage = run_response.get("usage") or {}
    trace_response = client.get(f"/v1/runs/{run_id}/trace")
    trace_response.raise_for_status()
    trace_payload = trace_response.json() or {}
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
