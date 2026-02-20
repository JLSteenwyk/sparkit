from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from fastapi.testclient import TestClient

from services.api_gateway.app.main import app as gateway_app

from .evaluator import evaluate, load_questions
from .schemas import Prediction


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
        "token_usage_partial": False,
        "token_usage_notes": "Synthesis provider tokens are tracked from API usage when available; fallback heuristics may apply.",
    }


def run_benchmark(
    questions_path: str,
    mode: str = "single",
    providers: list[str] | None = None,
    max_questions: int | None = None,
    min_sources: int | None = None,
    max_latency_s: int = 120,
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
    max_latency_s: int,
    max_cost_usd: float,
) -> dict[str, Any]:
    client = TestClient(gateway_app)
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
    ask_response.raise_for_status()
    run_id = ask_response.json()["run_id"]
    run_response = client.get(f"/v1/runs/{run_id}").json()
    usage = run_response.get("usage") or {}
    trace_response = client.get(f"/v1/runs/{run_id}/trace")
    trace_response.raise_for_status()
    quality = (trace_response.json() or {}).get("quality_gates", {})
    answer = run_response.get("answer") or {}
    citations = run_response.get("citations") or []
    prediction = Prediction(
        id=question.id,
        answer_text=answer.get("final_text", ""),
        answer_confidence=answer.get("answer_confidence", 0.0),
        citation_count=len(citations),
    )
    return {
        "idx": idx,
        "run_id": run_id,
        "usage": {
            "cost_usd": float(usage.get("cost_usd", 0.0)),
            "latency_s": float(usage.get("latency_s", 0.0)),
            "tokens_input": int(usage.get("tokens_input", 0)),
            "tokens_output": int(usage.get("tokens_output", 0)),
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
    max_latency_s: int = 120,
    max_cost_usd: float = 3.0,
    parallel_workers: int = 1,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    if max_questions is not None:
        questions = questions[: max(0, max_questions)]

    predictions: list[Prediction] = []
    run_ids: list[str] = []
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
    return {
        "report": report_dict,
        "predictions": [prediction.model_dump(mode="json") for prediction in predictions],
        "run_ids": run_ids,
        "quality_summary": quality_summary,
        "usage_summary": usage_summary,
    }
