from __future__ import annotations

import json
from pathlib import Path

from services.eval_service.app import baseline_capture
from services.eval_service.app.baseline_capture import BaselineConfig, capture_baselines


def test_capture_baselines_includes_usage_fields(monkeypatch, tmp_path: Path) -> None:
    def _fake_run_benchmark_with_predictions(*args, **kwargs):
        return {
            "report": {
                "average_rubric_score": 0.7,
                "calibration": {"brier_score": 0.2, "ece": 0.1},
            },
            "predictions": [{"id": "q001", "answer_text": "x", "answer_confidence": 0.5, "citation_count": 1}],
            "quality_summary": {
                "avg_citation_coverage": 0.8,
                "avg_unsupported_claims": 0.2,
                "avg_contradiction_flags": 0.1,
                "max_unsupported_claims": 1,
                "max_contradiction_flags": 1,
            },
            "usage_summary": {
                "total_cost_usd": 1.2,
                "avg_cost_usd": 1.2,
                "total_latency_s": 12.0,
                "avg_latency_s": 12.0,
                "total_tokens_input": 1000,
                "total_tokens_output": 250,
                "cost_estimated": True,
                "token_usage_partial": False,
                "token_usage_notes": "Synthesis provider tokens are tracked from API usage when available; fallback heuristics may apply.",
            },
        }

    monkeypatch.setattr(baseline_capture, "run_benchmark_with_predictions", _fake_run_benchmark_with_predictions)

    manifest = capture_baselines(
        questions_path="benchmarks/stem_exam_200_sample/questions.json",
        output_dir=str(tmp_path),
        label="test",
        configs=[BaselineConfig(name="single_openai", mode="single", providers=["openai"])],
        max_questions=1,
        skip_missing_keys=False,
    )

    record = manifest["configs"][0]
    assert record["total_cost_usd"] == 1.2
    assert record["avg_cost_usd"] == 1.2
    assert record["total_latency_s"] == 12.0
    assert record["avg_latency_s"] == 12.0
    assert record["total_tokens_input"] == 1000
    assert record["total_tokens_output"] == 250
    assert record["cost_estimated"] is True
    assert record["token_usage_partial"] is False
    assert "Synthesis provider tokens" in record["token_usage_notes"]

    path = tmp_path / manifest["run_slug"] / "manifest.json"
    saved = json.loads(path.read_text())
    assert saved["configs"][0]["avg_latency_s"] == 12.0
