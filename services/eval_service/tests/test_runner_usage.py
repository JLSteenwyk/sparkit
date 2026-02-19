from __future__ import annotations

from services.eval_service.app.runner import _summarize_usage


def test_summarize_usage_aggregates_cost_latency_and_tokens() -> None:
    rows = [
        {"cost_usd": 0.5, "latency_s": 2.0, "tokens_input": 100, "tokens_output": 20},
        {"cost_usd": 1.0, "latency_s": 3.0, "tokens_input": 300, "tokens_output": 40},
    ]
    summary = _summarize_usage(rows)
    assert summary["num_runs"] == 2
    assert summary["total_cost_usd"] == 1.5
    assert summary["avg_cost_usd"] == 0.75
    assert summary["total_latency_s"] == 5.0
    assert summary["avg_latency_s"] == 2.5
    assert summary["total_tokens_input"] == 400
    assert summary["total_tokens_output"] == 60
    assert summary["cost_estimated"] is True
    assert summary["token_usage_partial"] is False
    assert "Synthesis provider tokens" in str(summary["token_usage_notes"])


def test_summarize_usage_handles_empty_rows() -> None:
    summary = _summarize_usage([])
    assert summary["num_runs"] == 0
    assert summary["total_cost_usd"] == 0.0
    assert summary["avg_cost_usd"] == 0.0
    assert summary["total_latency_s"] == 0.0
    assert summary["avg_latency_s"] == 0.0
    assert summary["cost_estimated"] is True
