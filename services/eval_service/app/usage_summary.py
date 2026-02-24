from __future__ import annotations


def summarize_usage(
    usage_rows: list[dict[str, float | int]],
    *,
    token_usage_partial: bool,
    token_usage_notes: str,
) -> dict[str, float | int | bool | str]:
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
        "token_usage_partial": token_usage_partial,
        "token_usage_notes": token_usage_notes,
    }

