from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetState:
    elapsed_s: float
    spent_usd: float


def estimate_stage_cost(stage: str, units: int = 1) -> float:
    # Lightweight deterministic estimator for policy gating.
    rates = {
        "retrieval": 0.004,
        "ingestion": 0.003,
        "verification": 0.002,
        "synthesis": 0.02,
        "ensemble": 0.035,
    }
    return rates.get(stage, 0.001) * max(1, units)


def estimate_generation_cost(
    provider: str,
    model: str,
    *,
    tokens_input: int,
    tokens_output: int,
    tokens_input_cached: int = 0,
) -> float:
    # Per-1M token pricing map. Fallback is deterministic stage estimator.
    pricing_per_million: dict[tuple[str, str], dict[str, float]] = {
        ("kimi", "kimi-k2.5"): {
            "input_cache_hit": 0.10,
            "input_cache_miss": 0.60,
            "output": 3.00,
        },
    }
    key = (provider.lower(), model)
    rates = pricing_per_million.get(key)
    if rates is None:
        return estimate_stage_cost("synthesis", units=max(1, tokens_output // 800))

    cached = max(0, int(tokens_input_cached))
    total_input = max(0, int(tokens_input))
    miss = max(0, total_input - cached)
    output = max(0, int(tokens_output))
    return (
        (cached / 1_000_000.0) * rates["input_cache_hit"]
        + (miss / 1_000_000.0) * rates["input_cache_miss"]
        + (output / 1_000_000.0) * rates["output"]
    )


def should_stop_early(
    state: BudgetState,
    max_latency_s: int,
    max_cost_usd: float,
    reserve_next_stage_usd: float = 0.0,
) -> bool:
    if state.elapsed_s >= max_latency_s:
        return True
    if state.spent_usd + reserve_next_stage_usd > max_cost_usd:
        return True
    return False


def contradiction_depth_from_budget(max_cost_usd: float, max_latency_s: int) -> int:
    if max_cost_usd >= 8.0 and max_latency_s >= 180:
        return 3
    if max_cost_usd >= 4.0 and max_latency_s >= 120:
        return 2
    return 1
