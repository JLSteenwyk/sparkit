from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class BudgetState:
    elapsed_s: float
    spent_usd: float


@dataclass(frozen=True)
class PricingRates:
    input_cache_hit: float
    input_cache_miss: float
    output: float


BRAVE_SEARCH_REQUEST_USD = 5.00 / 1000.0


def estimate_stage_cost(stage: str, units: int = 1) -> float:
    # Non-generation stages do not incur model API costs in the current stack.
    # Keep synthesis fallback deterministic only for models missing exact pricing.
    rates = {
        "retrieval": 0.0,
        "ingestion": 0.0,
        "verification": 0.0,
        "synthesis": 0.02,
        "ensemble": 0.0,
    }
    return rates.get(stage, 0.001) * max(1, units)


def estimate_brave_search_cost(request_count: int) -> float:
    return max(0, int(request_count)) * BRAVE_SEARCH_REQUEST_USD


DEFAULT_MODEL_PRICING: dict[tuple[str, str], PricingRates] = {
    ("openai", "gpt-5.2"): PricingRates(input_cache_hit=0.175, input_cache_miss=1.75, output=14.00),
    # OpenAI API pricing (2026-02-20): gpt-5.2-pro has no published cached-input discount.
    # Use the same rate for cache-hit and cache-miss input to avoid undercounting cost.
    ("openai", "gpt-5.2-pro"): PricingRates(input_cache_hit=21.00, input_cache_miss=21.00, output=168.00),
    ("anthropic", "claude-opus-4-6"): PricingRates(input_cache_hit=0.50, input_cache_miss=5.00, output=25.00),
    # Sonnet 4.6 pricing aligns with Sonnet family ($3 in / $15 out) and Anthropic prompt-caching
    # read multiplier (0.1x base input), yielding $0.30/MTok cache-hit input.
    ("anthropic", "claude-sonnet-4-6"): PricingRates(input_cache_hit=0.30, input_cache_miss=3.00, output=15.00),
    # Gemini Pro Preview variants use prompt-size tiered pricing; baseline tier <=200k tokens.
    ("gemini", "gemini-3-pro-preview"): PricingRates(input_cache_hit=0.20, input_cache_miss=2.00, output=12.00),
    ("google", "gemini-3-pro-preview"): PricingRates(input_cache_hit=0.20, input_cache_miss=2.00, output=12.00),
    ("gemini", "gemini-3.1-pro-preview"): PricingRates(input_cache_hit=0.20, input_cache_miss=2.00, output=12.00),
    ("google", "gemini-3.1-pro-preview"): PricingRates(input_cache_hit=0.20, input_cache_miss=2.00, output=12.00),
    ("kimi", "kimi-k2-turbo-preview"): PricingRates(input_cache_hit=0.10, input_cache_miss=0.60, output=3.00),
    # DeepSeek model pricing page (2026-02-20): unified pricing for V3.2-Exp and Reasoner.
    ("deepseek", "deepseek-reasoner"): PricingRates(input_cache_hit=0.028, input_cache_miss=0.28, output=0.42),
    # xAI pricing docs (2026-02-20).
    ("grok", "grok-4-0709"): PricingRates(input_cache_hit=0.75, input_cache_miss=3.00, output=15.00),
    ("grok", "grok-4-fast-reasoning"): PricingRates(input_cache_hit=0.20, input_cache_miss=0.20, output=0.50),
    ("grok", "grok-4-fast-non-reasoning"): PricingRates(input_cache_hit=0.20, input_cache_miss=0.20, output=0.50),
    # Mistral pricing docs (2026-02-20): mistral-large-2512 lists input/output only.
    # Cache-hit input is set equal to input miss unless a separate cached-input rate is published.
    ("mistral", "mistral-large-2512"): PricingRates(input_cache_hit=2.00, input_cache_miss=2.00, output=6.00),
}


def _parse_rate_blob(blob: dict) -> PricingRates | None:
    try:
        return PricingRates(
            input_cache_hit=float(blob["input_cache_hit"]),
            input_cache_miss=float(blob["input_cache_miss"]),
            output=float(blob["output"]),
        )
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=1)
def _load_env_model_pricing() -> dict[tuple[str, str], PricingRates]:
    raw = os.getenv("SPARKIT_MODEL_PRICING_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(parsed, dict):
        return {}

    pricing: dict[tuple[str, str], PricingRates] = {}

    # Format A:
    # {"provider:model":{"input_cache_hit":...,"input_cache_miss":...,"output":...}}
    for key, value in parsed.items():
        if not isinstance(key, str) or ":" not in key or not isinstance(value, dict):
            continue
        provider, model = key.split(":", 1)
        rates = _parse_rate_blob(value)
        if rates is not None:
            pricing[(provider.strip().lower(), model.strip())] = rates

    # Format B:
    # {"openai":{"gpt-5.2":{"input_cache_hit":...,"input_cache_miss":...,"output":...}}}
    for provider, provider_blob in parsed.items():
        if not isinstance(provider, str) or not isinstance(provider_blob, dict):
            continue
        for model, model_blob in provider_blob.items():
            if not isinstance(model, str) or not isinstance(model_blob, dict):
                continue
            rates = _parse_rate_blob(model_blob)
            if rates is not None:
                pricing[(provider.strip().lower(), model.strip())] = rates

    return pricing


def _resolve_pricing(provider: str, model: str) -> PricingRates | None:
    key = (provider.lower(), model)
    env_pricing = _load_env_model_pricing()
    if key in env_pricing:
        return env_pricing[key]
    return DEFAULT_MODEL_PRICING.get(key)


def has_exact_pricing(provider: str, model: str) -> bool:
    return _resolve_pricing(provider, model) is not None


def estimate_generation_cost(
    provider: str,
    model: str,
    *,
    tokens_input: int,
    tokens_output: int,
    tokens_input_cached: int = 0,
) -> float:
    # Per-1M token pricing map. Fallback is deterministic stage estimator.
    normalized_provider = provider.lower()
    rates = _resolve_pricing(normalized_provider, model)
    # Gemini Pro Preview variants use a higher tier for prompts >200k input tokens.
    if normalized_provider in {"gemini", "google"} and model in {"gemini-3-pro-preview", "gemini-3.1-pro-preview"}:
        if max(0, int(tokens_input)) > 200_000:
            rates = PricingRates(input_cache_hit=0.40, input_cache_miss=4.00, output=18.00)
    if rates is None:
        return estimate_stage_cost("synthesis", units=max(1, tokens_output // 800))

    cached = max(0, int(tokens_input_cached))
    total_input = max(0, int(tokens_input))
    miss = max(0, total_input - cached)
    output = max(0, int(tokens_output))
    return (
        (cached / 1_000_000.0) * rates.input_cache_hit
        + (miss / 1_000_000.0) * rates.input_cache_miss
        + (output / 1_000_000.0) * rates.output
    )


def should_stop_early(
    state: BudgetState,
    max_latency_s: int | None,
    max_cost_usd: float,
    reserve_next_stage_usd: float = 0.0,
) -> bool:
    if max_latency_s is not None and state.elapsed_s >= max_latency_s:
        return True
    if state.spent_usd + reserve_next_stage_usd > max_cost_usd:
        return True
    return False


def contradiction_depth_from_budget(max_cost_usd: float, max_latency_s: int | None) -> int:
    latency = max_latency_s if max_latency_s is not None else 10_000
    if max_cost_usd >= 8.0 and latency >= 180:
        return 3
    if max_cost_usd >= 4.0 and latency >= 120:
        return 2
    return 1
