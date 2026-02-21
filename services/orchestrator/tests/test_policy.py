from __future__ import annotations

from services.orchestrator.app.policy import (
    BudgetState,
    contradiction_depth_from_budget,
    estimate_generation_cost,
    estimate_stage_cost,
    has_exact_pricing,
    should_stop_early,
)


def test_policy_stage_cost_positive() -> None:
    assert estimate_stage_cost("retrieval", units=2) == 0.0
    assert estimate_stage_cost("synthesis", units=2) > 0


def test_policy_should_stop_by_cost() -> None:
    state = BudgetState(elapsed_s=10.0, spent_usd=1.9)
    assert should_stop_early(state, max_latency_s=None, max_cost_usd=2.0, reserve_next_stage_usd=0.2) is True


def test_policy_contradiction_depth() -> None:
    assert contradiction_depth_from_budget(10.0, 200) == 3
    assert contradiction_depth_from_budget(5.0, 120) == 2
    assert contradiction_depth_from_budget(2.0, 60) == 1
    assert contradiction_depth_from_budget(10.0, None) == 3


def test_generation_cost_kimi_k2_turbo_uses_cache_hit_miss_output_rates() -> None:
    cost = estimate_generation_cost(
        "kimi",
        "kimi-k2-turbo-preview",
        tokens_input=1_000_000,
        tokens_input_cached=250_000,
        tokens_output=1_000_000,
    )
    # 250k * 0.10 + 750k * 0.60 + 1000k * 3.00 per 1M tokens.
    assert abs(cost - 3.475) < 1e-9
    assert has_exact_pricing("kimi", "kimi-k2-turbo-preview") is True


def test_generation_cost_falls_back_for_unknown_model() -> None:
    cost = estimate_generation_cost(
        "openai",
        "gpt-4o-mini-unknown",
        tokens_input=1000,
        tokens_input_cached=0,
        tokens_output=1600,
    )
    assert cost == estimate_stage_cost("synthesis", units=2)


def test_generation_cost_uses_env_pricing(monkeypatch) -> None:
    monkeypatch.setenv(
        "SPARKIT_MODEL_PRICING_JSON",
        '{"openai:gpt-5.2":{"input_cache_hit":1.0,"input_cache_miss":2.0,"output":3.0}}',
    )
    # Reset cached env pricing for this process.
    import services.orchestrator.app.policy as policy

    policy._load_env_model_pricing.cache_clear()
    cost = policy.estimate_generation_cost(
        "openai",
        "gpt-5.2",
        tokens_input=1_000_000,
        tokens_input_cached=250_000,
        tokens_output=1_000_000,
    )
    assert abs(cost - 4.75) < 1e-9
    assert policy.has_exact_pricing("openai", "gpt-5.2") is True
    policy._load_env_model_pricing.cache_clear()


def test_generation_cost_known_model_defaults_are_exact() -> None:
    assert has_exact_pricing("openai", "gpt-5.2") is True
    assert has_exact_pricing("openai", "gpt-5.2-pro") is True
    assert has_exact_pricing("anthropic", "claude-opus-4-6") is True
    assert has_exact_pricing("anthropic", "claude-sonnet-4-6") is True
    assert has_exact_pricing("gemini", "gemini-3-pro-preview") is True
    assert has_exact_pricing("gemini", "gemini-3.1-pro-preview") is True
    assert has_exact_pricing("kimi", "kimi-k2-turbo-preview") is True
    assert has_exact_pricing("deepseek", "deepseek-reasoner") is True
    assert has_exact_pricing("grok", "grok-4-0709") is True
    assert has_exact_pricing("grok", "grok-4-fast-reasoning") is True
    assert has_exact_pricing("grok", "grok-4-fast-non-reasoning") is True
    assert has_exact_pricing("mistral", "mistral-large-2512") is True


def test_generation_cost_gemini_tiered_pricing_switches_above_200k_prompt() -> None:
    low_tier = estimate_generation_cost(
        "gemini",
        "gemini-3-pro-preview",
        tokens_input=100_000,
        tokens_input_cached=0,
        tokens_output=10_000,
    )
    high_tier = estimate_generation_cost(
        "gemini",
        "gemini-3-pro-preview",
        tokens_input=250_000,
        tokens_input_cached=0,
        tokens_output=10_000,
    )
    assert high_tier > low_tier


def test_generation_cost_gemini_31_tiered_pricing_switches_above_200k_prompt() -> None:
    low_tier = estimate_generation_cost(
        "gemini",
        "gemini-3.1-pro-preview",
        tokens_input=100_000,
        tokens_input_cached=0,
        tokens_output=10_000,
    )
    high_tier = estimate_generation_cost(
        "gemini",
        "gemini-3.1-pro-preview",
        tokens_input=250_000,
        tokens_input_cached=0,
        tokens_output=10_000,
    )
    assert high_tier > low_tier
