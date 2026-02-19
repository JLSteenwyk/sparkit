from __future__ import annotations

from services.orchestrator.app.providers import ProviderStatus
from services.orchestrator.app.routing import build_provider_plan


def test_single_mode_uses_requested_provider() -> None:
    statuses = [ProviderStatus(provider="openai", configured=True, env_var="OPENAI_API_KEY")]
    plan = build_provider_plan(mode="single", statuses=statuses, requested=["openai"])
    assert plan.planning == "openai"
    assert plan.retrieval == "openai"
    assert plan.synthesis == "openai"


def test_routed_mode_prefers_specialized_providers() -> None:
    statuses = [
        ProviderStatus(provider="anthropic", configured=True, env_var="ANTHROPIC_API_KEY"),
        ProviderStatus(provider="openai", configured=True, env_var="OPENAI_API_KEY"),
        ProviderStatus(provider="gemini", configured=True, env_var="GEMINI_API_KEY"),
    ]
    plan = build_provider_plan(mode="routed", statuses=statuses, requested=["openai", "anthropic", "gemini"])
    assert plan.planning == "anthropic"
    assert plan.retrieval == "openai"
    assert plan.synthesis == "anthropic"
    assert plan.verification == "gemini"


def test_ensemble_mode_picks_multiple_providers() -> None:
    statuses = [
        ProviderStatus(provider="openai", configured=True, env_var="OPENAI_API_KEY"),
        ProviderStatus(provider="anthropic", configured=True, env_var="ANTHROPIC_API_KEY"),
    ]
    plan = build_provider_plan(mode="ensemble", statuses=statuses, requested=["openai", "anthropic"])
    assert len(plan.ensemble) == 2
