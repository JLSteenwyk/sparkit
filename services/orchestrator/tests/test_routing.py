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


def test_unknown_mode_falls_back_to_ensemble_behavior() -> None:
    statuses = [
        ProviderStatus(provider="openai", configured=True, env_var="OPENAI_API_KEY"),
        ProviderStatus(provider="anthropic", configured=True, env_var="ANTHROPIC_API_KEY"),
    ]
    plan = build_provider_plan(mode="unknown", statuses=statuses, requested=["openai", "anthropic"])
    assert plan.ensemble


def test_routed_mode_mcq_prefers_gemini_for_retrieval_and_verification() -> None:
    statuses = [
        ProviderStatus(provider="anthropic", configured=True, env_var="ANTHROPIC_API_KEY"),
        ProviderStatus(provider="openai", configured=True, env_var="OPENAI_API_KEY"),
        ProviderStatus(provider="gemini", configured=True, env_var="GEMINI_API_KEY"),
    ]
    plan = build_provider_plan(
        mode="routed",
        statuses=statuses,
        requested=["openai", "anthropic", "gemini"],
        task_type="multiple_choice",
    )
    assert plan.retrieval == "gemini"
    assert plan.verification == "gemini"
    assert plan.synthesis == "anthropic"


def test_routed_mode_mechanism_prefers_deepseek_for_retrieval_when_available() -> None:
    statuses = [
        ProviderStatus(provider="anthropic", configured=True, env_var="ANTHROPIC_API_KEY"),
        ProviderStatus(provider="openai", configured=True, env_var="OPENAI_API_KEY"),
        ProviderStatus(provider="deepseek", configured=True, env_var="DEEPSEEK_API_KEY"),
        ProviderStatus(provider="gemini", configured=True, env_var="GEMINI_API_KEY"),
    ]
    plan = build_provider_plan(
        mode="routed",
        statuses=statuses,
        requested=["openai", "anthropic", "deepseek", "gemini"],
        task_type="mechanism",
    )
    assert plan.retrieval == "deepseek"
