from __future__ import annotations

from dataclasses import dataclass

from services.orchestrator.app.providers import ProviderStatus


@dataclass(frozen=True)
class ProviderPlan:
    planning: str
    retrieval: str
    synthesis: str
    verification: str
    ensemble: list[str]


def _configured(statuses: list[ProviderStatus]) -> list[str]:
    return [status.provider for status in statuses if status.configured]


def _pick(preferred: list[str], configured: list[str], fallback: str) -> str:
    for provider in preferred:
        if provider in configured:
            return provider
    return configured[0] if configured else fallback


def _routed_preferences(task_type: str | None) -> dict[str, list[str]]:
    normalized = (task_type or "").strip().lower()
    defaults = {
        "planning": ["anthropic", "openai", "gemini", "mistral", "deepseek", "grok", "kimi"],
        "retrieval": ["openai", "gemini", "deepseek", "mistral", "anthropic", "grok", "kimi"],
        "synthesis": ["anthropic", "openai", "mistral", "gemini", "deepseek", "grok", "kimi"],
        "verification": ["gemini", "openai", "anthropic", "mistral", "deepseek", "grok", "kimi"],
    }
    if normalized == "multiple_choice":
        return {
            "planning": ["anthropic", "openai", "gemini", "mistral", "deepseek", "grok", "kimi"],
            "retrieval": ["gemini", "openai", "deepseek", "mistral", "anthropic", "grok", "kimi"],
            "synthesis": ["anthropic", "openai", "gemini", "mistral", "deepseek", "grok", "kimi"],
            "verification": ["gemini", "anthropic", "openai", "mistral", "deepseek", "grok", "kimi"],
        }
    if normalized == "mechanism":
        return {
            "planning": ["openai", "anthropic", "deepseek", "gemini", "mistral", "grok", "kimi"],
            "retrieval": ["deepseek", "openai", "gemini", "anthropic", "mistral", "grok", "kimi"],
            "synthesis": ["anthropic", "openai", "deepseek", "gemini", "mistral", "grok", "kimi"],
            "verification": ["gemini", "openai", "anthropic", "deepseek", "mistral", "grok", "kimi"],
        }
    if normalized == "numerical":
        return {
            "planning": ["openai", "anthropic", "gemini", "mistral", "deepseek", "grok", "kimi"],
            "retrieval": ["openai", "gemini", "anthropic", "deepseek", "mistral", "grok", "kimi"],
            "synthesis": ["openai", "anthropic", "gemini", "mistral", "deepseek", "grok", "kimi"],
            "verification": ["gemini", "openai", "anthropic", "mistral", "deepseek", "grok", "kimi"],
        }
    return defaults


def build_provider_plan(
    mode: str,
    statuses: list[ProviderStatus],
    requested: list[str],
    task_type: str | None = None,
) -> ProviderPlan:
    configured = _configured(statuses)
    requested_non_empty = requested or ["openai"]

    if mode == "single":
        base = _pick(requested_non_empty, configured, requested_non_empty[0])
        return ProviderPlan(
            planning=base,
            retrieval=base,
            synthesis=base,
            verification=base,
            ensemble=[base],
        )

    if mode == "routed":
        prefs = _routed_preferences(task_type)
        return ProviderPlan(
            planning=_pick(prefs["planning"], configured, "anthropic"),
            retrieval=_pick(prefs["retrieval"], configured, "openai"),
            synthesis=_pick(prefs["synthesis"], configured, "anthropic"),
            verification=_pick(prefs["verification"], configured, "gemini"),
            ensemble=configured[:3] if configured else requested_non_empty[:1],
        )

    # ensemble mode
    preferred = ["openai", "anthropic", "gemini", "mistral", "deepseek", "grok", "kimi"]
    ensemble = [provider for provider in preferred if provider in configured]
    if not ensemble:
        ensemble = requested_non_empty[:3]
    return ProviderPlan(
        planning=ensemble[0],
        retrieval=ensemble[0],
        synthesis=ensemble[0],
        verification=ensemble[min(1, len(ensemble) - 1)],
        ensemble=ensemble[:3],
    )
