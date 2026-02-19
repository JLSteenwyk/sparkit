from __future__ import annotations

from services.orchestrator.app.providers import build_default_registry


def test_provider_registry_detects_env_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    registry = build_default_registry()
    statuses = registry.resolve(["openai", "anthropic"])

    by_provider = {s.provider: s for s in statuses}
    assert by_provider["openai"].configured is True
    assert by_provider["openai"].env_var == "OPENAI_API_KEY"
    assert by_provider["anthropic"].configured is False
    assert by_provider["anthropic"].env_var == "ANTHROPIC_API_KEY"


def test_provider_registry_supports_google_alias(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    registry = build_default_registry()
    statuses = registry.resolve(["gemini"])

    assert statuses[0].configured is True
    assert statuses[0].env_var == "GOOGLE_API_KEY"
