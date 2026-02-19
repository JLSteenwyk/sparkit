from __future__ import annotations

from services.orchestrator.app.providers.clients import generate_text, make_provider_client


def test_generate_text_fails_cleanly_without_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_text("openai", "test prompt")
    assert result.success is False
    assert "OPENAI_API_KEY" in (result.error or "")


def test_generate_text_supports_google_key_alias(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")

    client = make_provider_client("gemini")
    assert client.provider == "gemini"
