from __future__ import annotations

from services.eval_service.app import exact_match


def test_exact_match_no_llm_returns_false(monkeypatch) -> None:
    monkeypatch.setattr(exact_match, "_llm_exact_match_anthropic", lambda q, g, p: None)
    monkeypatch.setattr(exact_match, "_llm_exact_match_openai", lambda q, g, p: None)
    assert exact_match.exact_match_correct("q", "gold", "pred") is False


def test_exact_match_requires_both_llm_true(monkeypatch) -> None:
    monkeypatch.setattr(exact_match, "_llm_exact_match_anthropic", lambda q, g, p: True)
    monkeypatch.setattr(exact_match, "_llm_exact_match_openai", lambda q, g, p: False)
    assert exact_match.exact_match_correct("q", "gold", "pred") is False

    monkeypatch.setattr(exact_match, "_llm_exact_match_openai", lambda q, g, p: True)
    assert exact_match.exact_match_correct("q", "gold", "pred") is True
