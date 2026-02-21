from __future__ import annotations

import pytest

from services.eval_service.app.baseline_capture import config_required_vars, parse_configs
from services.eval_service.app.benchmark_generation import build_questions


def test_generate_stem_exam_200_question_count_and_ids() -> None:
    questions = build_questions()
    assert len(questions) == 200

    ids = [question["id"] for question in questions]
    assert len(set(ids)) == 200
    assert ids[0] == "q001"
    assert ids[-1] == "q200"


def test_parse_baseline_configs_subset() -> None:
    configs = parse_configs("single_openai,single_openai_pro,single_anthropic_sonnet,routed_frontier")
    assert [config.name for config in configs] == [
        "single_openai",
        "single_openai_pro",
        "single_anthropic_sonnet",
        "routed_frontier",
    ]


def test_parse_baseline_configs_invalid_name_raises() -> None:
    with pytest.raises(ValueError):
        parse_configs("does_not_exist")


def test_required_key_vars_for_gemini_provider() -> None:
    vars_needed = config_required_vars(["gemini"])
    assert "GEMINI_API_KEY" in vars_needed
    assert "GOOGLE_API_KEY" in vars_needed


def test_required_key_vars_for_new_providers() -> None:
    vars_needed = config_required_vars(["deepseek", "grok", "mistral"])
    assert "DEEPSEEK_API_KEY" in vars_needed
    assert "GROK_API_KEY" in vars_needed
    assert "MISTRAL_API_KEY" in vars_needed
