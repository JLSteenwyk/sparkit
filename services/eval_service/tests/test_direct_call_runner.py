from __future__ import annotations

import json
from pathlib import Path

import services.eval_service.app.direct_call_runner as direct_runner


def test_parse_direct_response_json_and_fallback() -> None:
    answer, conf = direct_runner._parse_direct_response('{"answer_text":"abc","answer_confidence":0.7}')
    assert answer == "abc"
    assert conf == 0.7

    answer2, conf2 = direct_runner._parse_direct_response("plain text output")
    assert answer2 == "plain text output"
    assert conf2 == 0.5


def test_run_direct_single_call_benchmark(monkeypatch, tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "What is catalyst selectivity?",
                    "domain": "chemistry",
                    "subdomain": "catalysis",
                    "required_keywords": ["catalyst", "selectivity"],
                    "optional_keywords": ["reaction"],
                    "must_have_citations": 1,
                    "difficulty": "medium",
                }
            ]
        )
    )

    class _FakeResult:
        success = True
        text = '{"answer_text":"Catalyst selectivity improves reaction outcomes.","answer_confidence":0.8}'
        tokens_input = 100
        tokens_input_cached = 0
        tokens_output = 40
        model = "fake-model"
        error = None

    monkeypatch.setattr(direct_runner, "generate_text", lambda provider, prompt, max_tokens=700: _FakeResult())

    result = direct_runner.run_direct_single_call_benchmark_with_predictions(
        questions_path=str(questions_path),
        provider="openai",
        max_questions=1,
    )
    assert result["report"]["num_questions"] == 1
    assert len(result["predictions"]) == 1
    assert result["usage_summary"]["total_tokens_input"] == 100
    assert result["usage_summary"]["total_tokens_output"] == 40
    assert result["report"]["direct_call"]["failure_count"] == 0

