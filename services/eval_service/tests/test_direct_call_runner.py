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
        latency_s = 0.25
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
    assert result["usage_summary"]["total_latency_s"] == 0.25
    assert result["report"]["direct_call"]["failure_count"] == 0


def test_run_direct_single_call_counts_empty_answers_as_failure(monkeypatch, tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "Explain an SN1 reaction mechanism.",
                    "domain": "chemistry",
                    "subdomain": "organic",
                    "required_keywords": ["reaction", "mechanism"],
                    "optional_keywords": [],
                    "must_have_citations": 1,
                    "difficulty": "medium",
                }
            ]
        )
    )

    class _FakeResult:
        success = True
        text = ""
        tokens_input = 30
        tokens_input_cached = 0
        tokens_output = 10
        latency_s = 0.1
        model = "fake-model"
        error = None

    monkeypatch.setattr(direct_runner, "generate_text", lambda provider, prompt, max_tokens=700: _FakeResult())

    result = direct_runner.run_direct_single_call_benchmark_with_predictions(
        questions_path=str(questions_path),
        provider="openai",
        max_questions=1,
    )
    assert result["report"]["direct_call"]["failure_count"] == 1
    assert result["report"]["direct_call"]["failures"][0]["error"] == "empty_answer_text"
    assert result["predictions"][0]["answer_confidence"] == 0.0


def test_run_direct_single_call_retries_timeout_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "What is ATP synthase?",
                    "domain": "biology",
                    "subdomain": "biochemistry",
                    "required_keywords": ["atp", "synthase"],
                    "optional_keywords": [],
                    "must_have_citations": 1,
                    "difficulty": "medium",
                }
            ]
        )
    )

    class _FailResult:
        success = False
        text = ""
        tokens_input = 0
        tokens_input_cached = 0
        tokens_output = 0
        latency_s = 0.2
        model = "fake-model"
        error = "The read operation timed out"

    class _OkResult:
        success = True
        text = '{"answer_text":"ATP synthase uses PMF.","answer_confidence":0.8}'
        tokens_input = 50
        tokens_input_cached = 0
        tokens_output = 20
        latency_s = 0.2
        model = "fake-model"
        error = None

    calls: list[tuple[str, str, int]] = []
    queue = [_FailResult(), _OkResult()]

    def _fake_generate_text(provider: str, prompt: str, max_tokens: int = 700):  # noqa: ANN001
        calls.append((provider, prompt, max_tokens))
        return queue.pop(0)

    monkeypatch.setenv("DIRECT_CALL_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("DIRECT_CALL_RETRY_BACKOFF_S", "0")
    monkeypatch.setattr(direct_runner, "generate_text", _fake_generate_text)

    result = direct_runner.run_direct_single_call_benchmark_with_predictions(
        questions_path=str(questions_path),
        provider="grok",
        max_questions=1,
    )
    assert len(calls) == 2
    assert result["report"]["direct_call"]["failure_count"] == 0
    assert result["predictions"][0]["answer_text"]


def test_run_direct_single_call_deepseek_reasoning_retry_hint(monkeypatch, tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q001",
                    "question": "Explain oxidative phosphorylation.",
                    "domain": "biology",
                    "subdomain": "bioenergetics",
                    "required_keywords": ["oxidative", "phosphorylation"],
                    "optional_keywords": [],
                    "must_have_citations": 1,
                    "difficulty": "medium",
                }
            ]
        )
    )

    class _ReasoningOnlyResult:
        success = False
        text = ""
        tokens_input = 0
        tokens_input_cached = 0
        tokens_output = 0
        latency_s = 0.2
        model = "deepseek-reasoner"
        error = "empty_content with reasoning_content"

    class _OkResult:
        success = True
        text = '{"answer_text":"OxPhos couples ETC proton gradients to ATP production.","answer_confidence":0.7}'
        tokens_input = 60
        tokens_input_cached = 0
        tokens_output = 30
        latency_s = 0.2
        model = "deepseek-reasoner"
        error = None

    prompts: list[str] = []
    queue = [_ReasoningOnlyResult(), _OkResult()]

    def _fake_generate_text(provider: str, prompt: str, max_tokens: int = 700):  # noqa: ANN001
        prompts.append(prompt)
        return queue.pop(0)

    monkeypatch.setenv("DIRECT_CALL_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("DIRECT_CALL_RETRY_BACKOFF_S", "0")
    monkeypatch.setattr(direct_runner, "generate_text", _fake_generate_text)

    result = direct_runner.run_direct_single_call_benchmark_with_predictions(
        questions_path=str(questions_path),
        provider="deepseek",
        max_questions=1,
    )
    assert len(prompts) == 2
    assert "Provide final answer content in message.content only" in prompts[1]
    assert result["report"]["direct_call"]["failure_count"] == 0
