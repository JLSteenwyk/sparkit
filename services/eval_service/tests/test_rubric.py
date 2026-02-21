from __future__ import annotations

from services.eval_service.app.rubric import score_prediction
from services.eval_service.app.schemas import BenchmarkQuestion, Prediction


def test_rubric_scoring_keyword_and_citation() -> None:
    question = BenchmarkQuestion(
        id="q1",
        question="test",
        required_keywords=["alpha", "beta"],
        optional_keywords=["gamma"],
        must_have_citations=2,
    )
    prediction = Prediction(
        id="q1",
        answer_text="Alpha and beta are discussed, plus gamma.",
        answer_confidence=0.8,
        citation_count=2,
    )
    result = score_prediction(question, prediction)
    assert result.total_score > 0.9


def test_rubric_scoring_multiple_choice_exact_match() -> None:
    question = BenchmarkQuestion(
        id="q_mc",
        question="Which is correct? A/B/C",
        answer_type="multipleChoice",
        correct_answer="B",
    )
    good = Prediction(
        id="q_mc",
        answer_text="<answer>B</answer>",
        answer_confidence=0.8,
        citation_count=0,
    )
    bad = Prediction(
        id="q_mc",
        answer_text="<answer>C</answer>",
        answer_confidence=0.8,
        citation_count=0,
    )
    assert score_prediction(question, good).total_score == 1.0
    assert score_prediction(question, bad).total_score == 0.0


def test_rubric_scoring_exactmatch_llm_only_default_false() -> None:
    question = BenchmarkQuestion(
        id="q_ex",
        question="Identify final compound.",
        answer_type="exactMatch",
        correct_answer="Racemate of 4-phenyl-2-bromobutane",
    )
    pred = Prediction(
        id="q_ex",
        answer_text="Final answer: racemic mixture 4-phenyl-2-bromobutane.",
        answer_confidence=0.7,
        citation_count=0,
    )
    assert score_prediction(question, pred).total_score == 0.0
