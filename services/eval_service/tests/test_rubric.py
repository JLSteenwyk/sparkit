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
