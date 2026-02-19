from __future__ import annotations

from .schemas import BenchmarkQuestion, Prediction, RubricScore


def _coverage(answer_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lowered = answer_text.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / len(keywords)


def score_prediction(question: BenchmarkQuestion, prediction: Prediction) -> RubricScore:
    required_cov = _coverage(prediction.answer_text, question.required_keywords)
    optional_cov = _coverage(prediction.answer_text, question.optional_keywords)
    keyword_coverage = min(1.0, 0.7 * required_cov + 0.3 * optional_cov)

    citation_score = min(1.0, prediction.citation_count / max(1, question.must_have_citations))
    total = 0.75 * keyword_coverage + 0.25 * citation_score

    return RubricScore(
        id=question.id,
        keyword_coverage=keyword_coverage,
        citation_score=citation_score,
        total_score=total,
    )
