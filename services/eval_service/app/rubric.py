from __future__ import annotations

import re

from .exact_match import exact_match_correct
from .schemas import BenchmarkQuestion, Prediction, RubricScore


def _coverage(answer_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lowered = answer_text.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / len(keywords)


_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)
_LEADING_CHOICE_RE = re.compile(r"^\s*([A-Za-z])(?:[\)\].:\s]|$)")


def _extract_choice(answer_text: str) -> str | None:
    if not answer_text:
        return None
    tagged = _ANSWER_TAG_RE.search(answer_text)
    if tagged:
        return tagged.group(1).upper()
    leading = _LEADING_CHOICE_RE.match(answer_text)
    if leading:
        return leading.group(1).upper()
    return None


def score_prediction(question: BenchmarkQuestion, prediction: Prediction) -> RubricScore:
    # MCQ grading is strict exact-match on selected option letter.
    if (question.answer_type or "").lower() == "multiplechoice" and question.correct_answer:
        predicted = _extract_choice(prediction.answer_text or "")
        correct = str(question.correct_answer).strip().upper()[:1]
        is_correct = 1.0 if predicted == correct else 0.0
        return RubricScore(
            id=question.id,
            keyword_coverage=is_correct,
            citation_score=1.0,
            total_score=is_correct,
        )

    # Exact-match grading uses deterministic normalization + optional LLM adjudication.
    if (question.answer_type or "").lower() == "exactmatch" and question.correct_answer:
        is_correct = 1.0 if exact_match_correct(question.question, str(question.correct_answer), prediction.answer_text or "") else 0.0
        return RubricScore(
            id=question.id,
            keyword_coverage=is_correct,
            citation_score=1.0,
            total_score=is_correct,
        )

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
