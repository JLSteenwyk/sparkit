from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkQuestion(BaseModel):
    id: str
    question: str
    domain: str = "general"
    subdomain: str = "general"
    answer_type: str = "exactMatch"
    correct_answer: str | None = None
    required_keywords: list[str] = Field(default_factory=list)
    optional_keywords: list[str] = Field(default_factory=list)
    must_have_citations: int = 1
    difficulty: str = "medium"


class Prediction(BaseModel):
    id: str
    answer_text: str
    answer_confidence: float = Field(ge=0.0, le=1.0)
    citation_count: int = 0


class RubricScore(BaseModel):
    id: str
    keyword_coverage: float
    citation_score: float
    total_score: float


class CalibrationMetrics(BaseModel):
    brier_score: float
    ece: float


class EvaluationReport(BaseModel):
    num_questions: int
    average_rubric_score: float
    rubric_scores: list[RubricScore]
    calibration: CalibrationMetrics
