from __future__ import annotations

import json
from pathlib import Path

from .metrics import calibration_metrics
from .rubric import score_prediction
from .schemas import BenchmarkQuestion, EvaluationReport, Prediction


def load_questions(path: str | Path) -> list[BenchmarkQuestion]:
    raw = json.loads(Path(path).read_text())
    return [BenchmarkQuestion.model_validate(item) for item in raw]


def load_predictions(path: str | Path) -> list[Prediction]:
    raw = json.loads(Path(path).read_text())
    return [Prediction.model_validate(item) for item in raw]


def evaluate(questions: list[BenchmarkQuestion], predictions: list[Prediction]) -> EvaluationReport:
    pred_map = {prediction.id: prediction for prediction in predictions}

    rubric_scores = []
    confidences: list[float] = []
    outcomes: list[int] = []

    for question in questions:
        prediction = pred_map.get(question.id)
        if prediction is None:
            prediction = Prediction(id=question.id, answer_text="", answer_confidence=0.0, citation_count=0)

        rubric = score_prediction(question, prediction)
        rubric_scores.append(rubric)

        confidences.append(prediction.answer_confidence)
        outcomes.append(1 if rubric.total_score >= 0.6 else 0)

    average_score = sum(score.total_score for score in rubric_scores) / max(1, len(rubric_scores))
    calibration = calibration_metrics(confidences, outcomes)

    return EvaluationReport(
        num_questions=len(questions),
        average_rubric_score=average_score,
        rubric_scores=rubric_scores,
        calibration=calibration,
    )
