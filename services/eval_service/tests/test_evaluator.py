from __future__ import annotations

from pathlib import Path

from services.eval_service.app.evaluator import evaluate, load_predictions, load_questions


BASE = Path('benchmarks/stem_exam_200_sample')


def test_evaluator_with_sample_files() -> None:
    questions = load_questions(BASE / 'questions.json')
    predictions = load_predictions(BASE / 'predictions_sample.json')

    report = evaluate(questions, predictions)
    assert report.num_questions == 3
    assert report.average_rubric_score > 0.5
    assert 0.0 <= report.calibration.brier_score <= 1.0
    assert 0.0 <= report.calibration.ece <= 1.0
