from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .evaluator import evaluate, load_predictions, load_questions
from .runner import run_benchmark

app = FastAPI(title="SPARKIT Eval Service", version="0.1.0")


class EvaluateRequest(BaseModel):
    questions_path: str
    predictions_path: str


class RunBenchmarkRequest(BaseModel):
    questions_path: str = Field(default="benchmarks/stem_exam_200_sample/questions.json")
    mode: str = Field(default="single")
    providers: list[str] | None = None
    max_questions: int | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "eval_service"}


@app.post("/evaluate")
def evaluate_from_files(request: EvaluateRequest):
    questions = load_questions(request.questions_path)
    predictions = load_predictions(request.predictions_path)
    report = evaluate(questions, predictions)
    return report.model_dump(mode="json")


@app.post("/run-benchmark")
def run_benchmark_endpoint(request: RunBenchmarkRequest):
    return run_benchmark(
        questions_path=request.questions_path,
        mode=request.mode,
        providers=request.providers,
        max_questions=request.max_questions,
    )
