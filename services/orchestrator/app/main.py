from fastapi import FastAPI
from pydantic import BaseModel, Field

from .engine import execute_orchestration

app = FastAPI(title="SPARKIT Orchestrator Service", version="0.1.0")


class OrchestrateRequest(BaseModel):
    question: str = Field(min_length=1)
    min_sources: int = Field(default=5, ge=1, le=50)
    providers: list[str] | None = None
    mode: str = "single"
    synthesis_max_tokens: int | None = Field(default=None, ge=128)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "orchestrator"}


@app.post("/orchestrate")
def orchestrate(request: OrchestrateRequest):
    result = execute_orchestration(
        run_id='run_orchestrator_preview',
        question=request.question,
        min_sources=request.min_sources,
        providers=request.providers,
        mode=request.mode,
        synthesis_max_tokens=request.synthesis_max_tokens,
    )
    return {
        "answer": result.answer,
        "citations": result.citations,
        "stages": result.stages,
        "quality_gates": result.quality_gates,
        "source_errors": result.source_errors,
    }
