from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from shared.schemas.api import AskAcceptedResponse, AskRequest
from shared.schemas.domain import Run, RunStatusResponse, RunTraceResponse, Status

from services.orchestrator.app.versioning import (
    CONFIG_VERSION,
    PROMPT_VERSION,
    build_reproducibility_record,
)

from .state import make_run_id, run_store

app = FastAPI(title="SPARKIT API Gateway", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api_gateway"}


@app.post("/v1/ask", response_model=AskAcceptedResponse, status_code=202)
def ask(request: AskRequest) -> AskAcceptedResponse:
    run_id = make_run_id()
    now = datetime.now(timezone.utc)
    providers = request.providers or []
    reproducibility = build_reproducibility_record(
        question=request.question,
        mode=request.mode.value,
        providers=providers,
        constraints=request.constraints.model_dump(mode='json'),
        prompt_version=PROMPT_VERSION,
        config_version=CONFIG_VERSION,
    )

    run = Run(
        run_id=run_id,
        question=request.question,
        mode=request.mode,
        status=Status.QUEUED,
        constraints=request.constraints,
        answer_style=request.answer_style,
        providers=providers,
        include_trace=request.include_trace,
        prompt_version=PROMPT_VERSION,
        config_version=CONFIG_VERSION,
        reproducibility=reproducibility,
        created_at=now,
        updated_at=now,
    )
    run_store.create_run(run)
    return AskAcceptedResponse(
        run_id=run_id,
        status=Status.QUEUED,
        created_at=now,
        poll_url=f"/v1/runs/{run_id}",
    )


@app.get("/v1/runs/{run_id}", response_model=RunStatusResponse)
def get_run(run_id: str) -> RunStatusResponse:
    run = run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    run_store.execute_run_if_needed(run_id)
    run = run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="run missing after execution")
    state = run_store.get_state(run_id)
    if state is None:
        raise HTTPException(status_code=500, detail="run state missing")

    return RunStatusResponse(
        run_id=run.run_id,
        status=run.status,
        mode=run.mode,
        progress=state["progress"],
        answer=state["answer"],
        citations=state["citations"],
        usage=state["usage"],
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@app.get("/v1/runs/{run_id}/trace", response_model=RunTraceResponse)
def get_trace(run_id: str) -> RunTraceResponse:
    run = run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    state = run_store.get_state(run_id)
    if state is None:
        raise HTTPException(status_code=500, detail="run state missing")

    return state["trace"]


@app.post("/v1/runs/{run_id}/cancel", status_code=202)
def cancel_run(run_id: str) -> dict[str, str]:
    result = run_store.cancel_run(run_id)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="run not found")
    if result in {"terminal", "already_cancelling"}:
        raise HTTPException(status_code=409, detail="run is already terminal or cancelling")
    return {"run_id": run_id, "status": "cancelling"}
