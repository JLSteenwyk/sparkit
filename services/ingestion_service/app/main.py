from fastapi import FastAPI, HTTPException

from .models import IngestionRequest, IngestionResponse
from .parser import fetch_and_parse

app = FastAPI(title="SPARKIT Ingestion Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ingestion_service"}


@app.post("/ingest", response_model=IngestionResponse)
def ingest(request: IngestionRequest) -> IngestionResponse:
    try:
        return fetch_and_parse(request.url, max_chars=request.max_chars)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"ingestion failed: {exc}") from exc
