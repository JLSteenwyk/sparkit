from fastapi import FastAPI
from pydantic import BaseModel, Field

from .aggregator import search_literature
from .models import LiteratureRecord

app = FastAPI(title="SPARKIT Retrieval Service", version="0.1.0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    max_results: int = Field(default=12, ge=1, le=50)


class SearchResponse(BaseModel):
    records: list[LiteratureRecord]
    source_errors: dict[str, str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "retrieval_service"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    records, source_errors = search_literature(request.query, request.max_results)
    return SearchResponse(records=records, source_errors=source_errors)
