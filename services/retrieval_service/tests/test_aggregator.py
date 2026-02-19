from __future__ import annotations

import json
from pathlib import Path

from services.retrieval_service.app import aggregator
from services.retrieval_service.app.models import LiteratureRecord


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> list[LiteratureRecord]:
    raw = json.loads((FIXTURE_DIR / name).read_text())
    return [LiteratureRecord.model_validate(item) for item in raw]


def test_search_literature_dedupes_and_ranks(monkeypatch):
    arxiv_records = _load_fixture("arxiv_fixture.json")
    crossref_records = _load_fixture("crossref_fixture.json")
    semantic_records = _load_fixture("semantic_fixture.json")

    monkeypatch.setattr(aggregator, "search_arxiv", lambda query, limit: arxiv_records)
    monkeypatch.setattr(aggregator, "search_crossref", lambda query, limit: crossref_records)
    monkeypatch.setattr(aggregator, "search_semantic_scholar", lambda query, limit: semantic_records)

    records, errors = aggregator.search_literature("gnn chemistry", max_results=10)

    assert errors == {}
    assert len(records) == 4  # one DOI duplicate merged
    assert records[0].year == 2025
    dois = [r.doi for r in records if r.doi]
    assert dois.count("10.1000/xyz123") == 1


def test_search_literature_collects_source_errors(monkeypatch):
    monkeypatch.setattr(aggregator, "search_arxiv", lambda query, limit: _load_fixture("arxiv_fixture.json"))
    monkeypatch.setattr(
        aggregator,
        "search_crossref",
        lambda query, limit: (_ for _ in ()).throw(RuntimeError("crossref unavailable")),
    )
    monkeypatch.setattr(
        aggregator, "search_semantic_scholar", lambda query, limit: _load_fixture("semantic_fixture.json")
    )

    records, errors = aggregator.search_literature("gnn", max_results=5)

    assert len(records) == 3
    assert "crossref" in errors
    assert "unavailable" in errors["crossref"]
