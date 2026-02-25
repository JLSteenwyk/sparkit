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
    monkeypatch.setattr(aggregator, "search_openalex", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_europe_pmc", lambda query, limit: [])

    records, errors, _stats = aggregator.search_literature("gnn chemistry", max_results=10)

    assert errors == {}
    assert len(records) == 4  # one DOI duplicate merged
    assert any(record.year == 2025 for record in records)
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
    monkeypatch.setattr(aggregator, "search_openalex", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_europe_pmc", lambda query, limit: [])

    records, errors, _stats = aggregator.search_literature("gnn", max_results=5)

    assert len(records) == 3
    crossref_keys = [key for key in errors if key.startswith("crossref:")]
    assert crossref_keys
    assert all("unavailable" in errors[key] for key in crossref_keys)


def test_search_literature_filters_disallowed_domains(monkeypatch):
    allowed = LiteratureRecord(
        source="arxiv",
        title="Allowed Paper",
        abstract="valid",
        year=2024,
        url="https://arxiv.org/abs/1234.5678",
    )
    blocked = LiteratureRecord(
        source="openalex",
        title="Blocked HLE mirror",
        abstract="should not be used",
        year=2024,
        url="https://huggingface.co/datasets/futurehouse/hle-gold-bio-chem",
    )
    monkeypatch.setattr(aggregator, "search_arxiv", lambda query, limit: [allowed])
    monkeypatch.setattr(aggregator, "search_crossref", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_semantic_scholar", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_openalex", lambda query, limit: [blocked])
    monkeypatch.setattr(aggregator, "search_europe_pmc", lambda query, limit: [])

    records, errors, _stats = aggregator.search_literature("oled radicals", max_results=5)
    assert errors == {}
    assert len(records) == 1
    assert records[0].title == "Allowed Paper"


def test_search_literature_retries_with_relaxed_query(monkeypatch):
    calls: list[str] = []

    def flaky_adapter(query: str, limit: int) -> list[LiteratureRecord]:
        calls.append(query)
        if '"' in query:
            raise RuntimeError("bad query syntax")
        return [
            LiteratureRecord(
                source="semantic_scholar",
                title="Recovered on relaxed query",
                abstract="ok",
                year=2024,
                url="https://example.org/recovered",
            )
        ]

    monkeypatch.setattr(aggregator, "search_arxiv", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_crossref", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_semantic_scholar", flaky_adapter)
    monkeypatch.setattr(aggregator, "search_openalex", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_europe_pmc", lambda query, limit: [])

    records, errors, _stats = aggregator.search_literature('"radical OLED broad FWHM"', max_results=5)
    assert records
    assert records[0].title == "Recovered on relaxed query"
    assert any('"' in query for query in calls)
    assert any('"' not in query for query in calls)
    assert errors == {}


def test_search_literature_tracks_brave_request_counts(monkeypatch):
    monkeypatch.setenv("SPARKIT_ENABLE_WEB_SEARCH", "1")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(aggregator, "search_arxiv", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_crossref", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_semantic_scholar", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_openalex", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_europe_pmc", lambda query, limit: [])
    monkeypatch.setattr(
        aggregator,
        "search_brave_web",
        lambda query, limit: [
            LiteratureRecord(
                source="brave_web",
                title=f"Brave hit: {query}",
                abstract="web",
                year=2025,
                url=f"https://example.org/{abs(hash(query))}",
            )
        ],
    )

    records, errors, stats = aggregator.search_literature("biology chemistry", max_results=6)
    assert errors == {}
    assert records
    assert (stats.get("requests_by_source") or {}).get("brave_web", 0) > 0


def test_search_literature_force_web_uses_brave_even_when_env_disabled(monkeypatch):
    monkeypatch.setenv("SPARKIT_ENABLE_WEB_SEARCH", "0")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(aggregator, "search_arxiv", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_crossref", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_semantic_scholar", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_openalex", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_europe_pmc", lambda query, limit: [])
    monkeypatch.setattr(
        aggregator,
        "search_brave_web",
        lambda query, limit: [
            LiteratureRecord(
                source="brave_web",
                title=f"Forced web hit: {query}",
                abstract="web",
                year=2025,
                url=f"https://example.org/force/{abs(hash(query))}",
            )
        ],
    )

    records, errors, stats = aggregator.search_literature("biology chemistry", max_results=6, force_web=True)
    assert errors == {}
    assert records
    assert any(record.source == "brave_web" for record in records)
    assert (stats.get("requests_by_source") or {}).get("brave_web", 0) > 0


def test_search_literature_auto_brave_fallback_on_dns_errors(monkeypatch):
    monkeypatch.setenv("SPARKIT_ENABLE_WEB_SEARCH", "0")
    monkeypatch.setenv("SPARKIT_ENABLE_BRAVE_FALLBACK", "1")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(
        aggregator,
        "search_arxiv",
        lambda query, limit: (_ for _ in ()).throw(RuntimeError("[Errno -2] Name or service not known")),
    )
    monkeypatch.setattr(
        aggregator,
        "search_crossref",
        lambda query, limit: (_ for _ in ()).throw(RuntimeError("[Errno -2] Name or service not known")),
    )
    monkeypatch.setattr(
        aggregator,
        "search_semantic_scholar",
        lambda query, limit: (_ for _ in ()).throw(RuntimeError("[Errno -2] Name or service not known")),
    )
    monkeypatch.setattr(aggregator, "search_openalex", lambda query, limit: [])
    monkeypatch.setattr(aggregator, "search_europe_pmc", lambda query, limit: [])
    monkeypatch.setattr(
        aggregator,
        "search_brave_web",
        lambda query, limit: [
            LiteratureRecord(
                source="brave_web",
                title=f"Brave fallback hit: {query}",
                abstract="web",
                year=2025,
                url=f"https://example.org/fallback/{abs(hash(query))}",
            )
        ],
    )

    records, _errors, stats = aggregator.search_literature("biology chemistry", max_results=6)
    assert records
    assert any(record.source == "brave_web" for record in records)
    assert stats.get("brave_fallback_used") is True


def test_search_literature_can_run_local_only(monkeypatch):
    monkeypatch.setenv("SPARKIT_ENABLE_LIVE_RETRIEVAL", "0")
    monkeypatch.setenv("SPARKIT_ENABLE_LOCAL_CORPUS", "1")
    monkeypatch.setenv("SPARKIT_MIN_QUALITY_SCORE", "0")
    monkeypatch.setenv("SPARKIT_EVIDENCE_ALLOW_DOMAINS", "")
    monkeypatch.setenv("SPARKIT_EVIDENCE_DENY_DOMAINS", "")
    monkeypatch.setattr(
        aggregator.LocalCorpusStore,
        "query",
        lambda self, query, max_results: [
            LiteratureRecord(
                source="local_corpus",
                title="Local-only hit",
                abstract="cached",
                year=2024,
                url="https://example.org/local-only",
            )
        ],
    )
    monkeypatch.setattr(
        aggregator,
        "search_arxiv",
        lambda query, limit: (_ for _ in ()).throw(AssertionError("live adapter should be disabled")),
    )

    records, errors, stats = aggregator.search_literature("local retrieval only", max_results=5)
    assert errors == {}
    assert records
    assert records[0].source == "local_corpus"
    assert stats.get("requests_by_source", {}) == {}
