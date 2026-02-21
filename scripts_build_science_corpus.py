from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from services.ingestion_service.app.parser import fetch_and_parse
from services.retrieval_service.app.adapters import (
    search_arxiv,
    search_brave_web,
    search_crossref,
    search_europe_pmc,
    search_openalex,
    search_semantic_scholar,
)
from services.retrieval_service.app.local_corpus import LocalCorpusStore
from services.retrieval_service.app.models import LiteratureRecord

DOMAIN_QUERIES: dict[str, list[str]] = {
    "biology": [
        "single cell RNA-seq differentiation trajectory",
        "gene regulation embryonic stem cell lineage commitment",
        "CRISPR functional genomics screen essential genes",
        "protein structure function prediction benchmark",
    ],
    "medicine": [
        "randomized clinical trial survival endpoint oncology",
        "systematic review meta analysis adverse events therapeutics",
        "diagnostic biomarker sensitivity specificity cohort study",
        "longitudinal cohort disease progression prediction",
    ],
    "chemistry": [
        "enantioselective catalysis reaction mechanism kinetics",
        "organic synthesis route optimization yield selectivity",
        "electrocatalysis overpotential turnover frequency stability",
        "spectroscopic assignment reaction intermediate",
    ],
    "physics": [
        "quantum materials transport measurement phase transition",
        "superconducting qubit decoherence mitigation benchmark",
        "plasma diagnostics spectroscopy confinement instability",
        "high energy physics cross section measurement",
    ],
    "materials_science": [
        "battery electrode degradation mechanism operando characterization",
        "perovskite stability encapsulation device performance",
        "polymer composite mechanical property prediction",
        "catalyst support morphology activity relationship",
    ],
    "computer_science": [
        "retrieval augmented generation evaluation faithfulness",
        "graph neural network benchmark chemistry property prediction",
        "foundation model robustness out of distribution",
        "reinforcement learning sample efficiency benchmark",
    ],
}


def _dedupe_key(record: LiteratureRecord) -> str:
    if record.doi:
        return f"doi:{record.doi.lower()}"
    return f"url:{record.url.lower()}"


def _collect_records(query: str, per_adapter: int) -> list[LiteratureRecord]:
    adapters = [
        search_arxiv,
        search_crossref,
        search_semantic_scholar,
        search_openalex,
        search_europe_pmc,
        search_brave_web,
    ]
    rows: list[LiteratureRecord] = []
    for adapter in adapters:
        try:
            rows.extend(adapter(query, per_adapter))
        except Exception:  # noqa: BLE001
            continue
    deduped: list[LiteratureRecord] = []
    seen: set[str] = set()
    for record in rows:
        key = _dedupe_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a broad local science corpus for SPARKIT local-first retrieval.")
    parser.add_argument("--domains", default="biology,medicine,chemistry,physics,materials_science,computer_science")
    parser.add_argument("--queries-per-domain", type=int, default=4)
    parser.add_argument("--per-adapter-results", type=int, default=10)
    parser.add_argument("--max-docs-per-domain", type=int, default=120)
    parser.add_argument("--ingest-max-chars", type=int, default=50000)
    parser.add_argument("--report-out", default="benchmarks/results/corpus_build_last.json")
    args = parser.parse_args()

    selected_domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    store = LocalCorpusStore()
    store.ensure_schema()

    counters = defaultdict(int)
    failures = defaultdict(int)
    started = datetime.now(timezone.utc)

    for domain in selected_domains:
        query_pool = DOMAIN_QUERIES.get(domain, [])
        if not query_pool:
            continue
        queries = query_pool[: max(1, args.queries_per_domain)]
        domain_records: list[LiteratureRecord] = []
        for query in queries:
            domain_records.extend(_collect_records(query, per_adapter=max(2, args.per_adapter_results)))
        deduped: list[LiteratureRecord] = []
        seen: set[str] = set()
        for record in domain_records:
            key = _dedupe_key(record)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
            if len(deduped) >= args.max_docs_per_domain:
                break

        for record in deduped:
            try:
                parsed = fetch_and_parse(record.url, max_chars=args.ingest_max_chars, timeout_s=20.0)
                text = "\n\n".join(section.text for section in parsed.sections if section.text.strip())
                if not text.strip():
                    text = record.abstract or ""
                if not text.strip():
                    failures[domain] += 1
                    continue
                store.upsert_document(record=record, text=text, domain=domain, subdomain=None)
                counters[domain] += 1
            except Exception:  # noqa: BLE001
                failures[domain] += 1

    finished = datetime.now(timezone.utc)
    report = {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "domains": selected_domains,
        "ingested_docs": dict(counters),
        "ingest_failures": dict(failures),
        "total_ingested": sum(counters.values()),
        "total_failures": sum(failures.values()),
    }
    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
