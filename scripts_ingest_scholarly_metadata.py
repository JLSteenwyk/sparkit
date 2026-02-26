from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.ingestion_service.app.parser import fetch_and_parse
from services.retrieval_service.app.adapters import (
    search_arxiv,
    search_crossref,
    search_europe_pmc,
    search_openalex,
    search_pubmed_metadata,
    search_semantic_scholar,
)
from services.retrieval_service.app.local_corpus import LocalCorpusStore
from services.retrieval_service.app.models import LiteratureRecord


def _dedupe_key(record: LiteratureRecord) -> str:
    if record.doi:
        return f"doi:{record.doi.lower()}"
    return f"url:{record.url.lower()}"


def _collect_metadata(query: str, per_adapter: int, include_pubmed: bool) -> list[LiteratureRecord]:
    adapters = [
        ("arxiv", search_arxiv),
        ("crossref", search_crossref),
        ("semantic_scholar", search_semantic_scholar),
        ("openalex", search_openalex),
        ("europe_pmc", search_europe_pmc),
    ]
    if include_pubmed:
        adapters.append(("pubmed", search_pubmed_metadata))

    rows: list[LiteratureRecord] = []
    for _name, adapter in adapters:
        try:
            rows.extend(adapter(query, limit=max(1, per_adapter)))
        except Exception:
            continue
    out: list[LiteratureRecord] = []
    seen: set[str] = set()
    for row in rows:
        key = _dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest scholarly API metadata into local corpus for a question set.")
    parser.add_argument("--questions", required=True, help="Path to question JSON.")
    parser.add_argument("--per-adapter", type=int, default=6)
    parser.add_argument("--max-docs-per-question", type=int, default=24)
    parser.add_argument("--include-pubmed", action="store_true")
    parser.add_argument(
        "--fulltext-top-k",
        type=int,
        default=6,
        help="For each question, attempt fulltext parsing on top-k metadata candidates; fallback to abstract/title.",
    )
    parser.add_argument("--ingest-max-chars", type=int, default=45000)
    parser.add_argument("--report-out", default="benchmarks/results/scholarly_metadata_ingest_last.json")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    store = LocalCorpusStore()
    store.ensure_schema()
    report: dict[str, object] = {
        "questions_path": args.questions,
        "per_adapter": args.per_adapter,
        "max_docs_per_question": args.max_docs_per_question,
        "include_pubmed": bool(args.include_pubmed),
        "fulltext_top_k": int(args.fulltext_top_k),
        "rows": [],
        "ingested_total": 0,
    }

    for row in questions:
        qid = str(row.get("id", "unknown"))
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        stem = question.splitlines()[0].strip()
        records = _collect_metadata(stem, per_adapter=args.per_adapter, include_pubmed=bool(args.include_pubmed))
        kept = records[: max(1, args.max_docs_per_question)]
        ingested_for_q = 0
        for idx, record in enumerate(kept):
            text = ""
            if idx < max(0, int(args.fulltext_top_k)):
                try:
                    parsed = fetch_and_parse(record.url, max_chars=max(1000, int(args.ingest_max_chars)), timeout_s=15.0)
                    sections = [section.text for section in parsed.sections if section.text.strip()]
                    if sections:
                        text = "\n\n".join(sections)
                except Exception:
                    text = ""
            if not text:
                text = (record.abstract or "").strip()
            if not text:
                # Metadata-only backfill from title/authors/year when abstract/fulltext is missing.
                author_str = ", ".join(record.authors[:5]) if record.authors else ""
                text = f"{record.title}\nYear: {record.year or 'n.d.'}\nAuthors: {author_str}".strip()
            store.upsert_document(record=record, text=text, domain="scholarly_metadata", subdomain=qid)
            ingested_for_q += 1
        report["rows"].append({"id": qid, "query": stem, "ingested_docs": ingested_for_q})
        report["ingested_total"] = int(report["ingested_total"]) + ingested_for_q

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"ingested_total": report["ingested_total"], "rows": len(report["rows"])}, indent=2))


if __name__ == "__main__":
    main()
