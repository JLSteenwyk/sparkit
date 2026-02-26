from __future__ import annotations

import argparse
import json
from pathlib import Path

from services.ingestion_service.app.parser import fetch_and_parse
from services.retrieval_service.app.local_corpus import LocalCorpusStore
from services.retrieval_service.app.models import LiteratureRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a Codex-curated evidence URL pack into local corpus.")
    parser.add_argument(
        "--pack",
        default="benchmarks/hle_gold_bio_chem/barometer10_codex_web_pack.json",
        help="Path to JSON evidence pack.",
    )
    parser.add_argument("--ingest-max-chars", type=int, default=60000)
    parser.add_argument("--report-out", default="benchmarks/results/codex_pack_ingest_last.json")
    args = parser.parse_args()

    entries = json.loads(Path(args.pack).read_text())
    store = LocalCorpusStore()
    store.ensure_schema()

    report: dict[str, object] = {
        "pack": args.pack,
        "attempted": 0,
        "ingested": 0,
        "failed": 0,
        "rows": [],
    }

    for row in entries:
        qid = str(row.get("qid", "unknown"))
        url = str(row.get("url", "")).strip()
        title = str(row.get("title", "")).strip() or url
        if not url:
            continue
        report["attempted"] = int(report["attempted"]) + 1
        record = LiteratureRecord(
            source="codex_web_pack",
            title=title,
            abstract=None,
            authors=[],
            year=None,
            doi=None,
            url=url,
        )
        try:
            parsed = fetch_and_parse(url, max_chars=args.ingest_max_chars, timeout_s=25.0)
            text = "\n\n".join(section.text for section in parsed.sections if section.text.strip())
            if not text.strip():
                text = title
            store.upsert_document(record=record, text=text, domain="codex_pack", subdomain=qid)
            report["ingested"] = int(report["ingested"]) + 1
            report["rows"].append({"qid": qid, "url": url, "status": "ingested"})
        except Exception as exc:  # noqa: BLE001
            report["failed"] = int(report["failed"]) + 1
            report["rows"].append({"qid": qid, "url": url, "status": "failed", "error": str(exc)})

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
