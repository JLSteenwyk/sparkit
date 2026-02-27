#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Any

import requests


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:120] or "doc"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _query_exa(*, api_key: str, query: str, num_results: int, timeout_s: float) -> list[dict[str, Any]]:
    url = os.getenv("EXA_SEARCH_URL", "https://api.exa.ai/search")
    payload = {"query": query, "numResults": max(1, min(25, num_results)), "type": "auto", "text": True}
    headers = {"x-api-key": api_key, "content-type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("results") if isinstance(data, dict) else None
    return rows if isinstance(rows, list) else []


def _fetch_url_text(url: str, timeout_s: float) -> str:
    try:
        resp = requests.get(url, timeout=timeout_s, headers={"user-agent": "sparkit-reboot/0.1"})
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()
        if "html" not in content_type and "text" not in content_type:
            return ""
        raw = resp.text
    except Exception:  # noqa: BLE001
        return ""

    raw = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw)
    raw = re.sub(r"(?is)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:15000]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local PAPERQA_PAPER_DIRECTORY from Exa science search results.")
    parser.add_argument("--questions", default="benchmarks/hle_gold/questions_full.json")
    parser.add_argument("--output-dir", default="data/paperqa_papers")
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--results-per-question", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=25.0)
    parser.add_argument(
        "--query-hint",
        default="pubmed OR doi OR systematic review OR randomized trial OR mechanism study",
    )
    args = parser.parse_args()

    api_key = os.getenv("EXA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("EXA_API_KEY is required.")

    questions = json.loads(Path(args.questions).read_text())
    if args.max_questions > 0:
        questions = questions[: args.max_questions]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_url: set[str] = set()
    manifest: list[dict[str, Any]] = []
    written = 0

    for row in questions:
        question = _safe_text(row.get("question"))
        qid = _safe_text(row.get("id"))
        if not question:
            continue
        query = f"{question} {args.query_hint}".strip()
        try:
            results = _query_exa(
                api_key=api_key,
                query=query,
                num_results=args.results_per_question,
                timeout_s=args.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            manifest.append({"question_id": qid, "error": str(exc), "results": 0})
            continue

        q_count = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            url = _safe_text(item.get("url"))
            if not url or url in seen_url:
                continue
            title = _safe_text(item.get("title")) or "Untitled"
            text = _safe_text(item.get("text")) or _safe_text(item.get("snippet"))
            if not text:
                text = _fetch_url_text(url, timeout_s=args.timeout_s)
            if not text:
                continue
            seen_url.add(url)

            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
            filename = f"{_slug(title)}-{digest}.md"
            path = out_dir / filename
            path.write_text(
                "\n".join(
                    [
                        f"# {title}",
                        "",
                        f"- source_url: {url}",
                        f"- source_provider: exa",
                        f"- question_id: {qid}",
                        "",
                        "## Extracted Text",
                        "",
                        text,
                        "",
                    ]
                )
            )
            written += 1
            q_count += 1
            manifest.append(
                {
                    "question_id": qid,
                    "title": title,
                    "url": url,
                    "file": str(path),
                }
            )
        if q_count == 0:
            manifest.append({"question_id": qid, "results": 0})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "documents_written": written,
                "unique_urls": len(seen_url),
                "manifest": str(manifest_path),
                "export_hint": f"export PAPERQA_PAPER_DIRECTORY='{out_dir.resolve()}'",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
