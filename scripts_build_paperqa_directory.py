#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import textwrap
import xml.etree.ElementTree as ET
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


def _short_query(question: str, max_words: int = 24) -> str:
    q = " ".join(question.split())
    words = q.split()
    return " ".join(words[:max_words]).strip()


def _query_arxiv(*, query: str, max_results: int, timeout_s: float) -> list[dict[str, str]]:
    # arXiv API (Atom): https://export.arxiv.org/api/query
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max(1, min(20, max_results)),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(url, params=params, timeout=timeout_s, headers={"user-agent": "sparkit-reboot/0.1"})
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ns):
        title = _safe_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = _safe_text(entry.findtext("atom:summary", default="", namespaces=ns))
        entry_id = _safe_text(entry.findtext("atom:id", default="", namespaces=ns))
        published = _safe_text(entry.findtext("atom:published", default="", namespaces=ns))
        if not title or not summary:
            continue
        out.append(
            {
                "title": " ".join(title.split()),
                "text": " ".join(summary.split()),
                "url": entry_id,
                "published": published,
                "provider": "arxiv_api",
            }
        )
    return out


def _write_doc(
    *,
    out_dir: Path,
    title: str,
    text: str,
    url: str,
    provider: str,
    qid: str,
    extra_meta: dict[str, Any] | None = None,
) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    filename = f"{_slug(title)}-{digest}.md"
    path = out_dir / filename
    lines = [
        f"# {title}",
        "",
        f"- source_url: {url}",
        f"- source_provider: {provider}",
        f"- question_id: {qid}",
    ]
    if extra_meta:
        for key, value in extra_meta.items():
            if value is None:
                continue
            lines.append(f"- {key}: {_safe_text(value)}")
    lines.extend(["", "## Extracted Text", "", text, ""])
    path.write_text("\n".join(lines))
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PAPERQA_PAPER_DIRECTORY from Exa + arXiv + bioRxiv-targeted retrieval."
    )
    parser.add_argument("--questions", default="benchmarks/hle_gold/questions_full.json")
    parser.add_argument("--output-dir", default="data/paperqa_papers")
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--results-per-question", type=int, default=8, help="General Exa results per question.")
    parser.add_argument("--arxiv-per-question", type=int, default=3, help="arXiv preprints per question.")
    parser.add_argument("--biorxiv-per-question", type=int, default=3, help="bioRxiv pages per question via Exa.")
    parser.add_argument("--timeout-s", type=float, default=25.0)
    parser.add_argument("--include-exa", action="store_true", help="Include general Exa science retrieval.")
    parser.add_argument("--include-arxiv", action="store_true", help="Include arXiv API preprints.")
    parser.add_argument("--include-biorxiv", action="store_true", help="Include bioRxiv-targeted retrieval.")
    parser.add_argument(
        "--query-hint",
        default="pubmed OR doi OR systematic review OR randomized trial OR mechanism study",
    )
    args = parser.parse_args()

    if not (args.include_exa or args.include_arxiv or args.include_biorxiv):
        # Default to all if nothing was explicitly selected.
        args.include_exa = True
        args.include_arxiv = True
        args.include_biorxiv = True

    api_key = os.getenv("EXA_API_KEY", "").strip()
    if (args.include_exa or args.include_biorxiv) and not api_key:
        raise SystemExit("EXA_API_KEY is required when include_exa/include_biorxiv are enabled.")

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
        short_q = _short_query(question)
        query = f"{short_q} {args.query_hint}".strip()
        q_count = 0

        if args.include_exa:
            try:
                results = _query_exa(
                    api_key=api_key,
                    query=query,
                    num_results=args.results_per_question,
                    timeout_s=args.timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                manifest.append({"question_id": qid, "source": "exa", "error": str(exc), "results": 0})
                results = []

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
                path_str = _write_doc(
                    out_dir=out_dir,
                    title=title,
                    text=text,
                    url=url,
                    provider="exa",
                    qid=qid,
                )
                written += 1
                q_count += 1
                manifest.append({"question_id": qid, "source": "exa", "title": title, "url": url, "file": path_str})

        if args.include_arxiv:
            try:
                arxiv_rows = _query_arxiv(query=short_q, max_results=args.arxiv_per_question, timeout_s=args.timeout_s)
            except Exception as exc:  # noqa: BLE001
                manifest.append({"question_id": qid, "source": "arxiv", "error": str(exc), "results": 0})
                arxiv_rows = []

            for item in arxiv_rows:
                url = _safe_text(item.get("url"))
                if not url or url in seen_url:
                    continue
                title = _safe_text(item.get("title")) or "arXiv preprint"
                text = _safe_text(item.get("text"))
                if not text:
                    continue
                seen_url.add(url)
                path_str = _write_doc(
                    out_dir=out_dir,
                    title=title,
                    text=textwrap.shorten(text, width=14000, placeholder="..."),
                    url=url,
                    provider="arxiv_api",
                    qid=qid,
                    extra_meta={"published": _safe_text(item.get("published"))},
                )
                written += 1
                q_count += 1
                manifest.append(
                    {
                        "question_id": qid,
                        "source": "arxiv",
                        "title": title,
                        "url": url,
                        "published": _safe_text(item.get("published")),
                        "file": path_str,
                    }
                )

        if args.include_biorxiv:
            bq = f"{short_q} site:biorxiv.org"
            try:
                biorxiv_rows = _query_exa(
                    api_key=api_key,
                    query=bq,
                    num_results=args.biorxiv_per_question,
                    timeout_s=args.timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                manifest.append({"question_id": qid, "source": "biorxiv", "error": str(exc), "results": 0})
                biorxiv_rows = []

            for item in biorxiv_rows:
                if not isinstance(item, dict):
                    continue
                url = _safe_text(item.get("url"))
                if not url or "biorxiv.org" not in url.lower() or url in seen_url:
                    continue
                title = _safe_text(item.get("title")) or "bioRxiv preprint"
                text = _safe_text(item.get("text")) or _safe_text(item.get("snippet"))
                if not text:
                    text = _fetch_url_text(url, timeout_s=args.timeout_s)
                if not text:
                    continue
                seen_url.add(url)
                path_str = _write_doc(
                    out_dir=out_dir,
                    title=title,
                    text=text,
                    url=url,
                    provider="exa_biorxiv",
                    qid=qid,
                )
                written += 1
                q_count += 1
                manifest.append(
                    {
                        "question_id": qid,
                        "source": "biorxiv",
                        "title": title,
                        "url": url,
                        "file": path_str,
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
