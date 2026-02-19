from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from .models import LiteratureRecord

ARXIV_API_URL = "http://export.arxiv.org/api/query"
CROSSREF_API_URL = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def _first(items: list[Any], default: Any = None) -> Any:
    return items[0] if items else default


def _get_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    headers: dict[str, str] | None = None,
    retries: int = 2,
    backoff_s: float = 0.6,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.get(url, params=params, headers=headers)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(backoff_s * (2**attempt))
                continue
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(backoff_s * (2**attempt))

    if last_exc:
        raise last_exc
    raise RuntimeError("request failed without exception")


def search_arxiv(query: str, limit: int = 5, timeout_s: float = 15.0) -> list[LiteratureRecord]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max(1, min(limit, 25)),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = _get_with_retry(client, ARXIV_API_URL, params=params)

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    records: list[LiteratureRecord] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip() or None
        url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
        authors = [
            (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for author in entry.findall("atom:author", ns)
        ]

        doi = None
        for link in entry.findall("atom:link", ns):
            href = link.attrib.get("href", "")
            if "doi.org" in href:
                doi = href.rsplit("/", 1)[-1]
                break

        if title and url:
            records.append(
                LiteratureRecord(
                    source="arxiv",
                    title=title,
                    abstract=summary,
                    authors=[a for a in authors if a],
                    year=year,
                    doi=doi,
                    url=url,
                )
            )
    return records


def search_crossref(query: str, limit: int = 5, timeout_s: float = 15.0) -> list[LiteratureRecord]:
    params = {
        "query": query,
        "rows": max(1, min(limit, 25)),
        "select": "DOI,title,author,issued,URL,abstract",
        "sort": "relevance",
        "order": "desc",
    }
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = _get_with_retry(client, CROSSREF_API_URL, params=params)

    items = response.json().get("message", {}).get("items", [])
    records: list[LiteratureRecord] = []
    for item in items:
        title = _first(item.get("title", []), "") or ""
        doi = item.get("DOI")
        url = item.get("URL")
        if not title or not url:
            continue

        year_parts = item.get("issued", {}).get("date-parts", [[None]])
        year = _first(_first(year_parts, [None]), None)

        authors = []
        for author in item.get("author", []):
            given = author.get("given", "").strip()
            family = author.get("family", "").strip()
            full = f"{given} {family}".strip()
            if full:
                authors.append(full)

        records.append(
            LiteratureRecord(
                source="crossref",
                title=title.strip(),
                abstract=item.get("abstract"),
                authors=authors,
                year=year if isinstance(year, int) else None,
                doi=doi,
                url=url,
            )
        )
    return records


def search_semantic_scholar(
    query: str, limit: int = 5, timeout_s: float = 15.0
) -> list[LiteratureRecord]:
    params = {
        "query": query,
        "limit": max(1, min(limit, 20)),
        "fields": "title,abstract,authors,year,doi,url",
    }
    headers = {"User-Agent": "SPARKIT/0.1"}
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = _get_with_retry(client, SEMANTIC_SCHOLAR_API_URL, params=params, headers=headers)

    items = response.json().get("data", [])
    records: list[LiteratureRecord] = []
    for item in items:
        title = (item.get("title") or "").strip()
        url = item.get("url")
        if not title or not url:
            continue

        authors = [a.get("name", "").strip() for a in item.get("authors", []) if a.get("name")]
        records.append(
            LiteratureRecord(
                source="semantic_scholar",
                title=title,
                abstract=item.get("abstract"),
                authors=authors,
                year=item.get("year") if isinstance(item.get("year"), int) else None,
                doi=item.get("doi"),
                url=url,
            )
        )
    return records
