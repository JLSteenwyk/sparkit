from __future__ import annotations

import time
import xml.etree.ElementTree as ET
import os
from urllib.parse import urlparse
from typing import Any

import httpx

from .models import LiteratureRecord

ARXIV_API_URL = "http://export.arxiv.org/api/query"
CROSSREF_API_URL = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API_URL = "https://api.openalex.org/works"
EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"

_SCIENCE_HOST_KEYWORDS = (
    "nature.com",
    "science.org",
    "cell.com",
    "thelancet.com",
    "bmj.com",
    "nejm.org",
    "pubmed.ncbi.nlm.nih.gov",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "acs.org",
    "rsc.org",
    "ieee.org",
    ".edu",
    ".gov",
)


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


def search_openalex(query: str, limit: int = 5, timeout_s: float = 15.0) -> list[LiteratureRecord]:
    params = {
        "search": query,
        "per-page": max(1, min(limit, 25)),
    }
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = _get_with_retry(client, OPENALEX_API_URL, params=params)

    items = response.json().get("results", [])
    records: list[LiteratureRecord] = []
    for item in items:
        title = (item.get("title") or "").strip()
        doi_raw = item.get("doi")
        doi = None
        if isinstance(doi_raw, str) and doi_raw:
            doi = doi_raw.rsplit("doi.org/", 1)[-1]
        url = item.get("primary_location", {}).get("landing_page_url") or item.get("id")
        year = item.get("publication_year")
        if not title or not url:
            continue
        authors = []
        for auth in item.get("authorships", []) or []:
            name = (auth.get("author") or {}).get("display_name")
            if isinstance(name, str) and name.strip():
                authors.append(name.strip())
        abstract = item.get("abstract")
        if abstract is None and isinstance(item.get("abstract_inverted_index"), dict):
            # OpenAlex may return abstract as inverted index.
            inv = item["abstract_inverted_index"]
            words: dict[int, str] = {}
            for token, positions in inv.items():
                for pos in positions:
                    if isinstance(pos, int):
                        words[pos] = token
            if words:
                abstract = " ".join(words[idx] for idx in sorted(words.keys()))
        records.append(
            LiteratureRecord(
                source="openalex",
                title=title,
                abstract=abstract if isinstance(abstract, str) else None,
                authors=authors,
                year=year if isinstance(year, int) else None,
                doi=doi,
                url=url,
            )
        )
    return records


def search_europe_pmc(query: str, limit: int = 5, timeout_s: float = 15.0) -> list[LiteratureRecord]:
    params = {
        "query": query,
        "format": "json",
        "pageSize": max(1, min(limit, 25)),
        "resultType": "core",
    }
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = _get_with_retry(client, EUROPE_PMC_API_URL, params=params)

    items = response.json().get("resultList", {}).get("result", [])
    records: list[LiteratureRecord] = []
    for item in items:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        doi = item.get("doi")
        pmid = item.get("pmid")
        pmcid = item.get("pmcid")
        url = item.get("fullTextUrl")
        if not url:
            if pmcid:
                url = f"https://europepmc.org/article/PMC/{pmcid}"
            elif pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            elif doi:
                url = f"https://doi.org/{doi}"
        if not url:
            continue
        year = None
        year_text = str(item.get("pubYear") or "")
        if year_text.isdigit():
            year = int(year_text)
        authors = []
        author_field = item.get("authorString")
        if isinstance(author_field, str) and author_field.strip():
            authors = [author.strip() for author in author_field.split(",") if author.strip()]
        records.append(
            LiteratureRecord(
                source="europe_pmc",
                title=title,
                abstract=item.get("abstractText"),
                authors=authors,
                year=year,
                doi=doi,
                url=url,
            )
        )
    return records


def search_brave_web(query: str, limit: int = 5, timeout_s: float = 15.0) -> list[LiteratureRecord]:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return []
    params = {
        "q": query,
        "count": max(1, min(limit * 2, 20)),
    }
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
        "User-Agent": "SPARKIT/0.1",
    }
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = _get_with_retry(client, BRAVE_SEARCH_API_URL, params=params, headers=headers)
    results = response.json().get("web", {}).get("results", []) or []
    records: list[LiteratureRecord] = []
    strict_host_filter = str(os.getenv("SPARKIT_BRAVE_STRICT_HOST_FILTER", "0")).lower() in {"1", "true", "yes"}
    for row in results:
        url = row.get("url")
        title = (row.get("title") or "").strip()
        if not url or not title:
            continue
        host = urlparse(url).netloc.lower()
        if strict_host_filter and not any(keyword in host for keyword in _SCIENCE_HOST_KEYWORDS):
            continue
        summary = (row.get("description") or "").strip() or None
        records.append(
            LiteratureRecord(
                source="brave_web",
                title=title,
                abstract=summary,
                authors=[],
                year=None,
                doi=None,
                url=url,
            )
        )
        if len(records) >= limit:
            break
    return records
