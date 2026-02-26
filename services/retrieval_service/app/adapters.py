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
PUBMED_ESEARCH_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_EFETCH_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
EXA_SEARCH_API_URL = "https://api.exa.ai/search"
EXA_CONTENTS_API_URL = "https://api.exa.ai/contents"
EXA_ANSWER_API_URL = "https://api.exa.ai/answer"
EXA_RESEARCH_API_URL = "https://api.exa.ai/research"

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


def search_pubmed_metadata(query: str, limit: int = 5, timeout_s: float = 15.0) -> list[LiteratureRecord]:
    # PubMed E-utilities: metadata-only retrieval path to avoid fulltext publisher scraping failures.
    tool = os.getenv("PUBMED_TOOL", "sparkit")
    email = os.getenv("PUBMED_EMAIL", "")
    api_key = os.getenv("PUBMED_API_KEY", "")
    common = {"tool": tool}
    if email:
        common["email"] = email
    if api_key:
        common["api_key"] = api_key

    search_params: dict[str, Any] = {
        "db": "pubmed",
        "retmode": "json",
        "sort": "relevance",
        "term": query,
        "retmax": max(1, min(limit * 2, 25)),
        **common,
    }
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        search_response = _get_with_retry(client, PUBMED_ESEARCH_API_URL, params=search_params)
        payload = search_response.json()
        id_list = ((payload.get("esearchresult") or {}).get("idlist") or []) if isinstance(payload, dict) else []
        if not id_list:
            return []
        pmids = [str(pid) for pid in id_list[: max(1, min(len(id_list), 50))]]
        summary_params = {
            "db": "pubmed",
            "retmode": "json",
            "id": ",".join(pmids),
            **common,
        }
        summary_response = _get_with_retry(client, PUBMED_ESUMMARY_API_URL, params=summary_params)
        summary_payload = summary_response.json()
        efetch_params = {
            "db": "pubmed",
            "retmode": "xml",
            "id": ",".join(pmids),
            **common,
        }
        efetch_response = _get_with_retry(client, PUBMED_EFETCH_API_URL, params=efetch_params)

    result = summary_payload.get("result") or {}
    uids = result.get("uids") or []
    abstract_by_pmid: dict[str, str] = {}
    try:
        root = ET.fromstring(efetch_response.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            if pmid_node is None or not (pmid_node.text or "").strip():
                continue
            pmid = (pmid_node.text or "").strip()
            parts: list[str] = []
            for abs_node in article.findall(".//Abstract/AbstractText"):
                label = (abs_node.attrib.get("Label") or "").strip()
                text = "".join(abs_node.itertext()).strip()
                if not text:
                    continue
                parts.append(f"{label}: {text}" if label else text)
            if parts:
                abstract_by_pmid[pmid] = " ".join(parts)
    except Exception:
        abstract_by_pmid = {}

    records: list[LiteratureRecord] = []
    for uid in uids:
        row = result.get(str(uid)) or {}
        if not isinstance(row, dict):
            continue
        title = (row.get("title") or "").strip()
        if not title:
            continue
        pubdate = str(row.get("pubdate") or "")
        year = None
        for token in pubdate.split():
            if len(token) == 4 and token.isdigit():
                year = int(token)
                break
        authors = []
        for author in (row.get("authors") or []):
            if not isinstance(author, dict):
                continue
            name = (author.get("name") or "").strip()
            if name:
                authors.append(name)
        article_ids = row.get("articleids") or []
        doi = None
        for aid in article_ids:
            if not isinstance(aid, dict):
                continue
            if str(aid.get("idtype") or "").lower() == "doi":
                doi = str(aid.get("value") or "").strip() or None
                break
        pmid = str(uid).strip()
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        records.append(
            LiteratureRecord(
                source="pubmed",
                title=title,
                abstract=abstract_by_pmid.get(pmid),
                authors=authors,
                year=year,
                doi=doi,
                url=url,
            )
        )
        if len(records) >= max(1, limit):
            break
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


def search_exa_web(query: str, limit: int = 5, timeout_s: float = 20.0) -> list[LiteratureRecord]:
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return []
    payload = {
        "query": query,
        "numResults": max(1, min(limit * 2, 25)),
    }
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": "SPARKIT/0.1",
    }
    last_exc: Exception | None = None
    response: httpx.Response | None = None
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        for attempt in range(3):
            try:
                response = client.post(EXA_SEARCH_API_URL, json=payload, headers=headers)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(0.6 * (2**attempt))
                    continue
                response.raise_for_status()
                break
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt >= 2:
                    raise
                time.sleep(0.6 * (2**attempt))
        if response is None:
            if last_exc:
                raise last_exc
            return []
    data = response.json()
    items = data.get("results", []) if isinstance(data, dict) else []
    records: list[LiteratureRecord] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        url = row.get("url")
        title = (row.get("title") or "").strip()
        if not url or not title:
            continue
        text = (row.get("text") or row.get("snippet") or row.get("summary") or "").strip()
        published = row.get("publishedDate") or row.get("published_date") or ""
        year = None
        if isinstance(published, str) and len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])
        author = row.get("author")
        authors = [author.strip()] if isinstance(author, str) and author.strip() else []
        records.append(
            LiteratureRecord(
                source="exa_web",
                title=title,
                abstract=text or None,
                authors=authors,
                year=year,
                doi=None,
                url=url,
            )
        )
        if len(records) >= limit:
            break
    return records


def _exa_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": "SPARKIT/0.1",
    }


def _exa_post_with_retry(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    retries: int = 2,
    backoff_s: float = 0.6,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    response: httpx.Response | None = None
    for attempt in range(retries + 1):
        try:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(backoff_s * (2**attempt))
                continue
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict):
                return body
            return {}
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(backoff_s * (2**attempt))
    if last_exc:
        raise last_exc
    return {}


def _exa_row_to_record(row: dict[str, Any], source: str) -> LiteratureRecord | None:
    url = row.get("url")
    title = (row.get("title") or "").strip()
    if not isinstance(url, str) or not url.strip() or not title:
        return None
    published = row.get("publishedDate") or row.get("published_date") or ""
    year = None
    if isinstance(published, str) and len(published) >= 4 and published[:4].isdigit():
        year = int(published[:4])
    author = row.get("author")
    authors = [author.strip()] if isinstance(author, str) and author.strip() else []
    text = (row.get("text") or row.get("snippet") or row.get("summary") or "").strip() or None
    return LiteratureRecord(
        source=source,
        title=title,
        abstract=text,
        authors=authors,
        year=year,
        doi=None,
        url=url.strip(),
    )


def fetch_exa_contents(
    urls: list[str],
    timeout_s: float = 30.0,
) -> list[LiteratureRecord]:
    api_key = os.getenv("EXA_API_KEY")
    if not api_key or not urls:
        return []
    clean_urls: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        if not isinstance(raw, str):
            continue
        url = raw.strip()
        if not url or url in seen:
            continue
        seen.add(url)
        clean_urls.append(url)
    if not clean_urls:
        return []
    payload: dict[str, Any] = {
        "urls": clean_urls[:25],
        "text": True,
    }
    headers = _exa_headers(api_key)
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        data = _exa_post_with_retry(client, EXA_CONTENTS_API_URL, payload, headers=headers)
    records: list[LiteratureRecord] = []
    for row in (data.get("results") or []):
        if not isinstance(row, dict):
            continue
        parsed = _exa_row_to_record(row, source="exa_content")
        if parsed:
            records.append(parsed)
    return records


def search_exa_answer(query: str, limit: int = 5, timeout_s: float = 30.0) -> list[LiteratureRecord]:
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return []
    payload: dict[str, Any] = {"query": query, "text": True}
    headers = _exa_headers(api_key)
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        data = _exa_post_with_retry(client, EXA_ANSWER_API_URL, payload, headers=headers)

    # Prefer explicit citations/sources, fallback to results.
    citation_rows: list[dict[str, Any]] = []
    for row in (data.get("citations") or []):
        if isinstance(row, dict):
            citation_rows.append(row)
    if citation_rows:
        out: list[LiteratureRecord] = []
        for row in citation_rows:
            url = row.get("url")
            title = (row.get("title") or "").strip() or "Exa cited source"
            if not isinstance(url, str) or not url.strip():
                continue
            out.append(
                LiteratureRecord(
                    source="exa_answer",
                    title=title,
                    abstract=None,
                    authors=[],
                    year=None,
                    doi=None,
                    url=url.strip(),
                )
            )
            if len(out) >= max(1, limit):
                break
        if out:
            return out

    records: list[LiteratureRecord] = []
    for row in (data.get("results") or []):
        if not isinstance(row, dict):
            continue
        parsed = _exa_row_to_record(row, source="exa_answer")
        if parsed:
            records.append(parsed)
            if len(records) >= max(1, limit):
                break
    return records


def search_exa_research(query: str, limit: int = 5, timeout_s: float = 90.0) -> list[LiteratureRecord]:
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return []
    model = os.getenv("EXA_RESEARCH_MODEL", "exa-research")
    poll_timeout_s = max(5, int(os.getenv("SPARKIT_EXA_RESEARCH_POLL_TIMEOUT_S", "90")))
    poll_interval_s = max(1.0, float(os.getenv("SPARKIT_EXA_RESEARCH_POLL_INTERVAL_S", "2")))
    create_payload: dict[str, Any] = {"instructions": query, "model": model}
    headers = _exa_headers(api_key)
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        created = _exa_post_with_retry(client, EXA_RESEARCH_API_URL, create_payload, headers=headers)
        research_id = (
            created.get("researchId")
            or created.get("id")
            or created.get("research_id")
            or ""
        )
        if not isinstance(research_id, str) or not research_id.strip():
            return []
        deadline = time.time() + poll_timeout_s
        result: dict[str, Any] = {}
        while time.time() < deadline:
            response = client.get(f"{EXA_RESEARCH_API_URL}/{research_id}", headers=headers)
            response.raise_for_status()
            payload = response.json()
            result = payload if isinstance(payload, dict) else {}
            status = str(result.get("status") or "").lower()
            if status in {"completed", "done", "succeeded", "success"}:
                break
            if status in {"failed", "error", "cancelled"}:
                return []
            time.sleep(poll_interval_s)

    out: list[LiteratureRecord] = []
    sources = result.get("sources") or result.get("citations") or result.get("results") or []
    if isinstance(sources, list):
        for row in sources:
            if not isinstance(row, dict):
                continue
            parsed = _exa_row_to_record(row, source="exa_research")
            if parsed:
                out.append(parsed)
            elif isinstance(row.get("url"), str):
                out.append(
                    LiteratureRecord(
                        source="exa_research",
                        title=(row.get("title") or "Exa research source").strip(),
                        abstract=None,
                        authors=[],
                        year=None,
                        doi=None,
                        url=row["url"].strip(),
                    )
                )
            if len(out) >= max(1, limit):
                break
    return out
