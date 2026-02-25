from __future__ import annotations

import os
import re
from collections.abc import Callable
from collections import defaultdict
from urllib.parse import urlparse

from .adapters import (
    search_arxiv,
    search_brave_web,
    search_crossref,
    search_europe_pmc,
    search_openalex,
    search_semantic_scholar,
)
from .models import LiteratureRecord
from .local_corpus import LocalCorpusStore

Adapter = Callable[[str, int], list[LiteratureRecord]]

_DISALLOWED_EVIDENCE_DOMAINS = {
    "huggingface.co",
    "futurehouse.org",
}

_HIGH_TRUST_HOST_SUFFIXES = {
    "arxiv.org",
    "nature.com",
    "science.org",
    "cell.com",
    "nejm.org",
    "thelancet.com",
    "pnas.org",
    "acs.org",
    "rsc.org",
    "wiley.com",
    "springer.com",
    "elsevier.com",
    "sciencedirect.com",
    "biomedcentral.com",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "europepmc.org",
}

_MEDIUM_TRUST_HOST_SUFFIXES = {
    "biorxiv.org",
    "medrxiv.org",
    "semanticscholar.org",
    "crossref.org",
    "openalex.org",
    "wikipedia.org",
}

_LOW_TRUST_HOST_SUFFIXES = {
    "reddit.com",
    "medium.com",
    "substack.com",
    "blogspot.com",
}


def _dedupe_key(record: LiteratureRecord) -> str:
    if record.doi:
        return f"doi:{record.doi.lower()}"
    return f"url:{record.url.lower()}"


_QUERY_REWRITES: dict[str, list[str]] = {
    "benchmark": ["evaluation", "comparison", "state of the art"],
    "limitations": ["failure modes", "caveats", "weaknesses"],
    "contradictory": ["negative results", "replication", "disagreement"],
    "biology": ["molecular", "cellular", "genomics"],
    "chemistry": ["reaction", "catalysis", "molecular"],
    "medicine": ["clinical", "patient", "trial"],
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _relax_query(query: str) -> str:
    no_quotes = query.replace('"', " ")
    no_parens = re.sub(r"[()]", " ", no_quotes)
    compact = re.sub(r"\s+", " ", no_parens).strip()
    # Keep reasonable length for APIs that reject long/over-quoted queries.
    terms = compact.split()
    if len(terms) > 18:
        terms = terms[:18]
    return " ".join(terms)


def _rewrite_queries(query: str, max_extra: int = 1) -> list[str]:
    lowered = query.lower()
    extras: list[str] = []
    for token, expansions in _QUERY_REWRITES.items():
        if token in lowered:
            extras.extend(expansions)
    deduped: list[str] = []
    seen: set[str] = set()
    for piece in extras:
        rewritten = f"{query} {piece}".strip()
        if rewritten in seen:
            continue
        seen.add(rewritten)
        deduped.append(rewritten)
        if len(deduped) >= max_extra:
            break
    return [query, *deduped]


def _relevance_score(record: LiteratureRecord, query_tokens: set[str]) -> float:
    hay = f"{record.title} {record.abstract or ''}".lower()
    overlap = sum(1 for token in query_tokens if token in hay)
    year_bonus = (record.year or 0) / 10000.0
    source_bonus = {"semantic_scholar": 0.03, "crossref": 0.02, "arxiv": 0.01}.get(record.source, 0.0)
    return overlap + year_bonus + source_bonus


def _limit_source_dominance(records: list[LiteratureRecord], max_results: int) -> list[LiteratureRecord]:
    if not records:
        return []
    per_source_cap = max(1, max_results // 2)
    selected: list[LiteratureRecord] = []
    counts: dict[str, int] = defaultdict(int)
    deferred: list[LiteratureRecord] = []
    for record in records:
        if counts[record.source] < per_source_cap:
            selected.append(record)
            counts[record.source] += 1
        else:
            deferred.append(record)
        if len(selected) >= max_results:
            return selected
    for record in deferred:
        selected.append(record)
        if len(selected) >= max_results:
            break
    return selected


def _is_disallowed_record(record: LiteratureRecord) -> bool:
    host = urlparse(record.url).netloc.lower()
    return any(host == dom or host.endswith(f".{dom}") for dom in _DISALLOWED_EVIDENCE_DOMAINS)


def _host_from_url(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


def _suffix_match(host: str, suffixes: set[str]) -> bool:
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


def _env_domain_set(name: str) -> set[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return set()
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _source_base_trust(record: LiteratureRecord) -> float:
    source = (record.source or "").lower()
    if source in {"arxiv", "crossref", "semantic_scholar", "openalex", "europe_pmc", "local_corpus"}:
        return 0.9
    if source == "brave_web":
        return 0.45
    return 0.4


def _domain_trust_score(record: LiteratureRecord) -> float:
    host = _host_from_url(record.url)
    if not host:
        return 0.35
    if _suffix_match(host, _HIGH_TRUST_HOST_SUFFIXES):
        return 0.95
    if _suffix_match(host, _MEDIUM_TRUST_HOST_SUFFIXES):
        return 0.7
    if _suffix_match(host, _LOW_TRUST_HOST_SUFFIXES):
        return 0.2
    return 0.5


def _quality_score(record: LiteratureRecord) -> float:
    source_trust = _source_base_trust(record)
    domain_trust = _domain_trust_score(record)
    doi_bonus = 0.05 if record.doi else 0.0
    recency_bonus = 0.0
    if record.year:
        recency_bonus = max(0.0, min(0.08, (record.year - 2015) / 200.0))
    return max(0.0, min(1.0, (0.6 * source_trust) + (0.4 * domain_trust) + doi_bonus + recency_bonus))


def _passes_domain_policy(record: LiteratureRecord) -> bool:
    host = _host_from_url(record.url)
    deny_domains = _env_domain_set("SPARKIT_EVIDENCE_DENY_DOMAINS")
    if not deny_domains:
        deny_domains = set(_DISALLOWED_EVIDENCE_DOMAINS)
    if any(host == dom or host.endswith(f".{dom}") for dom in deny_domains):
        return False
    allow_domains = _env_domain_set("SPARKIT_EVIDENCE_ALLOW_DOMAINS")
    if allow_domains:
        return any(host == dom or host.endswith(f".{dom}") for dom in allow_domains)
    return True


def _passes_quality_policy(record: LiteratureRecord) -> bool:
    min_quality_score = float(os.getenv("SPARKIT_MIN_QUALITY_SCORE", "0.35"))
    if _quality_score(record) < max(0.0, min(1.0, min_quality_score)):
        return False
    source = (record.source or "").lower()
    host = _host_from_url(record.url)
    allow_low_trust_brave = str(os.getenv("SPARKIT_ALLOW_LOW_TRUST_BRAVE", "0")).lower() in {"1", "true", "yes"}
    if source == "brave_web" and not allow_low_trust_brave:
        if _suffix_match(host, _LOW_TRUST_HOST_SUFFIXES):
            return False
    return True


def search_literature(
    query: str,
    max_results: int = 12,
    force_web: bool = False,
) -> tuple[list[LiteratureRecord], dict[str, str], dict[str, dict[str, int]]]:
    per_source = max(2, min(8, max_results // 3 + 1))
    rewritten_queries = _rewrite_queries(query, max_extra=1)
    query_tokens = _tokenize(query)
    live_enabled = str(os.getenv("SPARKIT_ENABLE_LIVE_RETRIEVAL", "1")).lower() not in {"0", "false", "no"}
    adapters: list[tuple[str, Adapter]] = []
    if live_enabled:
        adapters = [
            ("arxiv", search_arxiv),
            ("crossref", search_crossref),
            ("semantic_scholar", search_semantic_scholar),
            ("openalex", search_openalex),
            ("europe_pmc", search_europe_pmc),
        ]
    web_enabled = force_web or str(os.getenv("SPARKIT_ENABLE_WEB_SEARCH", "0")).lower() in {"1", "true", "yes"}
    if live_enabled and web_enabled:
        adapters.append(("brave_web", search_brave_web))

    combined: list[LiteratureRecord] = []
    errors: dict[str, str] = {}
    request_counts: dict[str, int] = defaultdict(int)
    success_counts: dict[str, int] = defaultdict(int)

    local_fraction = max_results if not live_enabled else max(2, max_results // 3)
    local_enabled = str(os.getenv("SPARKIT_ENABLE_LOCAL_CORPUS", "1")).lower() not in {"0", "false", "no"}
    if local_enabled:
        try:
            local_records = LocalCorpusStore().query(query=query, max_results=local_fraction)
            combined.extend(local_records)
        except Exception as exc:  # noqa: BLE001
            strict_local = str(os.getenv("SPARKIT_LOCAL_CORPUS_REQUIRED", "0")).lower() in {"1", "true", "yes"}
            if strict_local:
                errors["local_corpus"] = str(exc)

    for source_name, adapter in adapters:
        for rewritten in rewritten_queries:
            attempted = [rewritten]
            relaxed = _relax_query(rewritten)
            if relaxed and relaxed.lower() != rewritten.lower():
                attempted.append(relaxed)
            last_error: str | None = None
            any_success = False
            try:
                for candidate in attempted:
                    request_counts[source_name] += 1
                    hits = adapter(candidate, per_source)
                    if hits:
                        combined.extend(hits)
                        any_success = True
                        success_counts[source_name] += 1
                        break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                # Retry once with relaxed query form if first attempt fails.
                if len(attempted) > 1:
                    try:
                        request_counts[source_name] += 1
                        hits = adapter(attempted[-1], per_source)
                        if hits:
                            combined.extend(hits)
                            any_success = True
                            success_counts[source_name] += 1
                    except Exception as retry_exc:  # noqa: BLE001
                        last_error = str(retry_exc)
            if not any_success and last_error:
                errors[f"{source_name}:{rewritten}"] = last_error

    ranked = sorted(
        combined,
        key=lambda x: (_relevance_score(x, query_tokens) + (0.6 * _quality_score(x))),
        reverse=True,
    )

    deduped: list[LiteratureRecord] = []
    seen: set[str] = set()
    filtered_low_quality = 0
    filtered_domain_policy = 0
    for record in ranked:
        if _is_disallowed_record(record):
            continue
        if not _passes_domain_policy(record):
            filtered_domain_policy += 1
            continue
        if not _passes_quality_policy(record):
            filtered_low_quality += 1
            continue
        key = _dedupe_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    diverse = _limit_source_dominance(deduped, max_results=max_results)
    stats = {
        "requests_by_source": dict(request_counts),
        "successful_requests_by_source": dict(success_counts),
        "filtered_low_quality": {"count": filtered_low_quality},
        "filtered_domain_policy": {"count": filtered_domain_policy},
    }
    return diverse, errors, stats
