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


def search_literature(query: str, max_results: int = 12) -> tuple[list[LiteratureRecord], dict[str, str], dict[str, dict[str, int]]]:
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
    web_enabled = str(os.getenv("SPARKIT_ENABLE_WEB_SEARCH", "0")).lower() in {"1", "true", "yes"}
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

    ranked = sorted(combined, key=lambda x: _relevance_score(x, query_tokens), reverse=True)

    deduped: list[LiteratureRecord] = []
    seen: set[str] = set()
    for record in ranked:
        if _is_disallowed_record(record):
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
    }
    return diverse, errors, stats
