from __future__ import annotations

import re
from collections.abc import Callable
from collections import defaultdict

from .adapters import search_arxiv, search_crossref, search_semantic_scholar
from .models import LiteratureRecord

Adapter = Callable[[str, int], list[LiteratureRecord]]


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


def search_literature(query: str, max_results: int = 12) -> tuple[list[LiteratureRecord], dict[str, str]]:
    per_source = max(2, min(8, max_results // 3 + 1))
    rewritten_queries = _rewrite_queries(query, max_extra=1)
    query_tokens = _tokenize(query)
    adapters: list[tuple[str, Adapter]] = [
        ("arxiv", search_arxiv),
        ("crossref", search_crossref),
        ("semantic_scholar", search_semantic_scholar),
    ]

    combined: list[LiteratureRecord] = []
    errors: dict[str, str] = {}

    for source_name, adapter in adapters:
        for rewritten in rewritten_queries:
            try:
                combined.extend(adapter(rewritten, per_source))
            except Exception as exc:  # noqa: BLE001
                errors[source_name] = str(exc)
                break

    ranked = sorted(combined, key=lambda x: _relevance_score(x, query_tokens), reverse=True)

    deduped: list[LiteratureRecord] = []
    seen: set[str] = set()
    for record in ranked:
        key = _dedupe_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    diverse = _limit_source_dominance(deduped, max_results=max_results)
    return diverse, errors
