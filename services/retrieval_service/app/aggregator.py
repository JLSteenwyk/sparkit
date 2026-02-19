from __future__ import annotations

from collections.abc import Callable

from .adapters import search_arxiv, search_crossref, search_semantic_scholar
from .models import LiteratureRecord

Adapter = Callable[[str, int], list[LiteratureRecord]]


def _dedupe_key(record: LiteratureRecord) -> str:
    if record.doi:
        return f"doi:{record.doi.lower()}"
    return f"url:{record.url.lower()}"


def search_literature(query: str, max_results: int = 12) -> tuple[list[LiteratureRecord], dict[str, str]]:
    per_source = max(2, min(8, max_results // 3 + 1))
    adapters: list[tuple[str, Adapter]] = [
        ("arxiv", search_arxiv),
        ("crossref", search_crossref),
        ("semantic_scholar", search_semantic_scholar),
    ]

    combined: list[LiteratureRecord] = []
    errors: dict[str, str] = {}

    for source_name, adapter in adapters:
        try:
            combined.extend(adapter(query, per_source))
        except Exception as exc:  # noqa: BLE001
            errors[source_name] = str(exc)

    ranked = sorted(combined, key=lambda x: ((x.year or 0), len(x.title)), reverse=True)

    deduped: list[LiteratureRecord] = []
    seen: set[str] = set()
    for record in ranked:
        key = _dedupe_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
        if len(deduped) >= max_results:
            break

    return deduped, errors
