from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

from sparkit.evidence.schema import EvidenceItem, EvidenceType, FederatedEvidencePack, StudyType
from sparkit.providers.base import EvidenceProvider, ProviderQuery


@dataclass(frozen=True)
class FederationConfig:
    top_k: int = 30
    provider_max_items: int = 20
    enable_mcq_option_queries: bool = True
    max_query_variants: int = 6


_STUDY_WEIGHT = {
    StudyType.META_ANALYSIS: 1.00,
    StudyType.SYSTEMATIC_REVIEW: 0.95,
    StudyType.RANDOMIZED_TRIAL: 0.90,
    StudyType.OBSERVATIONAL: 0.75,
    StudyType.PRECLINICAL: 0.60,
    StudyType.CASE_REPORT: 0.50,
    StudyType.PREPRINT: 0.35,
    StudyType.UNKNOWN: 0.45,
}

_EVIDENCE_WEIGHT = {
    EvidenceType.SUPPORTING: 1.00,
    EvidenceType.NEUTRAL: 0.75,
    EvidenceType.CONTRADICTING: 0.95,
}

_PROVIDER_PRIOR = {
    "paperqa2": 1.00,
    "scite": 0.95,
    "exa": 0.90,
    "consensus": 0.92,
    "elicit": 0.92,
}

_CHOICE_RE = re.compile(r"^\s*([A-Z])[.)]\s*(.+?)\s*$", re.MULTILINE)


def _score(item: EvidenceItem) -> float:
    conf = max(0.0, min(1.0, item.confidence))
    return conf * _STUDY_WEIGHT[item.study_type] * _EVIDENCE_WEIGHT[item.evidence_type]


def _dedupe(items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
    by_id: dict[str, EvidenceItem] = {}
    for item in items:
        key = item.identity()
        existing = by_id.get(key)
        if existing is None or _score(item) > _score(existing):
            by_id[key] = item
    return list(by_id.values())


def build_evidence_pack(
    question: str,
    providers: list[EvidenceProvider],
    config: FederationConfig | None = None,
) -> FederatedEvidencePack:
    cfg = config or FederationConfig()
    queries = _query_variants(question, cfg)
    per_query_items = max(3, cfg.provider_max_items // max(1, len(queries)))
    collected: list[EvidenceItem] = []
    provider_stats: dict[str, dict[str, int | float]] = {}

    for provider in providers:
        provider_items: list[EvidenceItem] = []
        for variant in queries:
            items = provider.search(ProviderQuery(question=variant, max_items=per_query_items))
            for item in items:
                if "query_variant" not in item.provenance:
                    item.provenance["query_variant"] = variant
            provider_items.extend(items)
        collected.extend(provider_items)
        provider_stats[provider.name] = {
            "returned": len(provider_items),
            "avg_confidence": (sum(max(0.0, min(1.0, item.confidence)) for item in provider_items) / len(provider_items))
            if provider_items
            else 0.0,
        }

    deduped = _dedupe(collected)
    ranked = sorted(deduped, key=lambda item: _rerank_score(item, question), reverse=True)[: cfg.top_k]
    return FederatedEvidencePack(question=question, items=ranked, provider_stats=provider_stats)


def _query_variants(question: str, cfg: FederationConfig) -> list[str]:
    if not cfg.enable_mcq_option_queries:
        return [question]
    stem, choices = _parse_mcq(question)
    if not choices:
        return [question]
    queries = [stem] + [f"{stem} {text}".strip() for text in choices.values()]
    limited = queries[: max(1, cfg.max_query_variants)]
    # Preserve order while deduping.
    return list(dict.fromkeys(limited))


def _parse_mcq(question: str) -> tuple[str, dict[str, str]]:
    marker = "Answer Choices:"
    if marker not in question:
        return question.strip(), {}
    stem, rest = question.split(marker, 1)
    choices: dict[str, str] = {}
    for label, text in _CHOICE_RE.findall(rest):
        choices[label] = " ".join(text.split())
    return stem.strip(), choices


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _rerank_score(item: EvidenceItem, question: str) -> float:
    base = _score(item)
    text = " ".join(part for part in (item.claim, item.title or "", item.abstract or "") if part)
    if not text.strip():
        return base * 0.5

    stem, choices = _parse_mcq(question)
    q_tokens = _tokens(stem or question)
    t_tokens = _tokens(text)
    if not t_tokens:
        return base * 0.5

    q_overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
    option_overlaps = []
    for ctext in choices.values():
        c_tokens = _tokens(ctext)
        option_overlaps.append(len(c_tokens & t_tokens) / max(1, len(c_tokens)))
    choice_overlap = max(option_overlaps) if option_overlaps else 0.0
    provider_prior = _PROVIDER_PRIOR.get(item.provider, 0.85)

    lexical = (0.7 * q_overlap) + (0.3 * choice_overlap)
    return (0.55 * base) + (0.35 * lexical) + (0.10 * provider_prior)
