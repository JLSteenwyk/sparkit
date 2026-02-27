from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sparkit.evidence.schema import EvidenceItem, EvidenceType, FederatedEvidencePack, StudyType
from sparkit.providers.base import EvidenceProvider, ProviderQuery


@dataclass(frozen=True)
class FederationConfig:
    top_k: int = 30
    provider_max_items: int = 20


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
    collected: list[EvidenceItem] = []
    provider_stats: dict[str, dict[str, int | float]] = {}

    for provider in providers:
        items = provider.search(ProviderQuery(question=question, max_items=cfg.provider_max_items))
        collected.extend(items)
        provider_stats[provider.name] = {
            "returned": len(items),
            "avg_confidence": (sum(max(0.0, min(1.0, item.confidence)) for item in items) / len(items)) if items else 0.0,
        }

    deduped = _dedupe(collected)
    ranked = sorted(deduped, key=_score, reverse=True)[: cfg.top_k]
    return FederatedEvidencePack(question=question, items=ranked, provider_stats=provider_stats)

