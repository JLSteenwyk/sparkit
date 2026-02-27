from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceType(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"


class StudyType(str, Enum):
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    RANDOMIZED_TRIAL = "randomized_trial"
    OBSERVATIONAL = "observational"
    PRECLINICAL = "preclinical"
    CASE_REPORT = "case_report"
    PREPRINT = "preprint"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidenceItem:
    provider: str
    claim: str
    evidence_type: EvidenceType
    study_type: StudyType
    confidence: float
    title: str | None = None
    abstract: str | None = None
    doi: str | None = None
    pmid: str | None = None
    url: str | None = None
    year: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> str:
        if self.doi:
            return f"doi:{self.doi.strip().lower()}"
        if self.pmid:
            return f"pmid:{self.pmid.strip().lower()}"
        if self.title:
            return f"title:{' '.join(self.title.lower().split())}"
        return f"url:{(self.url or '').strip().lower()}"


@dataclass(frozen=True)
class FederatedEvidencePack:
    question: str
    items: list[EvidenceItem]
    provider_stats: dict[str, dict[str, int | float]]

