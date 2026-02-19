from __future__ import annotations

from dataclasses import dataclass

from services.retrieval_service.app.models import LiteratureRecord


CONTRADICTION_MARKERS = [
    "contradict",
    "inconsistent",
    "negative result",
    "fails to",
    "no evidence",
    "null result",
    "mixed evidence",
    "retraction",
]


@dataclass(frozen=True)
class VerificationResult:
    contradiction_flags: int
    penalties: dict[str, float]
    notes: list[str]
    ranked_contradictions: list[dict[str, str | float]]


def _score_contradiction(record: LiteratureRecord) -> float:
    text = f"{record.title} {record.abstract or ''}".lower()
    marker_hits = sum(1 for marker in CONTRADICTION_MARKERS if marker in text)
    recency_boost = 0.2 if (record.year or 0) >= 2022 else 0.0
    abstract_boost = 0.1 if record.abstract else 0.0
    return marker_hits + recency_boost + abstract_boost


def run_verifier(
    claim_ids: list[str],
    adversarial_records: list[LiteratureRecord],
    depth: int = 1,
    top_k: int = 5,
) -> VerificationResult:
    scored: list[tuple[float, LiteratureRecord]] = []
    for record in adversarial_records:
        score = _score_contradiction(record)
        if score > 0:
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    window = max(1, depth) * max(1, top_k)
    selected = scored[:window]
    contradictions = len(selected)

    penalties: dict[str, float] = {}
    notes: list[str] = []
    ranked = [
        {
            "title": record.title,
            "score": round(score, 3),
            "year": float(record.year or 0),
            "source": record.source,
        }
        for score, record in selected[:top_k]
    ]

    if contradictions > 0:
        notes.append(
            f"Potential contradictory evidence found (depth={depth}, screened={len(adversarial_records)}, selected={contradictions})"
        )
        penalty = min(0.35, 0.03 * contradictions)
        for claim_id in claim_ids:
            penalties[claim_id] = penalty

    return VerificationResult(
        contradiction_flags=contradictions,
        penalties=penalties,
        notes=notes,
        ranked_contradictions=ranked,
    )
