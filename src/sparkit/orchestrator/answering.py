from __future__ import annotations

import re
from dataclasses import dataclass

from sparkit.evidence.schema import FederatedEvidencePack

_CHOICE_RE = re.compile(r"^\s*([A-Z])\.\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class McqDecision:
    answer_letter: str | None
    confidence: float
    rationale: str


def parse_mcq(question: str) -> tuple[str, dict[str, str]]:
    marker = "Answer Choices:"
    if marker not in question:
        return question.strip(), {}
    stem, rest = question.split(marker, 1)
    choices: dict[str, str] = {}
    for label, text in _CHOICE_RE.findall(rest):
        choices[label] = " ".join(text.split())
    return stem.strip(), choices


def decide_mcq_from_evidence(question: str, pack: FederatedEvidencePack) -> McqDecision:
    stem, choices = parse_mcq(question)
    if not choices:
        return McqDecision(answer_letter=None, confidence=0.0, rationale="not_multiple_choice")

    corpus = " ".join(
        " ".join(
            part
            for part in (
                item.claim,
                item.title or "",
                item.abstract or "",
            )
            if part
        )
        for item in pack.items
    ).lower()
    if not corpus.strip():
        return McqDecision(answer_letter=None, confidence=0.0, rationale="no_evidence")

    stem_tokens = _tokens(stem)
    scores: dict[str, float] = {}
    for label, text in choices.items():
        choice_tokens = _tokens(text)
        overlap_hits = sum(1 for token in choice_tokens if token in corpus)
        stem_overlap = sum(1 for token in choice_tokens if token in stem_tokens)
        score = (1.6 * overlap_hits) + (0.4 * stem_overlap)
        scores[label] = score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = max(0.0, top_score - runner)
    conf = max(0.05, min(0.95, 0.35 + (margin / max(1.0, top_score + 1.0))))
    return McqDecision(answer_letter=top_label, confidence=conf, rationale=f"top_score={top_score:.2f},margin={margin:.2f}")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))

