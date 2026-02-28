from __future__ import annotations

import os
import re
from dataclasses import dataclass

from sparkit.evidence.schema import EvidenceItem, FederatedEvidencePack, StudyType

try:
    from openai import OpenAI
except Exception:  # noqa: BLE001
    OpenAI = None  # type: ignore

_CHOICE_RE = re.compile(r"^\s*([A-Z])[.)]\s*(.+?)\s*$", re.MULTILINE)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)
_CONF_TAG_RE = re.compile(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", re.IGNORECASE)


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

    model = os.getenv("SPARKIT_SYNTH_MODEL", "gpt-5.2").strip() or "gpt-5.2"
    evidence_limit = _env_int("SPARKIT_SYNTH_EVIDENCE_LIMIT", 12, minimum=1, maximum=30)
    per_item_chars = _env_int("SPARKIT_SYNTH_EVIDENCE_CHARS", 900, minimum=200, maximum=4000)
    llm_decision = _decide_with_llm(
        question,
        choices,
        pack,
        model=model,
        evidence_limit=evidence_limit,
        per_item_chars=per_item_chars,
    )
    if llm_decision is not None:
        return llm_decision

    fallback = _decide_with_overlap_heuristic(stem, choices, corpus)
    return McqDecision(
        answer_letter=fallback.answer_letter,
        confidence=fallback.confidence,
        rationale=f"heuristic_fallback:{fallback.rationale}",
    )


def decide_mcq_fast_consensus(question: str, pack: FederatedEvidencePack) -> McqDecision:
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

    votes: list[tuple[str, str]] = []

    heuristic = _decide_with_overlap_heuristic(stem, choices, corpus)
    if heuristic.answer_letter:
        votes.append(("heuristic", heuristic.answer_letter))

    citation_vote = _decide_with_citation_weighted_vote(choices, pack.items)
    if citation_vote.answer_letter:
        votes.append(("citation_weighted", citation_vote.answer_letter))

    fast_model = os.getenv("SPARKIT_FAST_JUDGE_MODEL", "gpt-5-nano").strip() or "gpt-5-nano"
    fast_evidence_limit = _env_int("SPARKIT_FAST_EVIDENCE_LIMIT", 4, minimum=1, maximum=10)
    fast_per_item_chars = _env_int("SPARKIT_FAST_EVIDENCE_CHARS", 320, minimum=120, maximum=1200)
    fast_llm = _decide_with_llm(
        question,
        choices,
        pack,
        model=fast_model,
        evidence_limit=fast_evidence_limit,
        per_item_chars=fast_per_item_chars,
    )
    if fast_llm is not None and fast_llm.answer_letter:
        votes.append(("fast_llm", fast_llm.answer_letter))

    winner, count = _majority_vote(votes)
    if winner is not None and count >= 2:
        conf = 0.82 if count >= 3 else 0.72
        return McqDecision(
            answer_letter=winner,
            confidence=conf,
            rationale=f"fast_consensus:agree={count}/3,votes={','.join(f'{name}:{letter}' for name, letter in votes)}",
        )

    full = decide_mcq_from_evidence(question, pack)
    return McqDecision(
        answer_letter=full.answer_letter,
        confidence=full.confidence,
        rationale=f"fast_consensus_fallback:{full.rationale}",
    )


def decide_mcq_nano_consensus10(question: str, pack: FederatedEvidencePack) -> McqDecision:
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

    model = os.getenv("SPARKIT_NANO_CONSENSUS_MODEL", "gpt-5-nano").strip() or "gpt-5-nano"
    votes_n = _env_int("SPARKIT_NANO_CONSENSUS_VOTES", 10, minimum=3, maximum=20)
    evidence_limit = _env_int("SPARKIT_NANO_CONSENSUS_EVIDENCE_LIMIT", 5, minimum=1, maximum=12)
    per_item_chars = _env_int("SPARKIT_NANO_CONSENSUS_EVIDENCE_CHARS", 320, minimum=120, maximum=1200)
    temperature = _env_float("SPARKIT_NANO_CONSENSUS_TEMP", 0.7, minimum=0.0, maximum=1.5)

    votes: list[str] = []
    for _ in range(votes_n):
        decision = _decide_with_llm(
            question,
            choices,
            pack,
            model=model,
            evidence_limit=evidence_limit,
            per_item_chars=per_item_chars,
            temperature=temperature,
        )
        if decision is not None and decision.answer_letter:
            votes.append(decision.answer_letter)

    if not votes:
        fallback = decide_mcq_from_evidence(question, pack)
        return McqDecision(
            answer_letter=fallback.answer_letter,
            confidence=fallback.confidence,
            rationale=f"nano_consensus10_fallback:{fallback.rationale}",
        )

    counts: dict[str, int] = {}
    for letter in votes:
        counts[letter] = counts.get(letter, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    winner, winner_count = ranked[0]
    tie = len(ranked) > 1 and ranked[1][1] == winner_count
    if tie:
        fallback = decide_mcq_from_evidence(question, pack)
        return McqDecision(
            answer_letter=fallback.answer_letter,
            confidence=fallback.confidence,
            rationale=f"nano_consensus10_tie_fallback:{fallback.rationale}",
        )

    conf = _clamp(winner_count / max(1, len(votes)), 0.35, 0.98)
    return McqDecision(
        answer_letter=winner,
        confidence=conf,
        rationale=f"nano_consensus10:model={model},votes={len(votes)},winner={winner},count={winner_count}",
    )


def _decide_with_llm(
    question: str,
    choices: dict[str, str],
    pack: FederatedEvidencePack,
    *,
    model: str,
    evidence_limit: int,
    per_item_chars: int,
    temperature: float = 0.0,
) -> McqDecision | None:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    evidence_lines: list[str] = []
    for idx, item in enumerate(pack.items[:evidence_limit], start=1):
        snippet = _compact_text(item.claim, max_chars=per_item_chars)
        if not snippet:
            continue
        title = _compact_text(item.title or "", max_chars=200)
        prefix = f"[E{idx}] provider={item.provider}"
        if title:
            prefix += f" title={title}"
        if item.url:
            prefix += f" url={item.url}"
        evidence_lines.append(f"{prefix}\n{snippet}")
    if not evidence_lines:
        return None

    prompt = (
        "You are solving a multiple-choice STEM question from retrieved evidence.\n"
        "Choose exactly one answer option from the provided choices.\n"
        "Output format requirements:\n"
        "1) First line: <answer>X</answer> where X is one capital letter.\n"
        "2) Second line: <confidence>Y</confidence> where Y is a number in [0,1].\n"
        "3) Third line: one short justification sentence.\n"
        "Do not output any other tags.\n\n"
        f"QUESTION:\n{question}\n\n"
        "CHOICES:\n"
        + "\n".join(f"{label}: {text}" for label, text in choices.items())
        + "\n\nEVIDENCE:\n"
        + "\n\n".join(evidence_lines)
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer carefully and obey output format exactly."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        content = (response.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001
        return None

    answer_match = _ANSWER_TAG_RE.search(content)
    if not answer_match:
        return None
    letter = answer_match.group(1).upper()
    if letter not in choices:
        return None

    conf_match = _CONF_TAG_RE.search(content)
    confidence = _clamp(float(conf_match.group(1)), 0.0, 1.0) if conf_match else 0.65
    return McqDecision(
        answer_letter=letter,
        confidence=confidence,
        rationale=f"llm_synth:model={model},evidence={len(evidence_lines)}",
    )


def _decide_with_citation_weighted_vote(choices: dict[str, str], items: list[EvidenceItem]) -> McqDecision:
    if not items:
        return McqDecision(answer_letter=None, confidence=0.0, rationale="no_items")

    scores: dict[str, float] = {label: 0.0 for label in choices}
    for item in items:
        text_blob = " ".join(part for part in (item.claim, item.title or "", item.abstract or "") if part).lower()
        if not text_blob.strip():
            continue
        item_weight = _item_quality_weight(item)
        for label, choice_text in choices.items():
            tks = _tokens(choice_text)
            if not tks:
                continue
            overlap = len(tks & _tokens(text_blob)) / max(1, len(tks))
            scores[label] += item_weight * overlap

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return McqDecision(answer_letter=None, confidence=0.0, rationale="no_ranked")
    top_label, top_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = max(0.0, top_score - runner)
    conf = max(0.2, min(0.85, 0.45 + margin))
    return McqDecision(answer_letter=top_label, confidence=conf, rationale=f"citation_top={top_score:.3f},margin={margin:.3f}")


def _item_quality_weight(item: EvidenceItem) -> float:
    study_weight = {
        StudyType.META_ANALYSIS: 1.00,
        StudyType.SYSTEMATIC_REVIEW: 0.95,
        StudyType.RANDOMIZED_TRIAL: 0.90,
        StudyType.OBSERVATIONAL: 0.78,
        StudyType.PRECLINICAL: 0.62,
        StudyType.CASE_REPORT: 0.55,
        StudyType.PREPRINT: 0.40,
        StudyType.UNKNOWN: 0.55,
    }[item.study_type]
    provider_weight = {
        "paperqa2": 1.00,
        "scite": 0.95,
        "exa": 0.90,
        "consensus": 0.92,
        "elicit": 0.92,
    }.get(item.provider, 0.85)
    return max(0.0, min(1.0, item.confidence)) * study_weight * provider_weight


def _majority_vote(votes: list[tuple[str, str]]) -> tuple[str | None, int]:
    if not votes:
        return None, 0
    counts: dict[str, int] = {}
    for _, letter in votes:
        counts[letter] = counts.get(letter, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0], ranked[0][1]


def _decide_with_overlap_heuristic(stem: str, choices: dict[str, str], corpus: str) -> McqDecision:
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


def _compact_text(text: str, *, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
