from __future__ import annotations

import os
import re
from dataclasses import dataclass

from sparkit.evidence.schema import FederatedEvidencePack

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

    llm_decision = _decide_with_llm(question, choices, pack)
    if llm_decision is not None:
        return llm_decision

    fallback = _decide_with_overlap_heuristic(stem, choices, corpus)
    return McqDecision(
        answer_letter=fallback.answer_letter,
        confidence=fallback.confidence,
        rationale=f"heuristic_fallback:{fallback.rationale}",
    )


def _decide_with_llm(question: str, choices: dict[str, str], pack: FederatedEvidencePack) -> McqDecision | None:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("SPARKIT_SYNTH_MODEL", "gpt-5.2").strip() or "gpt-5.2"
    evidence_limit = _env_int("SPARKIT_SYNTH_EVIDENCE_LIMIT", 12, minimum=1, maximum=30)
    per_item_chars = _env_int("SPARKIT_SYNTH_EVIDENCE_CHARS", 900, minimum=200, maximum=4000)

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
            temperature=0,
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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
