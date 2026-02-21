from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from shared.schemas.domain import (
    Answer,
    Citation,
    ClaimConfidence,
    Mode,
    ProviderUsage,
    QualityGates,
    Status,
    TraceStage,
)

from services.ingestion_service.app.parser import fetch_and_parse
from services.orchestrator.app.calibration import CalibrationFeatures, calibrate_answer, features_to_dict
from services.orchestrator.app.calibration_store import CalibrationStore
from services.orchestrator.app.evidence_store import EvidenceStore
from services.orchestrator.app.observability import RunObservability, StageMetric
from services.orchestrator.app.observability_store import ObservabilityStore
from services.orchestrator.app.policy import (
    BudgetState,
    contradiction_depth_from_budget,
    estimate_brave_search_cost,
    estimate_generation_cost,
    estimate_stage_cost,
    should_stop_early,
)
from services.orchestrator.app.providers import build_default_registry, generate_text
from services.orchestrator.app.routing import build_provider_plan
from services.orchestrator.app.verifier import run_verifier
from services.retrieval_service.app.aggregator import search_literature
from services.retrieval_service.app.models import LiteratureRecord


@dataclass
class OrchestrationResult:
    answer: Answer
    citations: list[Citation]
    stages: list[TraceStage]
    quality_gates: QualityGates
    source_errors: dict[str, str]
    provider_usage: list[ProviderUsage]


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "using",
    "used",
    "study",
    "evidence",
    "question",
    "results",
    "analysis",
    "model",
    "models",
}


@dataclass
class ClaimEvidence:
    claim_id: str
    claim_text: str
    title: str
    year: int | None
    section_name: str
    section_text: str


@dataclass(frozen=True)
class EffortProfile:
    name: str
    rounds: list[tuple[str, list[str]]]
    retrieval_min_results: int
    ingestion_target_docs: int
    synthesis_max_tokens: int | None
    contradiction_depth_bonus: int = 0
    synthesis_revision_pass: bool = False


@dataclass(frozen=True)
class ResearchPlan:
    task_type: str
    sub_claims: list[str]
    output_schema: list[str]
    disambiguations: list[str]


@dataclass(frozen=True)
class RetrievalPlan:
    segments: list[str]
    focus_terms: list[str]
    intent_queries: dict[str, list[str]]
    answer_choices: dict[str, str]


def _dedupe_records(records: list[LiteratureRecord]) -> list[LiteratureRecord]:
    deduped: list[LiteratureRecord] = []
    seen: set[str] = set()
    for record in sorted(records, key=lambda x: ((x.year or 0), len(x.title)), reverse=True):
        key = (record.doi or record.url).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _record_relevance_score(question: str, record: LiteratureRecord, boost_terms: list[str] | None = None) -> float:
    query_tokens = set(_tokenize(question))
    if boost_terms:
        query_tokens.update(_tokenize(" ".join(boost_terms)))
    title_tokens = set(_tokenize(record.title))
    abstract_tokens = set(_tokenize(record.abstract or ""))
    overlap_title = len(query_tokens & title_tokens)
    overlap_abstract = len(query_tokens & abstract_tokens)
    recency_bonus = 0.0
    if record.year is not None:
        recency_bonus = max(0.0, min(1.0, (record.year - 2015) / 15.0))
    return 2.0 * overlap_title + 1.0 * overlap_abstract + 0.25 * recency_bonus


def _select_records_for_ingestion(
    question: str,
    records: list[LiteratureRecord],
    target_docs: int,
    boost_terms: list[str] | None = None,
) -> list[LiteratureRecord]:
    if not records:
        return []
    scored = sorted(
        records,
        key=lambda item: (_record_relevance_score(question, item, boost_terms), item.year or 0, len(item.title)),
        reverse=True,
    )

    # First pass: take the strongest record from each source for diversity.
    selected: list[LiteratureRecord] = []
    seen_sources: set[str] = set()
    for record in scored:
        source = record.source.lower()
        if source in seen_sources:
            continue
        selected.append(record)
        seen_sources.add(source)
        if len(selected) >= target_docs:
            return selected

    # Second pass: fill remaining slots by relevance.
    for record in scored:
        if record in selected:
            continue
        selected.append(record)
        if len(selected) >= target_docs:
            break
    return selected


def _abstain_reasons(
    *,
    min_sources: int,
    retrieved_count: int,
    support_coverage: float,
    unsupported_claims: int,
    contradiction_flags: int,
    synthesis_failures: list[str],
) -> list[str]:
    reasons: list[str] = []
    if retrieved_count < max(2, min_sources // 2):
        reasons.append("retrieved_evidence_too_sparse")
    if support_coverage < 0.40:
        reasons.append("citation_coverage_below_threshold")
    if unsupported_claims >= 3:
        reasons.append("unsupported_claims_high")
    if contradiction_flags >= 4 and support_coverage < 0.60:
        reasons.append("high_contradiction_with_weak_support")
    if synthesis_failures:
        reasons.append("synthesis_generation_instability")
    return reasons


def _build_round_queries(question: str) -> list[tuple[str, list[str]]]:
    return [
        ("retrieval_round_1", [question, f"{question} review"]),
        ("retrieval_round_2_gap_fill", [f"{question} limitations", f"{question} benchmark comparison"]),
        ("retrieval_round_3_adversarial", [f"{question} contradictory findings", f"{question} negative results"]),
    ]


def _dedupe_queries(queries: list[str], max_items: int = 8) -> list[str]:
    def _normalize_query_text(query: str, max_terms: int = 18) -> str:
        text = " ".join(query.replace("\n", " ").split())
        # Strip common noisy MCQ scaffolding from retrieval queries.
        text = re.sub(r"(?i)\banswer choices?\b[:\s]*", " ", text)
        text = re.sub(r"(?i)\bchoose (?:one|the best)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" .;:,")
        terms = text.split()
        if len(terms) > max_terms:
            terms = terms[:max_terms]
        compact = " ".join(terms)
        return compact.strip()

    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = _normalize_query_text(query)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
        if len(out) >= max_items:
            break
    return out


def _question_has_answer_choices(question: str) -> bool:
    lowered = question.lower()
    if "answer choices" in lowered:
        return True
    return bool(re.search(r"(?:^|\n)\s*[a-n]\.\s+", lowered))


def _split_question_and_choices(question: str) -> tuple[str, dict[str, str]]:
    lines = [line.rstrip() for line in question.splitlines()]
    stem_lines: list[str] = []
    choices: dict[str, str] = {}
    in_choices = False
    for line in lines:
        if not in_choices and "answer choices" in line.lower():
            in_choices = True
            continue
        match = re.match(r"^\s*([A-N])\.\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if match:
            in_choices = True
            label = match.group(1).upper()
            text = " ".join(match.group(2).split())
            if text:
                choices[label] = text
            continue
        if in_choices:
            # Support wrapped multi-line choice text.
            if choices and line.strip():
                last = sorted(choices.keys())[-1]
                choices[last] = f"{choices[last]} {' '.join(line.split())}".strip()
            continue
        if line.strip():
            stem_lines.append(line.strip())
    stem = " ".join(stem_lines).strip() or question.strip()
    return stem, choices


def _extract_answer_letter(text: str) -> str | None:
    match = re.search(r"<answer>\s*([A-N])\s*</answer>", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # fallback for model responses that ignore tag requirement
    loose = re.search(r"\b([A-N])\b", text, flags=re.IGNORECASE)
    if loose:
        return loose.group(1).upper()
    return None


def _parse_mcq_option_scores(text: str, answer_choices: dict[str, str]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    if not text.strip():
        return scores
    for label in sorted(answer_choices.keys()):
        pattern = (
            rf"(?:^|\n)\s*{re.escape(label)}\s*:\s*"
            rf"support\s*=\s*([01](?:\.\d+)?)\s*[,;]?\s*"
            rf"contradiction\s*=\s*([01](?:\.\d+)?)"
        )
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        support = max(0.0, min(1.0, float(match.group(1))))
        contradiction = max(0.0, min(1.0, float(match.group(2))))
        scores[label] = {
            "support": support,
            "contradiction": contradiction,
            "net": support - contradiction,
        }
    return scores


def _has_discriminative_option_scores(scores: dict[str, dict[str, float]]) -> bool:
    if not scores:
        return False
    nets = [round(item["net"], 6) for item in scores.values()]
    if len(set(nets)) <= 1:
        return False
    return True


def _select_confident_blended_option(
    blended_scores: dict[str, dict[str, float]],
    min_margin: float = 0.06,
    min_top_score: float = 0.02,
) -> str | None:
    if not blended_scores:
        return None
    ranked = sorted(
        blended_scores.items(),
        key=lambda kv: (kv[1]["blended"], kv[1]["net"], kv[1]["lexical"]),
        reverse=True,
    )
    top_label, top_row = ranked[0]
    if top_row["blended"] < min_top_score:
        return None
    if len(ranked) == 1:
        return top_label
    second_row = ranked[1][1]
    if (top_row["blended"] - second_row["blended"]) < min_margin:
        return None
    return top_label


def _normalize_mcq_text(value: str) -> str:
    lowered = value.lower()
    lowered = lowered.replace("fwdh", "fwhm")
    lowered = lowered.replace("radical-radical", "radical radical")
    lowered = lowered.replace("non-radical", "non radical")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _mcq_lexical_option_scores(
    answer_choices: dict[str, str],
    claim_texts: list[str],
    section_summaries: list[dict[str, str]] | None = None,
) -> dict[str, dict[str, float]]:
    corpus = " ".join(claim_texts)
    if section_summaries:
        corpus = f"{corpus} " + " ".join(row.get("summary", "") for row in section_summaries)
    corpus_n = _normalize_mcq_text(corpus)
    scores: dict[str, dict[str, float]] = {}
    for label, choice in sorted(answer_choices.items()):
        choice_n = _normalize_mcq_text(choice)
        terms = [token for token in choice_n.split() if len(token) > 2]
        if not terms:
            scores[label] = {"lexical": 0.0}
            continue
        unique_terms = list(dict.fromkeys(terms))
        hits = sum(1 for term in unique_terms if term in corpus_n)
        phrase_hit = 1.0 if choice_n and choice_n in corpus_n else 0.0
        lexical = min(1.0, (hits / max(1, len(unique_terms))) * 0.7 + phrase_hit * 0.3)
        scores[label] = {"lexical": lexical}
    return scores


def _build_option_evidence_packs(
    stem: str,
    answer_choices: dict[str, str],
    claim_texts: list[str],
    top_k: int = 4,
) -> dict[str, list[str]]:
    stem_tokens = set(_tokenize(stem))
    packs: dict[str, list[str]] = {}
    for label, choice in sorted(answer_choices.items()):
        choice_tokens = set(_tokenize(choice))
        scored: list[tuple[float, str]] = []
        for claim in claim_texts:
            claim_tokens = set(_tokenize(claim))
            overlap_choice = len(choice_tokens & claim_tokens)
            overlap_stem = len(stem_tokens & claim_tokens)
            if overlap_choice == 0 and overlap_stem == 0:
                continue
            score = (2.0 * overlap_choice) + (1.0 * overlap_stem)
            scored.append((score, claim))
        ranked = [text for _, text in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]
        packs[label] = ranked
    return packs


def _clean_segments(segments: list[str], question: str) -> list[str]:
    stem, _ = _split_question_and_choices(question)
    cleaned: list[str] = []
    for segment in segments:
        normalized = " ".join(segment.replace("\n", " ").split()).strip()
        if not normalized:
            continue
        if normalized.lower().startswith("answer choices"):
            continue
        if len(normalized) <= 2 and normalized.upper() in {chr(code) for code in range(ord("A"), ord("N") + 1)}:
            continue
        cleaned.append(normalized)
    if not cleaned:
        return [stem]
    return cleaned[:6]


def _build_option_hypothesis_queries(stem: str, answer_choices: dict[str, str], max_items: int = 8) -> list[str]:
    queries: list[str] = []
    for label, choice in sorted(answer_choices.items()):
        if not choice.strip():
            continue
        queries.append(f"{stem} {choice}")
        queries.append(f"{stem} evidence for {choice}")
    return _dedupe_queries(queries, max_items=max_items)


def _extract_lexical_anchors(question: str, max_items: int = 10) -> list[str]:
    # Preserve high-signal technical strings (e.g., hyphenated chemistry terms).
    hyphen_tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+){1,}", question)
    symbol_tokens = re.findall(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+", question)
    raw_tokens = hyphen_tokens + symbol_tokens
    anchors: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        normalized = token.strip(".,;:()[]{}").lower()
        if normalized in seen:
            continue
        if len(normalized) < 6:
            continue
        if not any(ch.isdigit() for ch in normalized) and "-" not in normalized and "/" not in normalized:
            continue
        if normalized in {"h2o/meoh"}:
            continue
        seen.add(normalized)
        anchors.append(token.strip(".,;:()[]{}"))
        if len(anchors) >= max_items:
            break
    return anchors


def _chunk_text(text: str, max_chars: int = 1200, stride_chars: int = 900) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start += stride_chars
    return chunks


def _chunk_relevance_score(question_tokens: set[str], focus_tokens: set[str], chunk: str) -> float:
    chunk_tokens = set(_tokenize(chunk))
    if not chunk_tokens:
        return 0.0
    question_overlap = len(question_tokens & chunk_tokens)
    focus_overlap = len(focus_tokens & chunk_tokens)
    return (1.4 * question_overlap) + (1.9 * focus_overlap) + min(len(chunk_tokens), 150) / 500.0


def _select_best_section_chunk(question: str, sections: list[tuple[str, str]], focus_terms: list[str]) -> tuple[str, str]:
    if not sections:
        return "abstract", ""
    question_tokens = set(_tokenize(question))
    focus_tokens = set(_tokenize(" ".join(focus_terms)))
    best_heading = sections[0][0]
    best_text = sections[0][1]
    best_score = -1.0
    for heading, text in sections:
        for chunk in _chunk_text(text):
            score = _chunk_relevance_score(question_tokens, focus_tokens, chunk)
            if score > best_score:
                best_score = score
                best_heading = heading
                best_text = chunk
    return best_heading, best_text


def _anchor_coverage(text: str, anchors: list[str]) -> float:
    if not anchors:
        return 1.0
    lowered = text.lower()
    hits = sum(1 for anchor in anchors if anchor.lower() in lowered)
    return hits / len(anchors)


def _infer_task_type(question: str) -> str:
    lowered = question.lower()
    if "answer choices" in lowered:
        return "multiple_choice"
    if "what is the product" in lowered or "compound" in lowered:
        return "mechanism"
    if "how many" in lowered or "calculate" in lowered:
        return "numerical"
    if "compare" in lowered or "difference" in lowered:
        return "comparative"
    return "factual"


def _heuristic_decomposition(question: str) -> ResearchPlan:
    chunks = [piece.strip() for piece in re.split(r"[?.;]", question) if piece.strip()]
    sub_claims = [chunk for chunk in chunks[:4]] or [question]
    task_type = _infer_task_type(question)
    if task_type == "multiple_choice":
        schema = ["final_option", "rationale", "why_not_alternatives", "caveats"]
    elif task_type == "mechanism":
        schema = ["mechanism_steps", "key_intermediates", "final_product", "caveats"]
    else:
        schema = ["key_findings", "evidence_support", "caveats"]
    return ResearchPlan(
        task_type=task_type,
        sub_claims=sub_claims,
        output_schema=schema,
        disambiguations=[f"Clarify scope for: {sub_claims[0]}"] if sub_claims else [],
    )


def _heuristic_retrieval_plan(question: str, research_plan: ResearchPlan | None = None) -> RetrievalPlan:
    stem, answer_choices = _split_question_and_choices(question)
    segments = (research_plan.sub_claims[:4] if research_plan else []) or [
        piece.strip() for piece in re.split(r"[?.;]", stem) if piece.strip()
    ][:4]
    segments = _clean_segments(segments or [stem], question)
    anchors = _extract_lexical_anchors(stem, max_items=8)
    lead = segments[0]
    option_queries = _build_option_hypothesis_queries(stem, answer_choices, max_items=8)
    base_topics = " ".join(segments[:2]).strip() or stem
    intent_queries = {
        "primary": _dedupe_queries([f"{base_topics} primary evidence", f"{lead} mechanistic evidence"], max_items=8),
        "methods": _dedupe_queries([f"{lead} methods", f"{lead} protocol", f"{lead} experimental design"], max_items=6),
        "adversarial": _dedupe_queries(
            [f"{lead} contradictory findings", f"{lead} failed replication", f"{lead} negative results"],
            max_items=6,
        ),
        "reference": _dedupe_queries([f"{lead} systematic review", f"{lead} meta-analysis"], max_items=6),
        "options": option_queries,
    }
    for segment in segments[1:3]:
        intent_queries["primary"] = _dedupe_queries([*intent_queries["primary"], f"{segment} evidence"], max_items=8)
        intent_queries["methods"] = _dedupe_queries([*intent_queries["methods"], f"{segment} methods"], max_items=6)
    if option_queries:
        intent_queries["primary"] = _dedupe_queries([*intent_queries["primary"], *option_queries], max_items=10)
    focus_terms = _dedupe_queries([*anchors, *segments, *answer_choices.values()], max_items=14)
    return RetrievalPlan(
        segments=segments,
        focus_terms=focus_terms,
        intent_queries=intent_queries,
        answer_choices=answer_choices,
    )


def _decompose_question(question: str, planning_provider: str) -> ResearchPlan:
    prompt = (
        "Decompose this STEM question into a compact execution plan.\n"
        "Return plain text using exactly these lines:\n"
        "task_type: ...\n"
        "sub_claims: item1 | item2 | item3\n"
        "output_schema: field1 | field2 | field3\n"
        "disambiguations: item1 | item2\n\n"
        f"Question: {question}"
    )
    result = generate_text(planning_provider, prompt, max_tokens=420)
    if not result.success or not result.text.strip():
        return _heuristic_decomposition(question)

    task_type = _infer_task_type(question)
    sub_claims: list[str] = []
    output_schema: list[str] = []
    disambiguations: list[str] = []

    for raw_line in result.text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_norm = key.strip().lower()
        parts = [item.strip() for item in value.split("|") if item.strip()]
        if key_norm == "task_type" and value.strip():
            task_type = value.strip().lower().replace(" ", "_")
        elif key_norm == "sub_claims":
            sub_claims = parts[:6]
        elif key_norm == "output_schema":
            output_schema = parts[:6]
        elif key_norm == "disambiguations":
            disambiguations = parts[:4]

    fallback = _heuristic_decomposition(question)
    return ResearchPlan(
        task_type=task_type or fallback.task_type,
        sub_claims=sub_claims or fallback.sub_claims,
        output_schema=output_schema or fallback.output_schema,
        disambiguations=disambiguations or fallback.disambiguations,
    )


def _decompose_retrieval(question: str, planning_provider: str, research_plan: ResearchPlan | None = None) -> RetrievalPlan:
    stem, answer_choices = _split_question_and_choices(question)
    choices_line = " | ".join(f"{label}: {text}" for label, text in answer_choices.items())
    prompt = (
        "Create a STEM literature retrieval plan for this question.\n"
        "Return plain text with exactly these keys (one line each):\n"
        "segments: item1 | item2 | item3\n"
        "focus_terms: term1 | term2 | term3\n"
        "queries_primary: query1 | query2 | query3\n"
        "queries_options: query1 | query2 | query3\n"
        "queries_methods: query1 | query2 | query3\n"
        "queries_adversarial: query1 | query2\n"
        "queries_reference: query1 | query2\n\n"
        "Rules: queries must be specific, technical, and suitable for scholarly search APIs.\n"
        "Keep each query concise (<= 16 terms), avoid copying the full question verbatim.\n"
        "If answer choices are present, use them as discriminative retrieval cues in queries_options.\n"
        "Do not include explanations.\n\n"
        f"Question stem: {stem}\n"
        f"Answer choices: {choices_line or '(none)'}"
    )
    result = generate_text(planning_provider, prompt, max_tokens=420)
    if not result.success or not result.text.strip():
        return _heuristic_retrieval_plan(question, research_plan)

    segments: list[str] = []
    focus_terms: list[str] = []
    intent_queries: dict[str, list[str]] = {
        "primary": [],
        "options": [],
        "methods": [],
        "adversarial": [],
        "reference": [],
    }
    key_map = {
        "queries_primary": "primary",
        "queries_options": "options",
        "queries_methods": "methods",
        "queries_adversarial": "adversarial",
        "queries_reference": "reference",
    }
    for raw_line in result.text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_norm = key.strip().lower()
        items = [item.strip() for item in value.split("|") if item.strip()]
        if key_norm == "segments":
            segments = _clean_segments(items[:6], question)
        elif key_norm == "focus_terms":
            focus_terms = items[:10]
        elif key_norm in key_map:
            intent_queries[key_map[key_norm]] = _dedupe_queries(items, max_items=8)

    fallback = _heuristic_retrieval_plan(question, research_plan)
    merged_intents: dict[str, list[str]] = {}
    for intent in ("primary", "options", "methods", "adversarial", "reference"):
        merged_intents[intent] = intent_queries.get(intent) or fallback.intent_queries[intent]
    if merged_intents.get("options"):
        merged_intents["primary"] = _dedupe_queries(
            [*merged_intents["primary"], *merged_intents["options"]],
            max_items=10,
        )
    elif answer_choices:
        merged_intents["options"] = _build_option_hypothesis_queries(stem, answer_choices, max_items=8)
        merged_intents["primary"] = _dedupe_queries(
            [*merged_intents["primary"], *merged_intents["options"]],
            max_items=10,
        )

    merged_focus_terms = _dedupe_queries([*focus_terms, *fallback.focus_terms, *answer_choices.values()], max_items=14)
    return RetrievalPlan(
        segments=(segments or fallback.segments),
        focus_terms=merged_focus_terms,
        intent_queries=merged_intents,
        answer_choices=answer_choices or fallback.answer_choices,
    )


def _build_round_queries_from_plan(mode: str, question: str, plan: RetrievalPlan) -> list[tuple[str, list[str]]]:
    primary_queries = _dedupe_queries([*plan.intent_queries["primary"], *plan.intent_queries.get("options", [])], max_items=10)
    if mode == Mode.RESEARCH_MAX.value:
        return [
            ("retrieval_primary", primary_queries),
            ("retrieval_option_hypotheses", plan.intent_queries.get("options", [])),
            ("retrieval_methods", plan.intent_queries["methods"]),
            ("retrieval_adversarial", plan.intent_queries["adversarial"]),
            ("retrieval_reference", plan.intent_queries["reference"]),
        ]
    option_queries = _dedupe_queries(plan.intent_queries.get("options", []), max_items=8)
    return [
        ("retrieval_round_1", primary_queries),
        ("retrieval_round_option_hypotheses", option_queries),
        ("retrieval_round_2_gap_fill", _dedupe_queries([*plan.intent_queries["methods"], *plan.intent_queries["reference"]], max_items=8)),
        ("retrieval_round_3_adversarial", plan.intent_queries["adversarial"]),
    ]


def _record_identity(record: LiteratureRecord) -> str:
    if record.doi:
        return f"doi:{record.doi.lower()}"
    return f"url:{record.url.lower()}"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if not raw:
        return max(default, minimum)
    try:
        return max(int(raw), minimum)
    except ValueError:
        return max(default, minimum)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if not raw:
        return max(default, minimum)
    try:
        return max(float(raw), minimum)
    except ValueError:
        return max(default, minimum)


def _avg_relevance(question: str, records: list[LiteratureRecord], boost_terms: list[str] | None = None) -> float:
    if not records:
        return 0.0
    scores = [_record_relevance_score(question, item, boost_terms) for item in records]
    return sum(scores) / max(1, len(scores))


def _build_evidence_graph(claim_evidence: list[ClaimEvidence], verifier_titles: list[str]) -> dict[str, object]:
    doc_nodes = []
    claim_nodes = []
    edges = []
    seen_docs: set[str] = set()
    for item in claim_evidence:
        doc_key = f"{item.title}:{item.year or 'n.d.'}"
        if doc_key not in seen_docs:
            seen_docs.add(doc_key)
            doc_nodes.append({"id": doc_key, "type": "document", "title": item.title, "year": item.year})
        claim_nodes.append({"id": item.claim_id, "type": "claim", "text": item.claim_text})
        edges.append({"from": item.claim_id, "to": doc_key, "relation": "supports"})

    for title in verifier_titles:
        edges.append({"from": "question", "to": title, "relation": "contradicts"})

    return {
        "nodes": {"documents": doc_nodes, "claims": claim_nodes},
        "edges": edges,
        "summary": {
            "documents": len(doc_nodes),
            "claims": len(claim_nodes),
            "supports_edges": sum(1 for edge in edges if edge["relation"] == "supports"),
            "contradiction_edges": sum(1 for edge in edges if edge["relation"] == "contradicts"),
        },
    }


def _research_finalizer(question: str, draft: str, task_type: str) -> str:
    if task_type == "multiple_choice":
        match = re.search(r"\b([A-N])\b", draft)
        option = match.group(1) if match else "Unknown"
        return (
            f"Final option: {option}\n"
            f"Rationale: {draft}\n"
            "Why not alternatives: Primary alternatives were evaluated against contradiction and methods evidence."
        )
    if task_type == "mechanism":
        return f"Mechanism answer:\n{draft}"
    return draft


def _effort_profile(mode: str, question: str, min_sources: int) -> EffortProfile:
    def _env_int(name: str, default: int, minimum: int) -> int:
        raw = os.getenv(name)
        if not raw:
            return max(default, minimum)
        try:
            return max(int(raw), minimum)
        except ValueError:
            return max(default, minimum)

    if mode == Mode.RESEARCH_MAX.value:
        retrieval_extra = _env_int("SPARKIT_RETRIEVAL_EXTRA_RESULTS_RESEARCH_MAX", 12, 0)
        retrieval_floor = _env_int("SPARKIT_RETRIEVAL_MIN_RESULTS_FLOOR_RESEARCH_MAX", 18, 6)
        ingestion_extra = _env_int("SPARKIT_INGESTION_EXTRA_DOCS_RESEARCH_MAX", 8, 0)
        ingestion_floor = _env_int("SPARKIT_INGESTION_TARGET_DOCS_FLOOR_RESEARCH_MAX", 14, 3)
        return EffortProfile(
            name="research_max",
            rounds=[],
            retrieval_min_results=max(min_sources + retrieval_extra, retrieval_floor),
            ingestion_target_docs=max(min_sources + ingestion_extra, ingestion_floor),
            synthesis_max_tokens=None,
            contradiction_depth_bonus=2,
            synthesis_revision_pass=False,
        )
    retrieval_extra = _env_int("SPARKIT_RETRIEVAL_EXTRA_RESULTS", 8, 0)
    retrieval_floor = _env_int("SPARKIT_RETRIEVAL_MIN_RESULTS_FLOOR", 14, 6)
    ingestion_extra = _env_int("SPARKIT_INGESTION_EXTRA_DOCS", 6, 0)
    ingestion_floor = _env_int("SPARKIT_INGESTION_TARGET_DOCS_FLOOR", 10, 3)
    return EffortProfile(
        name="standard",
        rounds=_build_round_queries(question),
        retrieval_min_results=max(min_sources + retrieval_extra, retrieval_floor),
        ingestion_target_docs=max(min_sources + ingestion_extra, ingestion_floor),
        synthesis_max_tokens=None,
    )


def _tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", value.lower())
    return [token for token in tokens if token not in _STOPWORDS]


def _section_bucket(section_name: str) -> str:
    lowered = section_name.lower()
    if any(word in lowered for word in ("method", "materials", "protocol", "experiment")):
        return "methods"
    if any(word in lowered for word in ("result", "finding", "evaluation", "benchmark")):
        return "results"
    if any(word in lowered for word in ("discussion", "conclusion", "limitation", "caveat")):
        return "discussion"
    if any(word in lowered for word in ("abstract", "introduction", "background")):
        return "overview"
    return "other"


def _first_sentence(text: str, max_chars: int = 220) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    sentence = parts[0] if parts else cleaned
    return sentence[:max_chars]


def _build_claim_clusters(evidence: list[ClaimEvidence], max_clusters: int = 4) -> list[dict[str, object]]:
    if not evidence:
        return []

    clusters: dict[str, list[ClaimEvidence]] = defaultdict(list)
    for item in evidence:
        topic_tokens = _tokenize(item.title)
        label_tokens = topic_tokens[:2] or _tokenize(item.claim_text)[:2] or ["general"]
        cluster_key = " ".join(label_tokens)
        clusters[cluster_key].append(item)

    ranked = sorted(clusters.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)[:max_clusters]
    cluster_rows: list[dict[str, object]] = []
    for label, items in ranked:
        cluster_rows.append(
            {
                "label": label,
                "count": len(items),
                "sample_claims": [item.claim_text for item in items[:2]],
            }
        )
    return cluster_rows


def _build_section_summaries(evidence: list[ClaimEvidence], max_sections: int = 4) -> list[dict[str, str]]:
    if not evidence:
        return []

    grouped: dict[str, list[str]] = defaultdict(list)
    for item in evidence:
        summary = _first_sentence(item.section_text)
        if not summary:
            continue
        grouped[_section_bucket(item.section_name)].append(summary)

    priority = ["overview", "methods", "results", "discussion", "other"]
    rows: list[dict[str, str]] = []
    for bucket in priority:
        snippets = grouped.get(bucket, [])
        if not snippets:
            continue
        rows.append({"section": bucket, "summary": snippets[0]})
        if len(rows) >= max_sections:
            break
    return rows


def _build_answer_text(
    question: str,
    claims: list[str],
    unsupported_claims: int,
    clusters: list[dict[str, object]] | None = None,
    section_summaries: list[dict[str, str]] | None = None,
) -> str:
    if not claims:
        return (
            f"Insufficient evidence retrieved to answer the question with confidence: {question}. "
            "Try broadening the query or increasing retrieval budget."
        )

    joined = " ".join(claims[:3])
    cluster_hint = ""
    if clusters:
        labels = [str(cluster["label"]) for cluster in clusters[:3]]
        cluster_hint = f" Dominant evidence themes: {', '.join(labels)}."

    section_hint = ""
    if section_summaries:
        bullets = [f"{row['section']}: {row['summary']}" for row in section_summaries[:2]]
        section_hint = f" Section highlights: {' | '.join(bullets)}."

    suffix = " Some claims remain weakly supported." if unsupported_claims > 0 else ""
    return f"Answer synthesis for '{question}': {joined}.{cluster_hint}{section_hint}{suffix}"


def _build_synthesis_prompt(
    question: str,
    claim_texts: list[str],
    claim_clusters: list[dict[str, object]] | None = None,
    section_summaries: list[dict[str, str]] | None = None,
) -> str:
    evidence_lines = "\n".join(f"- {claim}" for claim in claim_texts[:8]) or "- No evidence lines available."
    cluster_lines = "\n".join(
        f"- {cluster['label']} (n={cluster['count']}): {'; '.join(cluster['sample_claims'])}"
        for cluster in (claim_clusters or [])[:4]
    ) or "- No claim clusters available."
    section_lines = "\n".join(
        f"- {row['section']}: {row['summary']}" for row in (section_summaries or [])[:4]
    ) or "- No section summaries available."
    if _question_has_answer_choices(question):
        return (
            "You are a scientific QA synthesizer. Use only the provided evidence bullets.\n"
            f"Question: {question}\n"
            "Claim clusters:\n"
            f"{cluster_lines}\n"
            "Section-aware summaries:\n"
            f"{section_lines}\n"
            "Evidence:\n"
            f"{evidence_lines}\n"
            "Return ONLY one XML tag with the final multiple-choice letter.\n"
            "Format exactly: <answer>X</answer>\n"
            "No additional text."
        )

    return (
        "You are a scientific QA synthesizer. Use only the provided evidence bullets.\n"
        f"Question: {question}\n"
        "Claim clusters:\n"
        f"{cluster_lines}\n"
        "Section-aware summaries:\n"
        f"{section_lines}\n"
        "Evidence:\n"
        f"{evidence_lines}\n"
        "Return a concise technical answer with: key findings, methods/results caveats, and confidence caveats."
    )


def _build_mcq_option_judge_prompt(
    question: str,
    answer_choices: dict[str, str],
    claim_texts: list[str],
    claim_clusters: list[dict[str, object]] | None = None,
    section_summaries: list[dict[str, str]] | None = None,
    option_evidence_packs: dict[str, list[str]] | None = None,
) -> str:
    stem, _ = _split_question_and_choices(question)
    choices_block = "\n".join(f"{label}. {text}" for label, text in sorted(answer_choices.items()))
    evidence_lines = "\n".join(f"- {claim}" for claim in claim_texts[:12]) or "- No evidence lines available."
    cluster_lines = "\n".join(
        f"- {cluster['label']} (n={cluster['count']}): {'; '.join(cluster['sample_claims'])}"
        for cluster in (claim_clusters or [])[:4]
    ) or "- No claim clusters available."
    section_lines = "\n".join(
        f"- {row['section']}: {row['summary']}" for row in (section_summaries or [])[:4]
    ) or "- No section summaries available."
    option_pack_lines: list[str] = []
    for label, text in sorted(answer_choices.items()):
        pack = (option_evidence_packs or {}).get(label, [])
        evidence = " | ".join(pack[:3]) if pack else "no focused snippets"
        option_pack_lines.append(f"- {label}. {text} -> {evidence}")
    option_pack_block = "\n".join(option_pack_lines)
    return (
        "You are a rigorous STEM MCQ adjudicator.\n"
        "Use only the provided evidence to choose the best option.\n"
        "If evidence is weak, still pick the most supported option.\n"
        "Return ONLY one XML tag.\n"
        "Format exactly: <answer>X</answer>\n\n"
        f"Question stem: {stem}\n"
        "Answer choices:\n"
        f"{choices_block}\n\n"
        "Claim clusters:\n"
        f"{cluster_lines}\n"
        "Section-aware summaries:\n"
        f"{section_lines}\n"
        "Option-focused evidence snippets:\n"
        f"{option_pack_block}\n"
        "Evidence:\n"
        f"{evidence_lines}\n"
    )


def _build_mcq_option_scoring_prompt(
    question: str,
    answer_choices: dict[str, str],
    claim_texts: list[str],
    claim_clusters: list[dict[str, object]] | None = None,
    section_summaries: list[dict[str, str]] | None = None,
    option_evidence_packs: dict[str, list[str]] | None = None,
) -> str:
    stem, _ = _split_question_and_choices(question)
    choices_block = "\n".join(f"{label}. {text}" for label, text in sorted(answer_choices.items()))
    evidence_lines = "\n".join(f"- {claim}" for claim in claim_texts[:12]) or "- No evidence lines available."
    cluster_lines = "\n".join(
        f"- {cluster['label']} (n={cluster['count']}): {'; '.join(cluster['sample_claims'])}"
        for cluster in (claim_clusters or [])[:4]
    ) or "- No claim clusters available."
    section_lines = "\n".join(
        f"- {row['section']}: {row['summary']}" for row in (section_summaries or [])[:4]
    ) or "- No section summaries available."
    option_pack_lines: list[str] = []
    for label, text in sorted(answer_choices.items()):
        pack = (option_evidence_packs or {}).get(label, [])
        evidence = " | ".join(pack[:3]) if pack else "no focused snippets"
        option_pack_lines.append(f"- {label}. {text} -> {evidence}")
    option_pack_block = "\n".join(option_pack_lines)
    return (
        "You are a strict STEM MCQ evidence scorer.\n"
        "For each choice, score how well evidence supports it and contradicts it.\n"
        "Use only the evidence below.\n"
        "Output exactly one line per choice in this format:\n"
        "A: support=0.00, contradiction=0.00\n"
        "B: support=0.00, contradiction=0.00\n"
        "...\n"
        "Do not include extra text.\n\n"
        f"Question stem: {stem}\n"
        "Answer choices:\n"
        f"{choices_block}\n\n"
        "Claim clusters:\n"
        f"{cluster_lines}\n"
        "Section-aware summaries:\n"
        f"{section_lines}\n"
        "Option-focused evidence snippets:\n"
        f"{option_pack_block}\n"
        "Evidence:\n"
        f"{evidence_lines}\n"
    )


def _ensemble_agreement(drafts: list[str]) -> float:
    if len(drafts) <= 1:
        return 1.0
    token_sets = [set(draft.lower().split()) for draft in drafts]
    intersection = set.intersection(*token_sets)
    union = set.union(*token_sets)
    return len(intersection) / max(1, len(union))


def execute_orchestration(
    run_id: str,
    question: str,
    min_sources: int = 5,
    providers: list[str] | None = None,
    mode: str = Mode.SINGLE.value,
    max_latency_s: int | None = None,
    max_cost_usd: float = 3.0,
    synthesis_max_tokens: int | None = None,
    prompt_version: str = 'synthesis_v1.2',
    config_version: str = 'orchestration_v1.2',
    reproducibility: dict | None = None,
) -> OrchestrationResult:
    started = datetime.now(timezone.utc)
    ingestion_max_chars = int(os.getenv("SPARKIT_INGESTION_MAX_CHARS", "10000"))
    observability = RunObservability(run_id=run_id)

    provider_list = providers or ["openai"]
    provider_statuses = build_default_registry().resolve(provider_list)
    missing_keys = [status for status in provider_statuses if not status.configured]
    provider_plan = build_provider_plan(mode=mode, statuses=provider_statuses, requested=provider_list)
    effort = _effort_profile(mode=mode, question=question, min_sources=min_sources)
    synthesis_token_budget = synthesis_max_tokens if synthesis_max_tokens is not None else effort.synthesis_max_tokens
    research_plan: ResearchPlan | None = None
    retrieval_plan: RetrievalPlan | None = None
    if mode == Mode.RESEARCH_MAX.value:
        research_plan = _decompose_question(question, provider_plan.planning)
    retrieval_plan = _decompose_retrieval(question, provider_plan.planning, research_plan)
    rounds = _build_round_queries_from_plan(mode, question, retrieval_plan)
    records_by_round: dict[str, list[LiteratureRecord]] = {}
    all_records: list[LiteratureRecord] = []
    aggregate_errors: dict[str, str] = {}
    stages: list[TraceStage] = [
        TraceStage(
            name="plan",
            status=Status.COMPLETED,
            model=provider_plan.planning,
            started_at=started,
            ended_at=started,
            artifacts={
                "strategy": f"{effort.name}-effort retrieval + verification + calibration + policy",
                "question": question,
                "mode": mode,
                "provider_plan": {
                    "planning": provider_plan.planning,
                    "retrieval": provider_plan.retrieval,
                    "synthesis": provider_plan.synthesis,
                    "verification": provider_plan.verification,
                    "ensemble": provider_plan.ensemble,
                },
                "providers": [
                    {"provider": status.provider, "configured": status.configured, "env_var": status.env_var}
                    for status in provider_statuses
                ],
                "budget": {"max_latency_s": max_latency_s, "max_cost_usd": max_cost_usd},
                "effort_profile": {
                    "name": effort.name,
                    "rounds": len(rounds),
                    "retrieval_min_results": effort.retrieval_min_results,
                    "ingestion_target_docs": effort.ingestion_target_docs,
                    "ingestion_max_chars": ingestion_max_chars,
                    "synthesis_max_tokens": synthesis_token_budget,
                    "contradiction_depth_bonus": effort.contradiction_depth_bonus,
                    "synthesis_revision_pass": effort.synthesis_revision_pass,
                    "adaptive_retrieval": {
                        "enabled": _env_bool("SPARKIT_ADAPTIVE_RETRIEVAL", True),
                        "min_rounds": _env_int("SPARKIT_ADAPTIVE_MIN_ROUNDS", 2, minimum=1),
                        "max_rounds": _env_int("SPARKIT_ADAPTIVE_MAX_ROUNDS", len(rounds), minimum=1),
                        "min_new_docs": _env_int("SPARKIT_ADAPTIVE_MIN_NEW_DOCS", 2, minimum=0),
                        "min_quality_gain": _env_float("SPARKIT_ADAPTIVE_MIN_QUALITY_GAIN", 0.03, minimum=0.0),
                    },
                },
                "prompt_version": prompt_version,
                "config_version": config_version,
                "reproducibility": reproducibility or {},
                "research_plan": None if research_plan is None else {
                    "task_type": research_plan.task_type,
                    "sub_claims": research_plan.sub_claims,
                    "output_schema": research_plan.output_schema,
                    "disambiguations": research_plan.disambiguations,
                },
                "retrieval_plan": {
                    "segments": retrieval_plan.segments if retrieval_plan else [],
                    "focus_terms": retrieval_plan.focus_terms if retrieval_plan else [],
                    "intents": retrieval_plan.intent_queries if retrieval_plan else {},
                    "answer_choices": retrieval_plan.answer_choices if retrieval_plan else {},
                },
            },
        )
    ]

    if research_plan is not None:
        stages.append(
            TraceStage(
                name="question_decomposition",
                status=Status.COMPLETED,
                model=provider_plan.planning,
                started_at=started,
                ended_at=datetime.now(timezone.utc),
                artifacts={
                    "task_type": research_plan.task_type,
                    "sub_claims": research_plan.sub_claims,
                    "output_schema": research_plan.output_schema,
                    "disambiguations": research_plan.disambiguations,
                },
            )
        )
    if retrieval_plan is not None:
        stages.append(
            TraceStage(
                name="retrieval_planner",
                status=Status.COMPLETED,
                model=provider_plan.planning,
                started_at=started,
                ended_at=datetime.now(timezone.utc),
                artifacts={
                    "segments": retrieval_plan.segments,
                    "focus_terms": retrieval_plan.focus_terms,
                    "intents": retrieval_plan.intent_queries,
                    "answer_choices": retrieval_plan.answer_choices,
                },
            )
        )

    budget_stop_reason: str | None = None
    spent_usd = 0.0
    retrieval_base_cost_usd = 0.0
    retrieval_brave_cost_usd = 0.0
    brave_request_count = 0
    adaptive_retrieval = _env_bool("SPARKIT_ADAPTIVE_RETRIEVAL", True)
    adaptive_min_rounds = _env_int("SPARKIT_ADAPTIVE_MIN_ROUNDS", 2, minimum=1)
    adaptive_max_rounds = _env_int("SPARKIT_ADAPTIVE_MAX_ROUNDS", len(rounds), minimum=1)
    adaptive_min_new_docs = _env_int("SPARKIT_ADAPTIVE_MIN_NEW_DOCS", 2, minimum=0)
    adaptive_min_quality_gain = _env_float("SPARKIT_ADAPTIVE_MIN_QUALITY_GAIN", 0.03, minimum=0.0)
    seen_record_ids: set[str] = set()
    prev_selected_quality = 0.0

    for stage_idx, (stage_name, queries) in enumerate(rounds, start=1):
        elapsed_s = (datetime.now(timezone.utc) - started).total_seconds()
        reserve = estimate_stage_cost("retrieval", units=len(queries))
        if should_stop_early(BudgetState(elapsed_s=elapsed_s, spent_usd=spent_usd), max_latency_s, max_cost_usd, reserve):
            budget_stop_reason = f"budget stop before {stage_name}"
            stages.append(
                TraceStage(
                    name="budget_guard",
                    status=Status.COMPLETED,
                    model="policy",
                    started_at=datetime.now(timezone.utc),
                    ended_at=datetime.now(timezone.utc),
                    artifacts={"reason": budget_stop_reason, "elapsed_s": elapsed_s, "spent_usd": spent_usd},
                )
            )
            break

        stage_start = datetime.now(timezone.utc)
        stage_records: list[LiteratureRecord] = []
        stage_errors: dict[str, str] = {}
        stage_brave_requests = 0

        for query in queries:
            found, errors, stats = search_literature(query, max_results=effort.retrieval_min_results)
            stage_records.extend(found)
            for source, err in errors.items():
                stage_errors[f"{source}:{query}"] = err
            stage_brave_requests += int((stats.get("requests_by_source") or {}).get("brave_web", 0))

        deduped_stage = _dedupe_records(stage_records)
        records_by_round[stage_name] = deduped_stage
        all_records.extend(deduped_stage)
        aggregate_errors.update(stage_errors)
        new_unique_docs = 0
        for record in deduped_stage:
            rid = _record_identity(record)
            if rid in seen_record_ids:
                continue
            seen_record_ids.add(rid)
            new_unique_docs += 1

        stage_base_cost = estimate_stage_cost("retrieval", units=len(queries))
        stage_brave_cost = estimate_brave_search_cost(stage_brave_requests)
        stage_cost = stage_base_cost + stage_brave_cost
        spent_usd += stage_cost
        retrieval_base_cost_usd += stage_base_cost
        retrieval_brave_cost_usd += stage_brave_cost
        brave_request_count += stage_brave_requests
        duration_ms = int((datetime.now(timezone.utc) - stage_start).total_seconds() * 1000)
        observability.add_stage(
            StageMetric(
                name=stage_name,
                duration_ms=duration_ms,
                documents_retrieved=len(deduped_stage),
                source_errors=len(stage_errors),
                estimated_cost_usd=stage_cost,
            )
        )

        stages.append(
            TraceStage(
                name=stage_name,
                status=Status.COMPLETED,
                model=provider_plan.retrieval,
                started_at=stage_start,
                ended_at=datetime.now(timezone.utc),
                artifacts={
                    "queries": queries,
                    "documents_retrieved": len(deduped_stage),
                    "new_unique_docs": new_unique_docs,
                    "source_errors": stage_errors,
                    "brave_requests": stage_brave_requests,
                    "brave_cost_usd": stage_brave_cost,
                    "estimated_cost_usd": stage_cost,
                },
            )
        )
        if adaptive_retrieval:
            if stage_idx >= adaptive_max_rounds:
                budget_stop_reason = budget_stop_reason or f"adaptive stop after {stage_name}: max rounds reached"
                stages.append(
                    TraceStage(
                        name="retrieval_adaptive_gate",
                        status=Status.COMPLETED,
                        model="policy",
                        started_at=datetime.now(timezone.utc),
                        ended_at=datetime.now(timezone.utc),
                        artifacts={
                            "reason": "max_rounds_reached",
                            "stage": stage_name,
                            "stage_idx": stage_idx,
                            "max_rounds": adaptive_max_rounds,
                        },
                    )
                )
                break
            selected_now = _select_records_for_ingestion(
                question=question,
                records=_dedupe_records(all_records),
                target_docs=effort.ingestion_target_docs,
                boost_terms=retrieval_plan.focus_terms if retrieval_plan else None,
            )
            selected_quality = _avg_relevance(
                question=question,
                records=selected_now,
                boost_terms=retrieval_plan.focus_terms if retrieval_plan else None,
            )
            quality_gain = selected_quality - prev_selected_quality
            prev_selected_quality = selected_quality
            if stage_idx >= adaptive_min_rounds:
                low_novelty = new_unique_docs < adaptive_min_new_docs
                low_gain = quality_gain < adaptive_min_quality_gain
                if low_novelty and low_gain:
                    budget_stop_reason = budget_stop_reason or f"adaptive stop after {stage_name}: low evidence gain"
                    stages.append(
                        TraceStage(
                            name="retrieval_adaptive_gate",
                            status=Status.COMPLETED,
                            model="policy",
                            started_at=datetime.now(timezone.utc),
                            ended_at=datetime.now(timezone.utc),
                            artifacts={
                                "reason": "low_evidence_gain",
                                "stage": stage_name,
                                "stage_idx": stage_idx,
                                "new_unique_docs": new_unique_docs,
                                "min_new_docs": adaptive_min_new_docs,
                                "selected_quality": selected_quality,
                                "quality_gain": quality_gain,
                                "min_quality_gain": adaptive_min_quality_gain,
                            },
                        )
                    )
                    break

    records = _dedupe_records(all_records)
    selected_records = _select_records_for_ingestion(
        question=question,
        records=records,
        target_docs=effort.ingestion_target_docs,
        boost_terms=retrieval_plan.focus_terms if retrieval_plan else None,
    )
    evidence_store = EvidenceStore()

    citations: list[Citation] = []
    claim_texts: list[str] = []
    claim_evidence: list[ClaimEvidence] = []
    base_claim_conf: dict[str, float] = {}
    unsupported_claims = 0

    for record in selected_records:
        claim_text = f"{record.title} ({record.year or 'n.d.'}) indicates relevant evidence for the question."
        section_name = "abstract"
        section_text = record.abstract or ""
        try:
            ingested = fetch_and_parse(record.url, max_chars=ingestion_max_chars, timeout_s=12.0)
            if ingested.sections:
                parsed_sections = [(section.heading, section.text) for section in ingested.sections if section.text.strip()]
                section_name, section_text = _select_best_section_chunk(
                    question=question,
                    sections=parsed_sections,
                    focus_terms=retrieval_plan.focus_terms if retrieval_plan else [],
                )
        except Exception:  # noqa: BLE001
            pass
        summary = _first_sentence(section_text, max_chars=180)
        if summary:
            claim_text = f"{record.title} ({record.year or 'n.d.'}) reports: {summary}"

        spent_usd += estimate_stage_cost("ingestion")
        evidence = evidence_store.upsert_document_with_passage(record=record, section=section_name, text=section_text)
        claim_id = evidence_store.insert_claim(
            run_id=run_id,
            text=claim_text,
            claim_type="fact",
            support_score=0.8,
            status="supported" if section_text else "weak_support",
        )

        if section_text:
            evidence_store.link_claim_to_passage(claim_id=claim_id, passage_id=evidence.passage_id)
            citations.append(Citation(claim_id=claim_id, doc_id=evidence.doc_id, passage_id=evidence.passage_id))
        else:
            unsupported_claims += 1

        base_claim_conf[claim_id] = 0.78 if section_text else 0.45
        claim_texts.append(claim_text)
        claim_evidence.append(
            ClaimEvidence(
                claim_id=claim_id,
                claim_text=claim_text,
                title=record.title,
                year=record.year,
                section_name=section_name,
                section_text=section_text,
            )
        )

    adversarial_stage_name = "retrieval_adversarial" if mode == Mode.RESEARCH_MAX.value else "retrieval_round_3_adversarial"
    verifier_start = datetime.now(timezone.utc)
    depth = contradiction_depth_from_budget(max_cost_usd=max_cost_usd, max_latency_s=max_latency_s) + effort.contradiction_depth_bonus
    verifier_result = run_verifier(
        claim_ids=list(base_claim_conf.keys()),
        adversarial_records=records_by_round.get(adversarial_stage_name, []),
        depth=depth,
        top_k=5,
    )
    verifier_cost = estimate_stage_cost("verification", units=max(1, depth))
    spent_usd += verifier_cost
    observability.add_stage(
        StageMetric(
            name="verification",
            duration_ms=int((datetime.now(timezone.utc) - verifier_start).total_seconds() * 1000),
            documents_retrieved=len(records_by_round.get(adversarial_stage_name, [])),
            source_errors=0,
            estimated_cost_usd=verifier_cost,
        )
    )
    stages.append(
        TraceStage(
            name="verification",
            status=Status.COMPLETED,
            model=provider_plan.verification,
            started_at=verifier_start,
            ended_at=datetime.now(timezone.utc),
            artifacts={
                "contradiction_flags": verifier_result.contradiction_flags,
                "notes": verifier_result.notes,
                "ranked_contradictions": verifier_result.ranked_contradictions,
                "depth": depth,
            },
        )
    )

    if mode == Mode.RESEARCH_MAX.value:
        evidence_graph = _build_evidence_graph(
            claim_evidence=claim_evidence,
            verifier_titles=[item.get("title", "") for item in verifier_result.ranked_contradictions if item.get("title")],
        )
        stages.append(
            TraceStage(
                name="evidence_graph",
                status=Status.COMPLETED,
                model="graph-builder",
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                artifacts=evidence_graph,
            )
        )

    adjusted_claim_conf: dict[str, float] = {}
    for claim_id, base in base_claim_conf.items():
        penalty = verifier_result.penalties.get(claim_id, 0.0)
        adjusted_claim_conf[claim_id] = max(0.05, min(0.95, base - penalty))

    claim_clusters = _build_claim_clusters(claim_evidence)
    section_summaries = _build_section_summaries(claim_evidence)
    synthesis_prompt = _build_synthesis_prompt(
        question,
        claim_texts,
        claim_clusters=claim_clusters,
        section_summaries=section_summaries,
    )
    stem_question, answer_choices = _split_question_and_choices(question)
    synthesis_start = datetime.now(timezone.utc)
    synthesis_failures: list[str] = []
    draft_texts: list[str] = []
    draft_usage: list[ProviderUsage] = []

    def _record_gen_usage(
        provider: str,
        model: str,
        draft: str,
        tokens_input: int = 0,
        tokens_input_cached: int = 0,
        tokens_output: int = 0,
    ) -> None:
        nonlocal spent_usd
        out_tokens = tokens_output if tokens_output > 0 else max(1, len(draft) // 4)
        input_tokens = tokens_input if tokens_input > 0 else max(1, len(synthesis_prompt) // 4)
        est = estimate_generation_cost(
            provider=provider,
            model=model,
            tokens_input=input_tokens,
            tokens_input_cached=tokens_input_cached,
            tokens_output=out_tokens,
        )
        spent_usd += est
        draft_usage.append(
            ProviderUsage(
                provider=provider,
                model=model,
                tokens_input=max(0, int(input_tokens)),
                tokens_output=max(0, int(out_tokens)),
                cost_usd=est,
            )
        )

    if should_stop_early(
        BudgetState(elapsed_s=(datetime.now(timezone.utc) - started).total_seconds(), spent_usd=spent_usd),
        max_latency_s,
        max_cost_usd,
        reserve_next_stage_usd=estimate_stage_cost("synthesis"),
    ):
        synthesis_failures.append("budget guard: synthesis skipped")
        draft_texts.append(
            _build_answer_text(
                question,
                claim_texts,
                unsupported_claims,
                clusters=claim_clusters,
                section_summaries=section_summaries,
            )
        )
    elif mode in {Mode.SINGLE.value, Mode.ROUTED.value}:
        if _question_has_answer_choices(question) and answer_choices:
            option_evidence_packs = _build_option_evidence_packs(
                stem=stem_question,
                answer_choices=answer_choices,
                claim_texts=claim_texts,
                top_k=4,
            )
            scorer_prompt = _build_mcq_option_scoring_prompt(
                question=stem_question,
                answer_choices=answer_choices,
                claim_texts=claim_texts,
                claim_clusters=claim_clusters,
                section_summaries=section_summaries,
                option_evidence_packs=option_evidence_packs,
            )
            scorer_result = generate_text(
                provider_plan.synthesis,
                scorer_prompt,
                max_tokens=min(420, synthesis_token_budget or 420),
            )
            parsed_scores = _parse_mcq_option_scores(scorer_result.text or "", answer_choices)
            lexical_scores = _mcq_lexical_option_scores(
                answer_choices=answer_choices,
                claim_texts=claim_texts,
                section_summaries=section_summaries,
            )
            blended_scores: dict[str, dict[str, float]] = {}
            for label in sorted(answer_choices.keys()):
                llm = parsed_scores.get(label, {"support": 0.0, "contradiction": 0.0, "net": 0.0})
                lex = lexical_scores.get(label, {"lexical": 0.0})
                blended = 0.7 * float(llm["net"]) + 0.3 * float(lex["lexical"])
                blended_scores[label] = {
                    "support": float(llm["support"]),
                    "contradiction": float(llm["contradiction"]),
                    "net": float(llm["net"]),
                    "lexical": float(lex["lexical"]),
                    "blended": blended,
                }
            selected_letter: str | None = None
            if blended_scores and len({round(row["blended"], 6) for row in blended_scores.values()}) > 1:
                min_margin = float(os.getenv("SPARKIT_MCQ_BLEND_MARGIN", "0.06"))
                min_top_score = float(os.getenv("SPARKIT_MCQ_BLEND_MIN_TOP", "0.02"))
                selected_letter = _select_confident_blended_option(
                    blended_scores=blended_scores,
                    min_margin=min_margin,
                    min_top_score=min_top_score,
                )

            if scorer_result.success and selected_letter and selected_letter in answer_choices:
                draft = f"<answer>{selected_letter}</answer>"
                draft_texts.append(draft)
                _record_gen_usage(
                    provider_plan.synthesis,
                    scorer_result.model,
                    draft,
                    tokens_input=scorer_result.tokens_input,
                    tokens_input_cached=scorer_result.tokens_input_cached,
                    tokens_output=scorer_result.tokens_output,
                )
                stages.append(
                    TraceStage(
                        name="mcq_option_scorer",
                        status=Status.COMPLETED,
                        model=provider_plan.synthesis,
                        started_at=datetime.now(timezone.utc),
                        ended_at=datetime.now(timezone.utc),
                        artifacts={
                            "selected_option": selected_letter,
                            "num_choices": len(answer_choices),
                            "option_scores": parsed_scores,
                            "lexical_scores": lexical_scores,
                            "blended_scores": blended_scores,
                            "option_evidence_packs": option_evidence_packs,
                            "scorer_raw_output": (scorer_result.text or "")[:2000],
                        },
                    )
                )
            else:
                synthesis_failures.append(
                    f"{provider_plan.synthesis}: mcq_scorer_failed ({scorer_result.error or 'invalid_or_low_margin_output'})"
                )
                judge_prompt = _build_mcq_option_judge_prompt(
                    question=stem_question,
                    answer_choices=answer_choices,
                    claim_texts=claim_texts,
                    claim_clusters=claim_clusters,
                    section_summaries=section_summaries,
                    option_evidence_packs=option_evidence_packs,
                )
                judge_result = generate_text(
                    provider_plan.synthesis, judge_prompt, max_tokens=min(220, synthesis_token_budget or 220)
                )
                selected_letter = _extract_answer_letter(judge_result.text or "")
                if judge_result.success and selected_letter and selected_letter in answer_choices:
                    draft = f"<answer>{selected_letter}</answer>"
                    draft_texts.append(draft)
                    _record_gen_usage(
                        provider_plan.synthesis,
                        judge_result.model,
                        draft,
                        tokens_input=judge_result.tokens_input,
                        tokens_input_cached=judge_result.tokens_input_cached,
                        tokens_output=judge_result.tokens_output,
                    )
                    stages.append(
                        TraceStage(
                            name="mcq_option_judge",
                            status=Status.COMPLETED,
                            model=provider_plan.synthesis,
                            started_at=datetime.now(timezone.utc),
                            ended_at=datetime.now(timezone.utc),
                            artifacts={"selected_option": selected_letter, "num_choices": len(answer_choices)},
                        )
                    )
                else:
                    synthesis_failures.append(
                        f"{provider_plan.synthesis}: mcq_judge_failed ({judge_result.error or 'invalid output'})"
                    )
                    synth_result = generate_text(provider_plan.synthesis, synthesis_prompt, max_tokens=synthesis_token_budget)
                    if synth_result.success and synth_result.text.strip():
                        draft = synth_result.text.strip()
                        draft_texts.append(draft)
                        _record_gen_usage(
                            provider_plan.synthesis,
                            synth_result.model,
                            draft,
                            tokens_input=synth_result.tokens_input,
                            tokens_input_cached=synth_result.tokens_input_cached,
                            tokens_output=synth_result.tokens_output,
                        )
                    else:
                        synthesis_failures.append(f"{provider_plan.synthesis}: {synth_result.error or 'empty output'}")
                        fallback = _build_answer_text(
                            question,
                            claim_texts,
                            unsupported_claims,
                            clusters=claim_clusters,
                            section_summaries=section_summaries,
                        )
                        draft_texts.append(fallback)
                        _record_gen_usage(
                            provider_plan.synthesis,
                            synth_result.model,
                            fallback,
                            tokens_input=synth_result.tokens_input,
                            tokens_input_cached=synth_result.tokens_input_cached,
                            tokens_output=synth_result.tokens_output,
                        )
        else:
            synth_result = generate_text(provider_plan.synthesis, synthesis_prompt, max_tokens=synthesis_token_budget)
            if synth_result.success and synth_result.text.strip():
                draft = synth_result.text.strip()
                draft_texts.append(draft)
                _record_gen_usage(
                    provider_plan.synthesis,
                    synth_result.model,
                    draft,
                    tokens_input=synth_result.tokens_input,
                    tokens_input_cached=synth_result.tokens_input_cached,
                    tokens_output=synth_result.tokens_output,
                )
            else:
                synthesis_failures.append(f"{provider_plan.synthesis}: {synth_result.error or 'empty output'}")
                fallback = _build_answer_text(
                    question,
                    claim_texts,
                    unsupported_claims,
                    clusters=claim_clusters,
                    section_summaries=section_summaries,
                )
                draft_texts.append(fallback)
                _record_gen_usage(
                    provider_plan.synthesis,
                    synth_result.model,
                    fallback,
                    tokens_input=synth_result.tokens_input,
                    tokens_input_cached=synth_result.tokens_input_cached,
                    tokens_output=synth_result.tokens_output,
                )
    elif mode == Mode.RESEARCH_MAX.value:
        solver_a_provider = provider_plan.synthesis
        solver_b_provider = provider_plan.verification if provider_plan.verification != solver_a_provider else provider_plan.planning
        dual_start = datetime.now(timezone.utc)

        solver_a_prompt = (
            "Produce a best-supported answer from evidence. "
            "Use explicit claims and caveats.\n\n"
            f"{synthesis_prompt}"
        )
        solver_b_prompt = (
            "Act as a skeptical reviewer. Try to falsify weak claims and propose a conservative answer.\n\n"
            f"{synthesis_prompt}"
        )

        solver_a = generate_text(solver_a_provider, solver_a_prompt, max_tokens=synthesis_token_budget)
        solver_b = generate_text(solver_b_provider, solver_b_prompt, max_tokens=synthesis_token_budget)

        draft_a = solver_a.text.strip() if solver_a.success and solver_a.text.strip() else ""
        draft_b = solver_b.text.strip() if solver_b.success and solver_b.text.strip() else ""
        if not draft_a:
            synthesis_failures.append(f"{solver_a_provider}: {solver_a.error or 'empty output'}")
            draft_a = _build_answer_text(question, claim_texts, unsupported_claims, claim_clusters, section_summaries)
        if not draft_b:
            synthesis_failures.append(f"{solver_b_provider}: {solver_b.error or 'empty output'}")
            draft_b = _build_answer_text(question, claim_texts, unsupported_claims, claim_clusters, section_summaries)

        _record_gen_usage(
            solver_a_provider,
            solver_a.model,
            draft_a,
            tokens_input=solver_a.tokens_input,
            tokens_input_cached=solver_a.tokens_input_cached,
            tokens_output=solver_a.tokens_output,
        )
        _record_gen_usage(
            solver_b_provider,
            solver_b.model,
            draft_b,
            tokens_input=solver_b.tokens_input,
            tokens_input_cached=solver_b.tokens_input_cached,
            tokens_output=solver_b.tokens_output,
        )
        stages.append(
            TraceStage(
                name="dual_solver",
                status=Status.COMPLETED,
                model=provider_plan.synthesis,
                started_at=dual_start,
                ended_at=datetime.now(timezone.utc),
                artifacts={
                    "solver_a_provider": solver_a_provider,
                    "solver_b_provider": solver_b_provider,
                    "solver_a_success": solver_a.success,
                    "solver_b_success": solver_b.success,
                },
            )
        )

        judge_start = datetime.now(timezone.utc)
        judge_prompt = (
            "You are an adjudicator. Compare Solver A and Solver B and output the most defensible final answer.\n"
            "Return: winning_summary, unresolved_uncertainties, and key caveats.\n\n"
            f"Question: {question}\n\n"
            f"Solver A:\n{draft_a}\n\n"
            f"Solver B:\n{draft_b}\n"
        )
        judge = generate_text(provider_plan.planning, judge_prompt, max_tokens=synthesis_token_budget)
        judge_text = judge.text.strip() if judge.success and judge.text.strip() else ""
        if not judge_text:
            synthesis_failures.append(f"{provider_plan.planning}: {judge.error or 'empty output'}")
            judge_text = draft_a if len(draft_a) >= len(draft_b) else draft_b
        _record_gen_usage(
            provider_plan.planning,
            judge.model,
            judge_text,
            tokens_input=judge.tokens_input,
            tokens_input_cached=judge.tokens_input_cached,
            tokens_output=judge.tokens_output,
        )
        stages.append(
            TraceStage(
                name="debate_judge",
                status=Status.COMPLETED,
                model=provider_plan.planning,
                started_at=judge_start,
                ended_at=datetime.now(timezone.utc),
                artifacts={
                    "judge_success": judge.success,
                    "task_type": research_plan.task_type if research_plan else "factual",
                },
            )
        )

        finalized = _research_finalizer(question, judge_text, research_plan.task_type if research_plan else "factual")
        stages.append(
            TraceStage(
                name="research_finalizer",
                status=Status.COMPLETED,
                model="policy",
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                artifacts={"task_type": research_plan.task_type if research_plan else "factual"},
            )
        )
        draft_texts = [finalized]

    ensemble_agreement = 1.0
    if mode == Mode.ENSEMBLE.value and not synthesis_failures:
        ensemble_start = datetime.now(timezone.utc)
        drafts: list[str] = []
        for provider in provider_plan.ensemble:
            if should_stop_early(
                BudgetState(elapsed_s=(datetime.now(timezone.utc) - started).total_seconds(), spent_usd=spent_usd),
                max_latency_s,
                max_cost_usd,
                reserve_next_stage_usd=estimate_stage_cost("ensemble"),
            ):
                synthesis_failures.append(f"budget guard: ensemble call skipped for {provider}")
                continue
            result = generate_text(provider, synthesis_prompt, max_tokens=synthesis_token_budget)
            if result.success and result.text.strip():
                draft = result.text.strip()
            else:
                synthesis_failures.append(f"{provider}: {result.error or 'empty output'}")
                draft = (
                    f"[{provider}] "
                    f"{_build_answer_text(question, claim_texts, unsupported_claims, claim_clusters, section_summaries)}"
                )
            drafts.append(draft)
            _record_gen_usage(
                provider,
                result.model,
                draft,
                tokens_input=result.tokens_input,
                tokens_input_cached=result.tokens_input_cached,
                tokens_output=result.tokens_output,
            )

        if drafts:
            draft_texts = drafts
            ensemble_agreement = _ensemble_agreement(drafts)
        else:
            draft_texts = [
                _build_answer_text(
                    question,
                    claim_texts,
                    unsupported_claims,
                    clusters=claim_clusters,
                    section_summaries=section_summaries,
                )
            ]

        stages.append(
            TraceStage(
                name="ensemble_adjudication",
                status=Status.COMPLETED,
                model=provider_plan.synthesis,
                started_at=ensemble_start,
                ended_at=datetime.now(timezone.utc),
                artifacts={"providers": provider_plan.ensemble, "draft_count": len(drafts), "agreement": ensemble_agreement},
            )
        )

    observability.add_stage(
        StageMetric(
            name="synthesis",
            duration_ms=int((datetime.now(timezone.utc) - synthesis_start).total_seconds() * 1000),
            documents_retrieved=0,
            source_errors=len(synthesis_failures),
            estimated_cost_usd=sum(item.cost_usd for item in draft_usage),
        )
    )
    stages.append(
        TraceStage(
            name="synthesis",
            status=Status.COMPLETED,
            model=provider_plan.synthesis,
            started_at=synthesis_start,
            ended_at=datetime.now(timezone.utc),
            artifacts={
                "draft_count": len(draft_texts),
                "failures": synthesis_failures,
                "spent_usd": spent_usd,
                "claim_clusters": claim_clusters,
                "section_summaries": section_summaries,
            },
        )
    )

    if effort.synthesis_revision_pass and draft_texts:
        revision_start = datetime.now(timezone.utc)
        if _question_has_answer_choices(question):
            stages.append(
                TraceStage(
                    name="synthesis_revision",
                    status=Status.COMPLETED,
                    model=provider_plan.synthesis,
                    started_at=revision_start,
                    ended_at=datetime.now(timezone.utc),
                    artifacts={"status": "skipped", "reason": "question_has_answer_choices"},
                )
            )
        else:
            anchors = _extract_lexical_anchors(question)
            draft_coverage = _anchor_coverage(draft_texts[0], anchors)
            anchors_block = ""
            if anchors:
                joined = ", ".join(anchors)
                anchors_block = f"Preserve these exact technical strings when relevant: {joined}\n\n"
            revision_prompt = (
                "Revise the answer to improve evidence-grounding and caveats. "
                "Keep it concise and do not introduce new claims beyond evidence bullets.\n"
                f"{anchors_block}"
                f"Question: {question}\n"
                f"Draft answer:\n{draft_texts[0]}"
            )
            revision = generate_text(provider_plan.synthesis, revision_prompt, max_tokens=synthesis_token_budget)
            if revision.success and revision.text.strip():
                revised_text = revision.text.strip()
                revised_coverage = _anchor_coverage(revised_text, anchors)
                if revised_coverage + 1e-9 < draft_coverage:
                    stages.append(
                        TraceStage(
                            name="synthesis_revision",
                            status=Status.COMPLETED,
                            model=provider_plan.synthesis,
                            started_at=revision_start,
                            ended_at=datetime.now(timezone.utc),
                            artifacts={
                                "status": "skipped",
                                "reason": "lexical_anchor_regression",
                                "anchor_count": len(anchors),
                                "draft_anchor_coverage": draft_coverage,
                                "revised_anchor_coverage": revised_coverage,
                            },
                        )
                    )
                else:
                    draft_texts = [revised_text]
                    _record_gen_usage(
                        provider_plan.synthesis,
                        revision.model,
                        revised_text,
                        tokens_input=revision.tokens_input,
                        tokens_input_cached=revision.tokens_input_cached,
                        tokens_output=revision.tokens_output,
                    )
                    stages.append(
                        TraceStage(
                            name="synthesis_revision",
                            status=Status.COMPLETED,
                            model=provider_plan.synthesis,
                            started_at=revision_start,
                            ended_at=datetime.now(timezone.utc),
                            artifacts={
                                "status": "applied",
                                "anchor_count": len(anchors),
                                "draft_anchor_coverage": draft_coverage,
                                "revised_anchor_coverage": revised_coverage,
                            },
                        )
                    )
            else:
                stages.append(
                    TraceStage(
                        name="synthesis_revision",
                        status=Status.COMPLETED,
                        model=provider_plan.synthesis,
                        started_at=revision_start,
                        ended_at=datetime.now(timezone.utc),
                        artifacts={"status": "skipped", "reason": revision.error or "empty output"},
                    )
                )

    configured_count = sum(1 for status in provider_statuses if status.configured)
    features = CalibrationFeatures(
        support_coverage=len(citations) / max(1, len(base_claim_conf)),
        unsupported_claims=unsupported_claims,
        contradiction_flags=verifier_result.contradiction_flags,
        provider_config_ratio=configured_count / max(1, len(provider_statuses)),
        ensemble_agreement=ensemble_agreement,
        evidence_count=len(selected_records),
    )
    answer_conf, calibrated_claims = calibrate_answer(features, adjusted_claim_conf)
    CalibrationStore().upsert_features(run_id=run_id, features=features_to_dict(features), answer_confidence=answer_conf)

    claim_confidences = [ClaimConfidence(claim_id=claim_id, confidence=confidence) for claim_id, confidence in calibrated_claims.items()]

    uncertainty_reasons: list[str] = []
    if len(records) < min_sources:
        uncertainty_reasons.append("Retrieved source count below requested minimum")
    if aggregate_errors:
        uncertainty_reasons.append("One or more retrieval providers failed or returned sparse results")
    if missing_keys:
        uncertainty_reasons.append(f"Missing API keys for providers: {', '.join(status.provider for status in missing_keys)}")
    if unsupported_claims:
        uncertainty_reasons.append("Some generated claims lacked direct passage support")
    if synthesis_failures:
        uncertainty_reasons.append("One or more provider generation calls failed; fallback synthesis used")
    if budget_stop_reason:
        uncertainty_reasons.append(budget_stop_reason)
    uncertainty_reasons.extend(verifier_result.notes)

    abstain_reason_codes = _abstain_reasons(
        min_sources=min_sources,
        retrieved_count=len(selected_records),
        support_coverage=features.support_coverage,
        unsupported_claims=unsupported_claims,
        contradiction_flags=verifier_result.contradiction_flags,
        synthesis_failures=synthesis_failures,
    )
    should_abstain = len(abstain_reason_codes) >= 2
    disable_hard_abstain = _env_bool("SPARKIT_DISABLE_HARD_ABSTAIN", False)

    final_text = (
        draft_texts[0]
        if draft_texts
        else _build_answer_text(
            question,
            claim_texts,
            unsupported_claims,
            clusters=claim_clusters,
            section_summaries=section_summaries,
        )
    )
    if mode == Mode.ENSEMBLE.value and draft_texts:
        final_text = max(draft_texts, key=len)
    if should_abstain:
        uncertainty_reasons.extend(f"abstain:{code}" for code in abstain_reason_codes)
        if disable_hard_abstain:
            answer_conf = min(answer_conf, 0.35)
            stages.append(
                TraceStage(
                    name="answerability_gate",
                    status=Status.COMPLETED,
                    model="policy",
                    started_at=datetime.now(timezone.utc),
                    ended_at=datetime.now(timezone.utc),
                    artifacts={
                        "abstained": False,
                        "soft_abstain": True,
                        "reasons": abstain_reason_codes,
                        "support_coverage": features.support_coverage,
                        "unsupported_claims": unsupported_claims,
                        "contradiction_flags": verifier_result.contradiction_flags,
                    },
                )
            )
        else:
            final_text = (
                "Insufficient evidence quality to provide a reliable answer. "
                "Retrieved evidence was sparse or weakly grounded for this question."
            )
            answer_conf = min(answer_conf, 0.2)
            stages.append(
                TraceStage(
                    name="answerability_gate",
                    status=Status.COMPLETED,
                    model="policy",
                    started_at=datetime.now(timezone.utc),
                    ended_at=datetime.now(timezone.utc),
                    artifacts={
                        "abstained": True,
                        "soft_abstain": False,
                        "reasons": abstain_reason_codes,
                        "support_coverage": features.support_coverage,
                        "unsupported_claims": unsupported_claims,
                        "contradiction_flags": verifier_result.contradiction_flags,
                    },
                )
            )
    else:
        stages.append(
            TraceStage(
                name="answerability_gate",
                status=Status.COMPLETED,
                model="policy",
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                artifacts={"abstained": False, "reasons": []},
            )
        )

    answer = Answer(final_text=final_text, answer_confidence=answer_conf, claim_confidences=claim_confidences, uncertainty_reasons=uncertainty_reasons)

    quality_gates = QualityGates(
        citation_coverage=features.support_coverage,
        unsupported_claims=unsupported_claims,
        contradiction_flags=verifier_result.contradiction_flags,
    )

    provider_usage = [
        ProviderUsage(provider=provider_plan.retrieval, model="retrieval-service", tokens_input=0, tokens_output=0, cost_usd=retrieval_base_cost_usd),
        ProviderUsage(provider=provider_plan.verification, model="verifier-v2", tokens_input=0, tokens_output=0, cost_usd=verifier_cost),
        *draft_usage,
    ]
    if brave_request_count > 0:
        provider_usage.append(
            ProviderUsage(provider="brave_web", model="search-api", tokens_input=0, tokens_output=0, cost_usd=retrieval_brave_cost_usd)
        )

    observability.finish(budget_stop_reason=budget_stop_reason)
    ObservabilityStore().upsert_metrics(run_id=run_id, metrics=observability.to_dict())

    return OrchestrationResult(
        answer=answer,
        citations=citations,
        stages=stages,
        quality_gates=quality_gates,
        source_errors=aggregate_errors,
        provider_usage=provider_usage,
    )
