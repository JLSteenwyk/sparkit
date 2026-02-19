from __future__ import annotations

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


def _build_round_queries(question: str) -> list[tuple[str, list[str]]]:
    return [
        ("retrieval_round_1", [question, f"{question} review"]),
        ("retrieval_round_2_gap_fill", [f"{question} limitations", f"{question} benchmark comparison"]),
        ("retrieval_round_3_adversarial", [f"{question} contradictory findings", f"{question} negative results"]),
    ]


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
    max_latency_s: int = 120,
    max_cost_usd: float = 3.0,
    prompt_version: str = 'synthesis_v1.2',
    config_version: str = 'orchestration_v1.2',
    reproducibility: dict | None = None,
) -> OrchestrationResult:
    started = datetime.now(timezone.utc)
    observability = RunObservability(run_id=run_id)

    provider_list = providers or ["openai"]
    provider_statuses = build_default_registry().resolve(provider_list)
    missing_keys = [status for status in provider_statuses if not status.configured]
    provider_plan = build_provider_plan(mode=mode, statuses=provider_statuses, requested=provider_list)

    rounds = _build_round_queries(question)
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
                "strategy": "three-round retrieval + verification + calibration + policy",
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
                "prompt_version": prompt_version,
                "config_version": config_version,
                "reproducibility": reproducibility or {},
            },
        )
    ]

    budget_stop_reason: str | None = None
    spent_usd = 0.0

    for stage_name, queries in rounds:
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

        for query in queries:
            found, errors = search_literature(query, max_results=max(min_sources, 6))
            stage_records.extend(found)
            for source, err in errors.items():
                stage_errors[f"{source}:{query}"] = err

        deduped_stage = _dedupe_records(stage_records)
        records_by_round[stage_name] = deduped_stage
        all_records.extend(deduped_stage)
        aggregate_errors.update(stage_errors)

        stage_cost = estimate_stage_cost("retrieval", units=len(queries))
        spent_usd += stage_cost
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
                    "source_errors": stage_errors,
                    "estimated_cost_usd": stage_cost,
                },
            )
        )

    records = _dedupe_records(all_records)
    evidence_store = EvidenceStore()

    citations: list[Citation] = []
    claim_texts: list[str] = []
    claim_evidence: list[ClaimEvidence] = []
    base_claim_conf: dict[str, float] = {}
    unsupported_claims = 0

    for record in records[: max(min_sources, 3)]:
        claim_text = f"{record.title} ({record.year or 'n.d.'}) indicates relevant evidence for the question."
        section_name = "abstract"
        section_text = record.abstract or ""
        try:
            ingested = fetch_and_parse(record.url, max_chars=4000, timeout_s=12.0)
            if ingested.sections:
                section_name = ingested.sections[0].heading
                section_text = ingested.sections[0].text
        except Exception:  # noqa: BLE001
            pass

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

    verifier_start = datetime.now(timezone.utc)
    depth = contradiction_depth_from_budget(max_cost_usd=max_cost_usd, max_latency_s=max_latency_s)
    verifier_result = run_verifier(
        claim_ids=list(base_claim_conf.keys()),
        adversarial_records=records_by_round.get("retrieval_round_3_adversarial", []),
        depth=depth,
        top_k=5,
    )
    verifier_cost = estimate_stage_cost("verification", units=max(1, depth))
    spent_usd += verifier_cost
    observability.add_stage(
        StageMetric(
            name="verification",
            duration_ms=int((datetime.now(timezone.utc) - verifier_start).total_seconds() * 1000),
            documents_retrieved=len(records_by_round.get("retrieval_round_3_adversarial", [])),
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
        synth_result = generate_text(provider_plan.synthesis, synthesis_prompt, max_tokens=700)
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
            result = generate_text(provider, synthesis_prompt, max_tokens=700)
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

    configured_count = sum(1 for status in provider_statuses if status.configured)
    features = CalibrationFeatures(
        support_coverage=len(citations) / max(1, len(base_claim_conf)),
        unsupported_claims=unsupported_claims,
        contradiction_flags=verifier_result.contradiction_flags,
        provider_config_ratio=configured_count / max(1, len(provider_statuses)),
        ensemble_agreement=ensemble_agreement,
        evidence_count=len(records),
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

    answer = Answer(final_text=final_text, answer_confidence=answer_conf, claim_confidences=claim_confidences, uncertainty_reasons=uncertainty_reasons)

    quality_gates = QualityGates(
        citation_coverage=features.support_coverage,
        unsupported_claims=unsupported_claims,
        contradiction_flags=verifier_result.contradiction_flags,
    )

    provider_usage = [
        ProviderUsage(provider=provider_plan.retrieval, model="retrieval-service", tokens_input=0, tokens_output=0, cost_usd=estimate_stage_cost("retrieval")),
        ProviderUsage(provider=provider_plan.verification, model="verifier-v2", tokens_input=0, tokens_output=0, cost_usd=verifier_cost),
        *draft_usage,
    ]

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
