from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from services.orchestrator.app.providers.registry import ProviderStatus
from services.orchestrator.app.routing import ProviderPlan
from services.orchestrator.app.routing import build_provider_plan
from services.orchestrator.app.verifier import VerificationResult, run_verifier
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


@dataclass(frozen=True)
class AdaptiveRetrievalConfig:
    enabled: bool
    min_rounds: int
    max_rounds: int
    min_new_docs: int
    min_quality_gain: float


@dataclass(frozen=True)
class RetrievalExecutionResult:
    records_by_round: dict[str, list[LiteratureRecord]]
    all_records: list[LiteratureRecord]
    aggregate_errors: dict[str, str]
    spent_usd: float
    retrieval_base_cost_usd: float
    retrieval_brave_cost_usd: float
    brave_request_count: int
    budget_stop_reason: str | None


@dataclass(frozen=True)
class LiveWebFetchResult:
    all_records: list[LiteratureRecord]
    aggregate_errors: dict[str, str]
    spent_usd: float
    retrieval_base_cost_usd: float
    retrieval_brave_cost_usd: float
    brave_request_count: int


@dataclass(frozen=True)
class UsageSnapshot:
    spent_usd: float
    usage_rows: list[ProviderUsage]


@dataclass(frozen=True)
class SynthesisExecutionResult:
    spent_usd: float
    synthesis_failures: list[str]
    draft_texts: list[str]
    draft_usage: list[ProviderUsage]
    ensemble_agreement: float


@dataclass(frozen=True)
class RevisionExecutionResult:
    spent_usd: float
    draft_texts: list[str]


@dataclass(frozen=True)
class EvidenceAssemblyResult:
    spent_usd: float
    citations: list[Citation]
    claim_texts: list[str]
    claim_evidence: list[ClaimEvidence]
    base_claim_conf: dict[str, float]
    unsupported_claims: int


@dataclass(frozen=True)
class VerificationExecutionResult:
    spent_usd: float
    verifier_cost: float
    verifier_result: VerificationResult
    adjusted_claim_conf: dict[str, float]


@dataclass(frozen=True)
class FinalizationResult:
    answer: Answer
    quality_gates: QualityGates
    provider_usage: list[ProviderUsage]


@dataclass(frozen=True)
class ExecutionContext:
    run_id: str
    question: str
    mode: str
    min_sources: int
    max_latency_s: int | None
    max_cost_usd: float
    provider_plan: ProviderPlan


@dataclass(frozen=True)
class EvidenceState:
    records: list[LiteratureRecord]
    selected_records: list[LiteratureRecord]
    citations: list[Citation]
    claim_texts: list[str]
    claim_evidence: list[ClaimEvidence]
    base_claim_conf: dict[str, float]
    unsupported_claims: int
    claim_clusters: list[dict[str, object]]
    section_summaries: list[dict[str, str]]


@dataclass(frozen=True)
class BudgetStateRuntime:
    spent_usd: float
    budget_stop_reason: str | None
    retrieval_base_cost_usd: float
    retrieval_brave_cost_usd: float
    brave_request_count: int
    verifier_cost: float = 0.0


@dataclass(frozen=True)
class OrchestrationConfig:
    min_sources: int
    providers: list[str]
    mode: str
    max_latency_s: int | None
    max_cost_usd: float
    synthesis_max_tokens: int | None
    prompt_version: str
    config_version: str
    reproducibility: dict

    @classmethod
    def from_inputs(
        cls,
        *,
        min_sources: int,
        providers: list[str] | None,
        mode: str,
        max_latency_s: int | None,
        max_cost_usd: float,
        synthesis_max_tokens: int | None,
        prompt_version: str,
        config_version: str,
        reproducibility: dict | None,
    ) -> "OrchestrationConfig":
        return cls(
            min_sources=min_sources,
            providers=providers or ["openai"],
            mode=mode,
            max_latency_s=max_latency_s,
            max_cost_usd=max_cost_usd,
            synthesis_max_tokens=synthesis_max_tokens,
            prompt_version=prompt_version,
            config_version=config_version,
            reproducibility=reproducibility or {},
        )


def _serialize_research_plan(plan: ResearchPlan | None) -> dict | None:
    if plan is None:
        return None
    return {
        "task_type": plan.task_type,
        "sub_claims": plan.sub_claims,
        "output_schema": plan.output_schema,
        "disambiguations": plan.disambiguations,
    }


def _serialize_retrieval_plan(plan: RetrievalPlan | None) -> dict:
    if plan is None:
        return {"segments": [], "focus_terms": [], "intents": {}, "answer_choices": {}}
    return {
        "segments": plan.segments,
        "focus_terms": plan.focus_terms,
        "intents": plan.intent_queries,
        "answer_choices": plan.answer_choices,
    }


def _adaptive_retrieval_config(round_count: int) -> AdaptiveRetrievalConfig:
    return AdaptiveRetrievalConfig(
        enabled=_env_bool("SPARKIT_ADAPTIVE_RETRIEVAL", True),
        min_rounds=_env_int("SPARKIT_ADAPTIVE_MIN_ROUNDS", 2, minimum=1),
        max_rounds=_env_int("SPARKIT_ADAPTIVE_MAX_ROUNDS", round_count, minimum=1),
        min_new_docs=_env_int("SPARKIT_ADAPTIVE_MIN_NEW_DOCS", 2, minimum=0),
        min_quality_gain=_env_float("SPARKIT_ADAPTIVE_MIN_QUALITY_GAIN", 0.03, minimum=0.0),
    )


def _append_plan_stages(
    *,
    stages: list[TraceStage],
    started: datetime,
    question: str,
    config: OrchestrationConfig,
    provider_plan: ProviderPlan,
    provider_statuses: list[ProviderStatus],
    effort: EffortProfile,
    rounds: list[tuple[str, list[str]]],
    ingestion_max_chars: int,
    synthesis_token_budget: int | None,
    research_plan: ResearchPlan | None,
    retrieval_plan: RetrievalPlan | None,
    adaptive: AdaptiveRetrievalConfig,
) -> None:
    stages.append(
        TraceStage(
            name="plan",
            status=Status.COMPLETED,
            model=provider_plan.planning,
            started_at=started,
            ended_at=started,
            artifacts={
                "strategy": f"{effort.name}-effort retrieval + verification + calibration + policy",
                "question": question,
                "mode": config.mode,
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
                "budget": {"max_latency_s": config.max_latency_s, "max_cost_usd": config.max_cost_usd},
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
                        "enabled": adaptive.enabled,
                        "min_rounds": adaptive.min_rounds,
                        "max_rounds": adaptive.max_rounds,
                        "min_new_docs": adaptive.min_new_docs,
                        "min_quality_gain": adaptive.min_quality_gain,
                    },
                },
                "prompt_version": config.prompt_version,
                "config_version": config.config_version,
                "reproducibility": config.reproducibility,
                "research_plan": _serialize_research_plan(research_plan),
                "retrieval_plan": _serialize_retrieval_plan(retrieval_plan),
            },
        )
    )

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


def _run_retrieval_rounds(
    *,
    started: datetime,
    question: str,
    rounds: list[tuple[str, list[str]]],
    effort: EffortProfile,
    provider_plan: ProviderPlan,
    retrieval_plan: RetrievalPlan | None,
    stages: list[TraceStage],
    observability: RunObservability,
    max_latency_s: int | None,
    max_cost_usd: float,
) -> RetrievalExecutionResult:
    budget_stop_reason: str | None = None
    spent_usd = 0.0
    retrieval_base_cost_usd = 0.0
    retrieval_brave_cost_usd = 0.0
    brave_request_count = 0
    adaptive = _adaptive_retrieval_config(len(rounds))
    seen_record_ids: set[str] = set()
    prev_selected_quality = 0.0
    records_by_round: dict[str, list[LiteratureRecord]] = {}
    all_records: list[LiteratureRecord] = []
    aggregate_errors: dict[str, str] = {}
    mutable_rounds = [(name, list(queries)) for name, queries in rounds]

    for stage_idx, (stage_name, queries) in enumerate(mutable_rounds, start=1):
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
        semantic_rerank_enabled = _env_bool("SPARKIT_ENABLE_SEMANTIC_RERANK", False)
        if semantic_rerank_enabled and _semantic_rerank_enabled_for_stage(stage_name) and deduped_stage:
            rerank_query = queries[0] if queries else question
            rerank_top_k = min(len(deduped_stage), effort.retrieval_min_results)
            deduped_stage = _semantic_rerank_records(
                question=f"{question}\nFocused query: {rerank_query}",
                records=deduped_stage,
                provider=provider_plan.retrieval,
                top_k=rerank_top_k,
            )
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
                    "semantic_reranked": bool(semantic_rerank_enabled and _semantic_rerank_enabled_for_stage(stage_name)),
                    "source_errors": stage_errors,
                    "brave_requests": stage_brave_requests,
                    "brave_cost_usd": stage_brave_cost,
                    "estimated_cost_usd": stage_cost,
                },
            )
        )
        stage_avg_relevance = _avg_relevance(
            question=question,
            records=deduped_stage,
            boost_terms=retrieval_plan.focus_terms if retrieval_plan else None,
        )
        elapsed_after_stage_s = (datetime.now(timezone.utc) - started).total_seconds()
        gap_allowed, gap_reason = _should_inject_claim_gap(
            stage_idx=stage_idx,
            total_stages=len(mutable_rounds),
            new_unique_docs=new_unique_docs,
            stage_avg_relevance=stage_avg_relevance,
            elapsed_s=elapsed_after_stage_s,
            spent_usd=spent_usd,
            max_latency_s=max_latency_s,
            max_cost_usd=max_cost_usd,
        )
        stages.append(
            TraceStage(
                name="retrieval_claim_gap_gate",
                status=Status.COMPLETED,
                model="policy",
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                artifacts={
                    "stage": stage_name,
                    "stage_idx": stage_idx,
                    "new_unique_docs": new_unique_docs,
                    "stage_avg_relevance": stage_avg_relevance,
                    "elapsed_s": elapsed_after_stage_s,
                    "spent_usd": spent_usd,
                    "reason": gap_reason,
                    "enabled": gap_allowed,
                },
            )
        )
        if gap_allowed and stage_idx < len(mutable_rounds):
            gap_queries = _build_claim_gap_queries(
                question=question,
                stage_name=stage_name,
                records=deduped_stage,
                planning_provider=provider_plan.planning,
                max_items=_env_int("SPARKIT_CLAIM_GAP_MAX_QUERIES", 4, minimum=1),
            )
            if gap_queries:
                next_stage_name, next_stage_queries = mutable_rounds[stage_idx]
                merged_next = _dedupe_queries(
                    [*next_stage_queries, *gap_queries],
                    max_items=_env_int("SPARKIT_CLAIM_GAP_MAX_NEXT_QUERIES", 12, minimum=4),
                )
                mutable_rounds[stage_idx] = (next_stage_name, merged_next)
                stages.append(
                    TraceStage(
                        name="retrieval_claim_gap_loop",
                        status=Status.COMPLETED,
                        model=provider_plan.planning,
                        started_at=datetime.now(timezone.utc),
                        ended_at=datetime.now(timezone.utc),
                        artifacts={
                            "from_stage": stage_name,
                            "to_stage": next_stage_name,
                            "injected_queries": gap_queries,
                            "injected_count": len(gap_queries),
                            "next_stage_query_count_after_merge": len(merged_next),
                        },
                    )
                )
        if adaptive.enabled:
            if stage_idx >= adaptive.max_rounds:
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
                            "max_rounds": adaptive.max_rounds,
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
            if stage_idx >= adaptive.min_rounds:
                low_novelty = new_unique_docs < adaptive.min_new_docs
                low_gain = quality_gain < adaptive.min_quality_gain
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
                                "min_new_docs": adaptive.min_new_docs,
                                "selected_quality": selected_quality,
                                "quality_gain": quality_gain,
                                "min_quality_gain": adaptive.min_quality_gain,
                            },
                        )
                    )
                    break

    return RetrievalExecutionResult(
        records_by_round=records_by_round,
        all_records=all_records,
        aggregate_errors=aggregate_errors,
        spent_usd=spent_usd,
        retrieval_base_cost_usd=retrieval_base_cost_usd,
        retrieval_brave_cost_usd=retrieval_brave_cost_usd,
        brave_request_count=brave_request_count,
        budget_stop_reason=budget_stop_reason,
    )


def _run_live_web_tool_loop(
    *,
    question: str,
    records: list[LiteratureRecord],
    planning_provider: str,
    retrieval_provider: str,
    max_results: int,
    spent_usd: float,
    retrieval_base_cost_usd: float,
    retrieval_brave_cost_usd: float,
    brave_request_count: int,
    stages: list[TraceStage],
    observability: RunObservability,
) -> LiveWebFetchResult:
    enabled = _env_bool("SPARKIT_ENABLE_LIVE_WEB_TOOL_LOOP", True)
    if not enabled:
        stages.append(
            TraceStage(
                name="retrieval_live_web_tool_loop",
                status=Status.COMPLETED,
                model="policy",
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                artifacts={"enabled": False, "reason": "disabled_by_env"},
            )
        )
        return LiveWebFetchResult(
            all_records=records,
            aggregate_errors={},
            spent_usd=spent_usd,
            retrieval_base_cost_usd=retrieval_base_cost_usd,
            retrieval_brave_cost_usd=retrieval_brave_cost_usd,
            brave_request_count=brave_request_count,
        )

    seed_records = _dedupe_records(records)[: _env_int("SPARKIT_LIVE_WEB_SEED_DOCS", 8, minimum=1)]
    tool_queries = _build_claim_gap_queries(
        question=question,
        stage_name="live_web_tool_loop",
        records=seed_records,
        planning_provider=planning_provider,
        max_items=_env_int("SPARKIT_LIVE_WEB_MAX_QUERIES", 2, minimum=1),
    )
    if not tool_queries:
        stages.append(
            TraceStage(
                name="retrieval_live_web_tool_loop",
                status=Status.COMPLETED,
                model=planning_provider,
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                artifacts={"enabled": True, "queries": [], "reason": "no_queries_generated"},
            )
        )
        return LiveWebFetchResult(
            all_records=records,
            aggregate_errors={},
            spent_usd=spent_usd,
            retrieval_base_cost_usd=retrieval_base_cost_usd,
            retrieval_brave_cost_usd=retrieval_brave_cost_usd,
            brave_request_count=brave_request_count,
        )

    stage_start = datetime.now(timezone.utc)
    fetched: list[LiteratureRecord] = []
    aggregate_errors: dict[str, str] = {}
    stage_brave_requests = 0
    per_query_results: dict[str, int] = {}
    for query in tool_queries:
        found, errors, stats = search_literature(query, max_results=max_results, force_web=True)
        deduped_found = _dedupe_records(found)
        fetched.extend(deduped_found)
        per_query_results[query] = len(deduped_found)
        for source, err in errors.items():
            aggregate_errors[f"live_web:{source}:{query}"] = err
        stage_brave_requests += int((stats.get("requests_by_source") or {}).get("brave_web", 0))

    merged_records = _dedupe_records([*records, *_dedupe_records(fetched)])
    stage_base_cost = estimate_stage_cost("retrieval", units=len(tool_queries))
    stage_brave_cost = estimate_brave_search_cost(stage_brave_requests)
    stage_cost = stage_base_cost + stage_brave_cost
    spent_usd += stage_cost
    retrieval_base_cost_usd += stage_base_cost
    retrieval_brave_cost_usd += stage_brave_cost
    brave_request_count += stage_brave_requests

    observability.add_stage(
        StageMetric(
            name="retrieval_live_web_tool_loop",
            duration_ms=int((datetime.now(timezone.utc) - stage_start).total_seconds() * 1000),
            documents_retrieved=len(fetched),
            source_errors=len(aggregate_errors),
            estimated_cost_usd=stage_cost,
        )
    )
    stages.append(
        TraceStage(
            name="retrieval_live_web_tool_loop",
            status=Status.COMPLETED,
            model=retrieval_provider,
            started_at=stage_start,
            ended_at=datetime.now(timezone.utc),
            artifacts={
                "enabled": True,
                "queries": tool_queries,
                "query_result_counts": per_query_results,
                "new_documents": len(_dedupe_records(fetched)),
                "merged_documents": len(merged_records),
                "source_errors": aggregate_errors,
                "brave_requests": stage_brave_requests,
                "brave_cost_usd": stage_brave_cost,
                "estimated_cost_usd": stage_cost,
            },
        )
    )

    return LiveWebFetchResult(
        all_records=merged_records,
        aggregate_errors=aggregate_errors,
        spent_usd=spent_usd,
        retrieval_base_cost_usd=retrieval_base_cost_usd,
        retrieval_brave_cost_usd=retrieval_brave_cost_usd,
        brave_request_count=brave_request_count,
    )


def _assemble_evidence_and_build_claims(
    *,
    run_id: str,
    question: str,
    selected_records: list[LiteratureRecord],
    retrieval_plan: RetrievalPlan | None,
    ingestion_max_chars: int,
    spent_usd: float,
) -> EvidenceAssemblyResult:
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

    return EvidenceAssemblyResult(
        spent_usd=spent_usd,
        citations=citations,
        claim_texts=claim_texts,
        claim_evidence=claim_evidence,
        base_claim_conf=base_claim_conf,
        unsupported_claims=unsupported_claims,
    )


def _run_verification_and_adjust_confidence(
    *,
    mode: str,
    max_cost_usd: float,
    max_latency_s: int | None,
    effort: EffortProfile,
    records_by_round: dict[str, list[LiteratureRecord]],
    base_claim_conf: dict[str, float],
    spent_usd: float,
    provider_plan: ProviderPlan,
    stages: list[TraceStage],
    observability: RunObservability,
) -> VerificationExecutionResult:
    adversarial_stage_name = "retrieval_adversarial" if mode == Mode.RESEARCH_MAX.value else "retrieval_round_3_adversarial"
    verifier_records = list(records_by_round.get(adversarial_stage_name, []))
    verifier_records.extend(records_by_round.get("retrieval_round_4_falsification", []))
    verifier_records = _dedupe_records(verifier_records)
    verifier_start = datetime.now(timezone.utc)
    depth = contradiction_depth_from_budget(max_cost_usd=max_cost_usd, max_latency_s=max_latency_s) + effort.contradiction_depth_bonus
    verifier_result = run_verifier(
        claim_ids=list(base_claim_conf.keys()),
        adversarial_records=verifier_records,
        depth=depth,
        top_k=5,
    )
    verifier_cost = estimate_stage_cost("verification", units=max(1, depth))
    spent_usd += verifier_cost
    observability.add_stage(
        StageMetric(
            name="verification",
            duration_ms=int((datetime.now(timezone.utc) - verifier_start).total_seconds() * 1000),
            documents_retrieved=len(verifier_records),
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
                "adversarial_stage_name": adversarial_stage_name,
                "falsification_docs_used": len(records_by_round.get("retrieval_round_4_falsification", [])),
            },
        )
    )

    adjusted_claim_conf: dict[str, float] = {}
    for claim_id, base in base_claim_conf.items():
        penalty = verifier_result.penalties.get(claim_id, 0.0)
        adjusted_claim_conf[claim_id] = max(0.05, min(0.95, base - penalty))

    return VerificationExecutionResult(
        spent_usd=spent_usd,
        verifier_cost=verifier_cost,
        verifier_result=verifier_result,
        adjusted_claim_conf=adjusted_claim_conf,
    )


def _append_generation_usage(
    *,
    spent_usd: float,
    usage_rows: list[ProviderUsage],
    provider: str,
    model: str,
    prompt_text: str,
    draft_text: str,
    tokens_input: int = 0,
    tokens_input_cached: int = 0,
    tokens_output: int = 0,
) -> UsageSnapshot:
    out_tokens = tokens_output if tokens_output > 0 else max(1, len(draft_text) // 4)
    input_tokens = tokens_input if tokens_input > 0 else max(1, len(prompt_text) // 4)
    est = estimate_generation_cost(
        provider=provider,
        model=model,
        tokens_input=input_tokens,
        tokens_input_cached=tokens_input_cached,
        tokens_output=out_tokens,
    )
    usage_rows.append(
        ProviderUsage(
            provider=provider,
            model=model,
            tokens_input=max(0, int(input_tokens)),
            tokens_output=max(0, int(out_tokens)),
            cost_usd=est,
        )
    )
    return UsageSnapshot(spent_usd=spent_usd + est, usage_rows=usage_rows)


def _run_synthesis_phase(
    *,
    started: datetime,
    mode: str,
    max_latency_s: int | None,
    max_cost_usd: float,
    spent_usd: float,
    provider_plan: ProviderPlan,
    question: str,
    stem_question: str,
    answer_choices: dict[str, str],
    claim_texts: list[str],
    unsupported_claims: int,
    claim_clusters: list[dict[str, object]],
    section_summaries: list[dict[str, str]],
    synthesis_prompt: str,
    synthesis_token_budget: int | None,
    research_plan: ResearchPlan | None,
    stages: list[TraceStage],
    difficulty_profile: str = "easy",
) -> SynthesisExecutionResult:
    synthesis_failures: list[str] = []
    draft_texts: list[str] = []
    draft_usage: list[ProviderUsage] = []
    ensemble_agreement = 1.0

    def _record_gen_usage(
        provider: str,
        model: str,
        draft: str,
        prompt_text: str | None = None,
        tokens_input: int = 0,
        tokens_input_cached: int = 0,
        tokens_output: int = 0,
    ) -> None:
        nonlocal spent_usd
        snapshot = _append_generation_usage(
            spent_usd=spent_usd,
            usage_rows=draft_usage,
            provider=provider,
            model=model,
            prompt_text=prompt_text or synthesis_prompt,
            draft_text=draft,
            tokens_input=tokens_input,
            tokens_input_cached=tokens_input_cached,
            tokens_output=tokens_output,
        )
        spent_usd = snapshot.spent_usd

    if should_stop_early(
        BudgetState(elapsed_s=(datetime.now(timezone.utc) - started).total_seconds(), spent_usd=spent_usd),
        max_latency_s,
        max_cost_usd,
        reserve_next_stage_usd=estimate_stage_cost("synthesis"),
    ):
        synthesis_failures.append("budget guard: synthesis skipped")
        if _question_has_answer_choices(question) and answer_choices:
            lexical_scores = _mcq_lexical_option_scores(
                answer_choices=answer_choices,
                claim_texts=claim_texts,
                section_summaries=section_summaries,
            )
            selected_letter = (
                max(
                    lexical_scores.items(),
                    key=lambda item: float(item[1].get("lexical", 0.0)),
                )[0]
                if lexical_scores
                else sorted(answer_choices.keys())[0]
            )
            draft = f"<answer>{selected_letter}</answer>"
            draft_texts.append(draft)
            _record_gen_usage(provider_plan.synthesis, provider_plan.synthesis, draft)
            stages.append(
                TraceStage(
                    name="mcq_budget_fallback",
                    status=Status.COMPLETED,
                    model="policy",
                    started_at=datetime.now(timezone.utc),
                    ended_at=datetime.now(timezone.utc),
                    artifacts={
                        "selected_option": selected_letter,
                        "reason": "synthesis_budget_guard",
                    },
                )
            )
        else:
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
        hard_parallel_enabled = _env_bool("SPARKIT_ENABLE_HARD_PARALLEL_CANDIDATES", True)
        if difficulty_profile == "hard" and hard_parallel_enabled and not (_question_has_answer_choices(question) and answer_choices):
            variant_count = _env_int("SPARKIT_HARD_PARALLEL_CANDIDATES", 3, minimum=2)
            variant_count = min(4, variant_count)
            prompt_variants = [
                synthesis_prompt,
                (
                    "Focus on falsification, alternative explanations, and uncertainty.\n"
                    f"{synthesis_prompt}"
                ),
                (
                    "Prioritize methods/results grounding. Penalize weak claims and overreach.\n"
                    f"{synthesis_prompt}"
                ),
                (
                    "Produce a conservative, evidence-first answer with explicit caveats.\n"
                    f"{synthesis_prompt}"
                ),
            ][:variant_count]
            generated: list[tuple[str, object]] = []
            start_parallel = datetime.now(timezone.utc)
            with ThreadPoolExecutor(max_workers=len(prompt_variants)) as executor:
                future_map = {
                    executor.submit(generate_text, provider_plan.synthesis, prompt_variant, synthesis_token_budget): prompt_variant
                    for prompt_variant in prompt_variants
                }
                for future in as_completed(future_map):
                    prompt_variant = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        synthesis_failures.append(f"{provider_plan.synthesis}: parallel_candidate_exception ({exc})")
                        continue
                    generated.append((prompt_variant, result))

            candidate_drafts: list[str] = []
            for prompt_variant, result in generated:
                if result.success and result.text.strip():
                    draft = result.text.strip()
                    candidate_drafts.append(draft)
                    _record_gen_usage(
                        provider_plan.synthesis,
                        result.model,
                        draft,
                        prompt_text=prompt_variant,
                        tokens_input=result.tokens_input,
                        tokens_input_cached=result.tokens_input_cached,
                        tokens_output=result.tokens_output,
                    )
                else:
                    synthesis_failures.append(f"{provider_plan.synthesis}: {result.error or 'empty output'}")

            if candidate_drafts:
                draft_texts = [_select_best_parallel_draft(question, candidate_drafts)]
                ensemble_agreement = _ensemble_agreement(candidate_drafts)
            else:
                fallback = _build_answer_text(
                    question,
                    claim_texts,
                    unsupported_claims,
                    clusters=claim_clusters,
                    section_summaries=section_summaries,
                )
                draft_texts = [fallback]
                _record_gen_usage(provider_plan.synthesis, provider_plan.synthesis, fallback)

            stages.append(
                TraceStage(
                    name="parallel_candidate_generation",
                    status=Status.COMPLETED,
                    model=provider_plan.synthesis,
                    started_at=start_parallel,
                    ended_at=datetime.now(timezone.utc),
                    artifacts={
                        "difficulty_profile": difficulty_profile,
                        "candidates_requested": len(prompt_variants),
                        "candidates_completed": len(generated),
                        "candidate_drafts": len(candidate_drafts),
                        "agreement": ensemble_agreement,
                    },
                )
            )
            return SynthesisExecutionResult(
                spent_usd=spent_usd,
                synthesis_failures=synthesis_failures,
                draft_texts=draft_texts,
                draft_usage=draft_usage,
                ensemble_agreement=ensemble_agreement,
            )
        if _question_has_answer_choices(question) and answer_choices:
            option_evidence_packs = _build_option_evidence_packs(
                stem=stem_question,
                answer_choices=answer_choices,
                claim_texts=claim_texts,
                top_k=4,
            )
            option_dossiers = _build_option_dossiers(
                stem=stem_question,
                answer_choices=answer_choices,
                claim_texts=claim_texts,
                section_summaries=section_summaries,
                top_k=4,
            )
            elimination_prompt = _build_mcq_option_elimination_prompt(
                question=stem_question,
                answer_choices=answer_choices,
                claim_texts=claim_texts,
                option_dossiers=option_dossiers,
            )
            elimination_result = generate_text(
                provider_plan.synthesis,
                elimination_prompt,
                max_tokens=min(220, synthesis_token_budget or 220),
            )
            elimination_decisions = _parse_mcq_option_elimination(elimination_result.text or "", answer_choices)
            allowed_labels = [
                label for label in sorted(answer_choices.keys()) if elimination_decisions.get(label, "KEEP") != "ELIMINATE"
            ]
            if not allowed_labels:
                allowed_labels = sorted(answer_choices.keys())
            scorer_prompt = _build_mcq_option_scoring_prompt(
                question=stem_question,
                answer_choices=answer_choices,
                claim_texts=claim_texts,
                claim_clusters=claim_clusters,
                section_summaries=section_summaries,
                option_evidence_packs=option_dossiers,
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
            dossier_pool = {label: row for label, row in option_dossiers.items() if label in allowed_labels}
            blended_pool = {label: row for label, row in blended_scores.items() if label in allowed_labels}
            dossier_selected = _select_option_from_dossiers(dossier_pool)
            if dossier_selected and dossier_selected in answer_choices:
                selected_letter = dossier_selected
            if blended_pool and len({round(row["blended"], 6) for row in blended_pool.values()}) > 1:
                min_margin = float(os.getenv("SPARKIT_MCQ_BLEND_MARGIN", "0.06"))
                min_top_score = float(os.getenv("SPARKIT_MCQ_BLEND_MIN_TOP", "0.02"))
                selected_letter = _select_confident_blended_option(
                    blended_scores=blended_pool,
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
                            "allowed_labels": allowed_labels,
                            "elimination_decisions": elimination_decisions,
                            "elimination_raw_output": (elimination_result.text or "")[:1200],
                            "option_evidence_packs": option_evidence_packs,
                            "option_dossiers": option_dossiers,
                            "dossier_selected": dossier_selected,
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
                    option_evidence_packs=option_dossiers,
                )
                judge_result = generate_text(
                    provider_plan.synthesis, judge_prompt, max_tokens=min(220, synthesis_token_budget or 220)
                )
                selected_letter = _extract_answer_letter(judge_result.text or "")
                if judge_result.success and selected_letter and selected_letter in allowed_labels:
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
                            artifacts={
                                "selected_option": selected_letter,
                                "num_choices": len(answer_choices),
                                "allowed_labels": allowed_labels,
                                "elimination_decisions": elimination_decisions,
                                "option_dossiers": option_dossiers,
                            },
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
            # Guarantee a parseable MCQ output even when provider calls degrade.
            candidate = draft_texts[0] if draft_texts else ""
            if not _extract_answer_letter(candidate):
                lexical_scores = _mcq_lexical_option_scores(
                    answer_choices=answer_choices,
                    claim_texts=claim_texts,
                    section_summaries=section_summaries,
                )
                selected_letter = (
                    max(
                        lexical_scores.items(),
                        key=lambda item: float(item[1].get("lexical", 0.0)),
                    )[0]
                    if lexical_scores
                    else sorted(answer_choices.keys())[0]
                )
                draft = f"<answer>{selected_letter}</answer>"
                draft_texts = [draft]
                _record_gen_usage(provider_plan.synthesis, provider_plan.synthesis, draft)
                stages.append(
                    TraceStage(
                        name="mcq_format_fallback",
                        status=Status.COMPLETED,
                        model="policy",
                        started_at=datetime.now(timezone.utc),
                        ended_at=datetime.now(timezone.utc),
                        artifacts={
                            "selected_option": selected_letter,
                            "reason": "missing_or_unparseable_mcq_answer",
                        },
                    )
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

    return SynthesisExecutionResult(
        spent_usd=spent_usd,
        synthesis_failures=synthesis_failures,
        draft_texts=draft_texts,
        draft_usage=draft_usage,
        ensemble_agreement=ensemble_agreement,
    )


def _run_synthesis_revision_pass(
    *,
    question: str,
    provider_plan: ProviderPlan,
    synthesis_token_budget: int | None,
    synthesis_prompt: str,
    draft_texts: list[str],
    draft_usage: list[ProviderUsage],
    spent_usd: float,
    stages: list[TraceStage],
) -> RevisionExecutionResult:
    if not draft_texts:
        return RevisionExecutionResult(spent_usd=spent_usd, draft_texts=draft_texts)

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
        return RevisionExecutionResult(spent_usd=spent_usd, draft_texts=draft_texts)

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
            return RevisionExecutionResult(spent_usd=spent_usd, draft_texts=draft_texts)

        snapshot = _append_generation_usage(
            spent_usd=spent_usd,
            usage_rows=draft_usage,
            provider=provider_plan.synthesis,
            model=revision.model,
            prompt_text=synthesis_prompt,
            draft_text=revised_text,
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
        return RevisionExecutionResult(spent_usd=snapshot.spent_usd, draft_texts=[revised_text])

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
    return RevisionExecutionResult(spent_usd=spent_usd, draft_texts=draft_texts)


def _finalize_answer_and_quality_gates(
    *,
    context: ExecutionContext,
    evidence: EvidenceState,
    budget: BudgetStateRuntime,
    provider_statuses: list[ProviderStatus],
    missing_keys: list[ProviderStatus],
    aggregate_errors: dict[str, str],
    synthesis_failures: list[str],
    verifier_result: VerificationResult,
    adjusted_claim_conf: dict[str, float],
    ensemble_agreement: float,
    draft_texts: list[str],
    draft_usage: list[ProviderUsage],
    stages: list[TraceStage],
) -> FinalizationResult:
    configured_count = sum(1 for status in provider_statuses if status.configured)
    features = CalibrationFeatures(
        support_coverage=len(evidence.citations) / max(1, len(evidence.base_claim_conf)),
        unsupported_claims=evidence.unsupported_claims,
        contradiction_flags=verifier_result.contradiction_flags,
        provider_config_ratio=configured_count / max(1, len(provider_statuses)),
        ensemble_agreement=ensemble_agreement,
        evidence_count=len(evidence.selected_records),
    )
    answer_conf, calibrated_claims = calibrate_answer(features, adjusted_claim_conf)
    CalibrationStore().upsert_features(
        run_id=context.run_id,
        features=features_to_dict(features),
        answer_confidence=answer_conf,
    )
    claim_confidences = [
        ClaimConfidence(claim_id=claim_id, confidence=confidence) for claim_id, confidence in calibrated_claims.items()
    ]

    uncertainty_reasons: list[str] = []
    if len(evidence.records) < context.min_sources:
        uncertainty_reasons.append("Retrieved source count below requested minimum")
    if aggregate_errors:
        uncertainty_reasons.append("One or more retrieval providers failed or returned sparse results")
    if missing_keys:
        uncertainty_reasons.append(f"Missing API keys for providers: {', '.join(status.provider for status in missing_keys)}")
    if evidence.unsupported_claims:
        uncertainty_reasons.append("Some generated claims lacked direct passage support")
    if synthesis_failures:
        uncertainty_reasons.append("One or more provider generation calls failed; fallback synthesis used")
    if budget.budget_stop_reason:
        uncertainty_reasons.append(budget.budget_stop_reason)
    uncertainty_reasons.extend(verifier_result.notes)

    abstain_reason_codes = _abstain_reasons(
        min_sources=context.min_sources,
        retrieved_count=len(evidence.selected_records),
        support_coverage=features.support_coverage,
        unsupported_claims=evidence.unsupported_claims,
        contradiction_flags=verifier_result.contradiction_flags,
        synthesis_failures=synthesis_failures,
    )
    should_abstain = len(abstain_reason_codes) >= 2
    disable_hard_abstain = _env_bool("SPARKIT_DISABLE_HARD_ABSTAIN", False)

    final_text = (
        draft_texts[0]
        if draft_texts
        else _build_answer_text(
            context.question,
            evidence.claim_texts,
            evidence.unsupported_claims,
            clusters=evidence.claim_clusters,
            section_summaries=evidence.section_summaries,
        )
    )
    if context.mode == Mode.ENSEMBLE.value and draft_texts:
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
                        "unsupported_claims": evidence.unsupported_claims,
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
                        "unsupported_claims": evidence.unsupported_claims,
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

    answer = Answer(
        final_text=final_text,
        answer_confidence=answer_conf,
        claim_confidences=claim_confidences,
        uncertainty_reasons=uncertainty_reasons,
    )
    quality_gates = QualityGates(
        citation_coverage=features.support_coverage,
        unsupported_claims=evidence.unsupported_claims,
        contradiction_flags=verifier_result.contradiction_flags,
    )
    provider_usage = [
        ProviderUsage(
            provider=context.provider_plan.retrieval,
            model="retrieval-service",
            tokens_input=0,
            tokens_output=0,
            cost_usd=budget.retrieval_base_cost_usd,
        ),
        ProviderUsage(
            provider=context.provider_plan.verification,
            model="verifier-v2",
            tokens_input=0,
            tokens_output=0,
            cost_usd=budget.verifier_cost,
        ),
        *draft_usage,
    ]
    if budget.brave_request_count > 0:
        provider_usage.append(
            ProviderUsage(
                provider="brave_web",
                model="search-api",
                tokens_input=0,
                tokens_output=0,
                cost_usd=budget.retrieval_brave_cost_usd,
            )
        )

    return FinalizationResult(answer=answer, quality_gates=quality_gates, provider_usage=provider_usage)


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

    # Second pass: fill remaining slots by relevance + novelty to avoid near-duplicates.
    mmr_lambda = _env_float("SPARKIT_INGESTION_DIVERSITY_LAMBDA", 0.75, minimum=0.0)
    mmr_lambda = min(1.0, mmr_lambda)
    selected_keys = {(_record_identity(item)) for item in selected}
    while len(selected) < target_docs:
        best_record: LiteratureRecord | None = None
        best_score = float("-inf")
        for record in scored:
            rid = _record_identity(record)
            if rid in selected_keys:
                continue
            base = _record_relevance_score(question, record, boost_terms)
            title_tokens = set(_tokenize(record.title))
            novelty_penalty = 0.0
            if selected:
                max_overlap = 0.0
                for prior in selected:
                    prior_tokens = set(_tokenize(prior.title))
                    if not title_tokens and not prior_tokens:
                        continue
                    overlap = len(title_tokens & prior_tokens) / max(1, len(title_tokens | prior_tokens))
                    if overlap > max_overlap:
                        max_overlap = overlap
                novelty_penalty = max_overlap
            mmr_score = (mmr_lambda * base) - ((1.0 - mmr_lambda) * novelty_penalty)
            if mmr_score > best_score:
                best_score = mmr_score
                best_record = record
        if best_record is None:
            break
        selected.append(best_record)
        selected_keys.add(_record_identity(best_record))
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


def _difficulty_signals(
    *,
    min_sources: int,
    selected_records_count: int,
    unsupported_claims: int,
    total_claims: int,
    contradiction_flags: int,
    retrieval_error_count: int,
) -> tuple[float, dict[str, float], str]:
    retrieval_sparsity = 1.0 - min(1.0, selected_records_count / max(1, min_sources))
    unsupported_ratio = min(1.0, unsupported_claims / max(1, total_claims))
    contradiction_ratio = min(1.0, contradiction_flags / 6.0)
    retrieval_error_ratio = 1.0 if retrieval_error_count > 0 else 0.0
    score = (
        0.35 * retrieval_sparsity
        + 0.25 * unsupported_ratio
        + 0.25 * contradiction_ratio
        + 0.15 * retrieval_error_ratio
    )
    threshold = _env_float("SPARKIT_DIFFICULTY_HARD_THRESHOLD", 0.45, minimum=0.0)
    profile = "hard" if score >= threshold else "easy"
    signals = {
        "retrieval_sparsity": retrieval_sparsity,
        "unsupported_ratio": unsupported_ratio,
        "contradiction_ratio": contradiction_ratio,
        "retrieval_error_ratio": retrieval_error_ratio,
        "hard_threshold": threshold,
    }
    return score, signals, profile


def _estimate_answer_confidence(
    *,
    provider_statuses: list[ProviderStatus],
    citations_count: int,
    base_claim_conf_count: int,
    unsupported_claims: int,
    contradiction_flags: int,
    ensemble_agreement: float,
    evidence_count: int,
    adjusted_claim_conf: dict[str, float],
) -> float:
    configured_count = sum(1 for status in provider_statuses if status.configured)
    features = CalibrationFeatures(
        support_coverage=citations_count / max(1, base_claim_conf_count),
        unsupported_claims=unsupported_claims,
        contradiction_flags=contradiction_flags,
        provider_config_ratio=configured_count / max(1, len(provider_statuses)),
        ensemble_agreement=ensemble_agreement,
        evidence_count=evidence_count,
    )
    answer_conf, _ = calibrate_answer(features, adjusted_claim_conf)
    return answer_conf


def _should_trigger_confidence_retry(
    *,
    question: str,
    draft_texts: list[str],
    synthesis_failures: list[str],
    provisional_confidence: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    min_conf = _env_float("SPARKIT_CONFIDENCE_RETRY_MIN_CONFIDENCE", 0.55, minimum=0.0)
    if provisional_confidence < min_conf:
        reasons.append(f"low_confidence<{min_conf:.2f}")
    if synthesis_failures:
        reasons.append("synthesis_failures_present")
    if _question_has_answer_choices(question):
        candidate = draft_texts[0] if draft_texts else ""
        if not _extract_answer_letter(candidate):
            reasons.append("mcq_answer_letter_missing")
    return (len(reasons) > 0), reasons


def _select_best_parallel_draft(question: str, drafts: list[str]) -> str:
    if not drafts:
        return ""
    anchors = _extract_lexical_anchors(question, max_items=10)
    ranked = sorted(
        drafts,
        key=lambda text: (_anchor_coverage(text, anchors), len(text)),
        reverse=True,
    )
    return ranked[0]


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
    # Accept a few explicit non-XML formats, but avoid matching incidental letters in prose.
    explicit = re.search(
        r"(?:^|\n|\b)(?:final\s+answer|answer|selected|choice|option)\s*[:\-]?\s*([A-N])\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).upper()
    single_line = re.match(r"^\s*([A-N])\s*$", text.strip(), flags=re.IGNORECASE)
    if single_line:
        return single_line.group(1).upper()
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


def _build_option_dossiers(
    stem: str,
    answer_choices: dict[str, str],
    claim_texts: list[str],
    section_summaries: list[dict[str, str]] | None = None,
    top_k: int = 4,
) -> dict[str, dict[str, object]]:
    corpus_snippets = list(claim_texts)
    if section_summaries:
        corpus_snippets.extend(row.get("summary", "") for row in section_summaries if row.get("summary"))
    choice_tokens = {label: set(_tokenize(text)) for label, text in answer_choices.items()}
    stem_tokens = set(_tokenize(stem))
    dossiers: dict[str, dict[str, object]] = {}

    for label, choice in sorted(answer_choices.items()):
        target = choice_tokens.get(label, set())
        scored_support: list[tuple[float, str]] = []
        scored_counter: list[tuple[float, str]] = []
        for snippet in corpus_snippets:
            snippet_tokens = set(_tokenize(snippet))
            if not snippet_tokens:
                continue
            own_overlap = len(target & snippet_tokens)
            stem_overlap = len(stem_tokens & snippet_tokens)
            other_overlap = max(
                (len(tokens & snippet_tokens) for other, tokens in choice_tokens.items() if other != label),
                default=0,
            )
            support_score = (2.0 * own_overlap) + (0.8 * stem_overlap) - (1.2 * other_overlap)
            if support_score > 0.2:
                scored_support.append((support_score, snippet))
            elif own_overlap == 0 and other_overlap > 0:
                scored_counter.append((float(other_overlap), snippet))
        support_snippets = [text for _, text in sorted(scored_support, key=lambda item: item[0], reverse=True)[:top_k]]
        counter_snippets = [text for _, text in sorted(scored_counter, key=lambda item: item[0], reverse=True)[:top_k]]
        dossier_score = sum(score for score, _ in sorted(scored_support, key=lambda item: item[0], reverse=True)[:top_k])
        dossiers[label] = {
            "choice": choice,
            "support_snippets": support_snippets,
            "counter_snippets": counter_snippets,
            "dossier_score": dossier_score,
        }
    return dossiers


def _select_option_from_dossiers(
    dossiers: dict[str, dict[str, object]],
    *,
    min_top_score: float = 2.0,
    min_margin: float = 1.5,
) -> str | None:
    if not dossiers:
        return None
    ranking = sorted(
        ((label, float(row.get("dossier_score", 0.0))) for label, row in dossiers.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranking:
        return None
    top_label, top_score = ranking[0]
    if top_score < min_top_score:
        return None
    if len(ranking) == 1:
        return top_label
    margin = top_score - ranking[1][1]
    if margin < min_margin:
        return None
    return top_label


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


def _candidate_option_labels_for_falsification(
    stem: str,
    answer_choices: dict[str, str],
    max_options: int = 2,
) -> list[str]:
    if not answer_choices:
        return []
    stem_tokens = set(_tokenize(stem))
    ranked: list[tuple[float, str]] = []
    for label, choice in sorted(answer_choices.items()):
        tokens = set(_tokenize(choice))
        overlap = len(tokens & stem_tokens)
        specificity = len(tokens)
        score = (2.0 * overlap) + (0.2 * specificity)
        ranked.append((score, label))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    cap = max(1, max_options)
    return [label for _, label in ranked[:cap]]


def _build_falsification_queries(
    stem: str,
    answer_choices: dict[str, str],
    segments: list[str],
    max_items: int = 10,
) -> list[str]:
    queries: list[str] = []
    if answer_choices:
        max_options = _env_int("SPARKIT_FALSIFICATION_MAX_OPTIONS", 2, minimum=1)
        selected_labels = _candidate_option_labels_for_falsification(stem, answer_choices, max_options=max_options)
        for label in selected_labels:
            choice = answer_choices.get(label, "")
            if not choice.strip():
                continue
            queries.append(f"{stem} evidence against {choice}")
            queries.append(f"{stem} why {choice} is incorrect")
            queries.append(f"{choice} contradictory findings")
    else:
        for segment in segments[:3]:
            if not segment.strip():
                continue
            queries.append(f"{segment} contradictory findings")
            queries.append(f"{segment} failed replication")
            queries.append(f"{segment} null results")
    return _dedupe_queries(queries, max_items=max_items)


def _semantic_rerank_records(
    question: str,
    records: list[LiteratureRecord],
    provider: str,
    top_k: int,
) -> list[LiteratureRecord]:
    if not records:
        return []
    capped = records[: max(1, min(len(records), _env_int("SPARKIT_SEMANTIC_RERANK_CANDIDATES", 18, minimum=3)))]
    lines = []
    for idx, rec in enumerate(capped, start=1):
        abstract = " ".join((rec.abstract or "").split())
        if len(abstract) > 420:
            abstract = abstract[:420] + "..."
        lines.append(f"{idx}. {rec.title} | {abstract}")
    prompt = (
        "Rank the following papers by how directly they answer the STEM question.\n"
        "Return only one line in this format: indices: i1 | i2 | i3 ...\n"
        f"Question: {question}\n\n"
        "Papers:\n"
        + "\n".join(lines)
    )
    result = generate_text(provider, prompt, max_tokens=180)
    if not result.success or not result.text.strip():
        return capped[:top_k]
    match = re.search(r"indices\s*:\s*(.+)", result.text, flags=re.IGNORECASE)
    raw = match.group(1) if match else result.text
    ordered_idxs: list[int] = []
    for piece in raw.split("|"):
        token = piece.strip()
        if not token:
            continue
        number = re.search(r"\b(\d+)\b", token)
        if not number:
            continue
        idx = int(number.group(1))
        if 1 <= idx <= len(capped) and idx not in ordered_idxs:
            ordered_idxs.append(idx)
    ranked: list[LiteratureRecord] = []
    for idx in ordered_idxs:
        ranked.append(capped[idx - 1])
        if len(ranked) >= top_k:
            break
    if len(ranked) < top_k:
        seen = {_record_identity(item) for item in ranked}
        for rec in capped:
            rid = _record_identity(rec)
            if rid in seen:
                continue
            ranked.append(rec)
            seen.add(rid)
            if len(ranked) >= top_k:
                break
    return ranked


def _semantic_rerank_enabled_for_stage(stage_name: str) -> bool:
    raw = os.getenv(
        "SPARKIT_SEMANTIC_RERANK_STAGES",
        "retrieval_round_2_gap_fill,retrieval_round_3_adversarial,retrieval_round_4_falsification",
    )
    stage_set = {item.strip() for item in raw.split(",") if item.strip()}
    return stage_name in stage_set


def _build_claim_gap_queries(
    question: str,
    stage_name: str,
    records: list[LiteratureRecord],
    planning_provider: str,
    max_items: int = 4,
) -> list[str]:
    if not records:
        return []
    evidence_lines: list[str] = []
    for idx, rec in enumerate(records[:6], start=1):
        abstract = " ".join((rec.abstract or "").split())
        if len(abstract) > 260:
            abstract = abstract[:260] + "..."
        evidence_lines.append(f"{idx}. {rec.title} | {abstract}")
    prompt = (
        "You are planning the next retrieval step for STEM QA.\n"
        "Given the question and current evidence, infer unresolved claim gaps.\n"
        "Return one line only:\n"
        "queries: query1 | query2 | query3 | query4\n"
        "Rules: concise, technical, scholarly-search friendly, no explanations.\n\n"
        f"Question: {question}\n"
        f"Current stage: {stage_name}\n"
        "Evidence snippets:\n"
        + "\n".join(evidence_lines)
    )
    result = generate_text(planning_provider, prompt, max_tokens=220)
    if not result.success or not result.text.strip():
        return _dedupe_queries(
            [
                f"{question} unresolved mechanism evidence",
                f"{question} contradictory findings",
                f"{question} benchmark comparison",
            ],
            max_items=max_items,
        )
    match = re.search(r"queries\s*:\s*(.+)", result.text, flags=re.IGNORECASE)
    blob = match.group(1) if match else result.text
    items = [item.strip() for item in blob.split("|") if item.strip()]
    return _dedupe_queries(items, max_items=max_items)


def _should_inject_claim_gap(
    *,
    stage_idx: int,
    total_stages: int,
    new_unique_docs: int,
    stage_avg_relevance: float,
    elapsed_s: float,
    spent_usd: float,
    max_latency_s: int | None,
    max_cost_usd: float,
) -> tuple[bool, str]:
    if stage_idx >= total_stages:
        return False, "no_next_stage"
    if not _env_bool("SPARKIT_ENABLE_CLAIM_GAP_LOOP", True):
        return False, "disabled"

    force = _env_bool("SPARKIT_CLAIM_GAP_FORCE", False)
    require_low = _env_bool("SPARKIT_CLAIM_GAP_REQUIRE_LOW_EVIDENCE", True)
    min_new_docs_trigger = _env_int("SPARKIT_CLAIM_GAP_MIN_NEW_DOCS_TRIGGER", 2, minimum=0)
    min_relevance_trigger = _env_float("SPARKIT_CLAIM_GAP_MIN_RELEVANCE_TRIGGER", 1.2, minimum=0.0)
    max_cost_ratio = _env_float("SPARKIT_CLAIM_GAP_MAX_COST_RATIO", 0.7, minimum=0.0)
    max_latency_ratio = _env_float("SPARKIT_CLAIM_GAP_MAX_LATENCY_RATIO", 0.7, minimum=0.0)

    if not force:
        if max_cost_usd > 0 and (spent_usd / max_cost_usd) >= max_cost_ratio:
            return False, "cost_headroom_low"
        if max_latency_s and max_latency_s > 0 and (elapsed_s / max_latency_s) >= max_latency_ratio:
            return False, "latency_headroom_low"

    low_evidence = (new_unique_docs <= min_new_docs_trigger) or (stage_avg_relevance <= min_relevance_trigger)
    if require_low and not low_evidence and not force:
        return False, "evidence_sufficient"
    return True, "inject"


def _extract_chemistry_entities(question: str, max_items: int = 6) -> list[str]:
    # Capture high-signal chemistry entities (formulae, hyphenated compounds, and key suffix terms).
    formulae = re.findall(r"\b(?:[A-Z][a-z]?\d*){2,}\b", question)
    hyphenated = re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+){1,}\b", question)
    suffix_terms = re.findall(r"\b[A-Za-z][A-Za-z0-9-]*(?:ane|ene|yne|ol|one|acid|amide|ester|ketone)\b", question, flags=re.IGNORECASE)
    candidates = [*formulae, *hyphenated, *suffix_terms]
    out: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        normalized = token.strip(".,;:()[]{}").lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(token.strip(".,;:()[]{}"))
        if len(out) >= max_items:
            break
    return out


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
        "factcheck": _dedupe_queries([f"{lead} canonical mechanism", f"{lead} textbook explanation"], max_items=4),
    }
    for segment in segments[1:3]:
        intent_queries["primary"] = _dedupe_queries([*intent_queries["primary"], f"{segment} evidence"], max_items=8)
        intent_queries["methods"] = _dedupe_queries([*intent_queries["methods"], f"{segment} methods"], max_items=6)
    if option_queries:
        intent_queries["primary"] = _dedupe_queries([*intent_queries["primary"], *option_queries], max_items=10)
    chemistry_mode = (research_plan.task_type if research_plan else _infer_task_type(question)) == "mechanism" or "reaction" in stem.lower()
    if chemistry_mode:
        chem_entities = _extract_chemistry_entities(stem, max_items=6)
        for entity in chem_entities:
            intent_queries["primary"] = _dedupe_queries(
                [*intent_queries["primary"], f"{entity} mechanism evidence", f"{entity} product selectivity"],
                max_items=10,
            )
            intent_queries["methods"] = _dedupe_queries(
                [*intent_queries["methods"], f"{entity} catalyst conditions", f"{entity} reaction protocol"],
                max_items=8,
            )
            intent_queries["adversarial"] = _dedupe_queries(
                [*intent_queries["adversarial"], f"{entity} contradictory mechanism", f"{entity} failed replication"],
                max_items=8,
            )
            intent_queries["factcheck"] = _dedupe_queries(
                [*intent_queries["factcheck"], f"{entity} standard mechanism", f"{entity} established pathway"],
                max_items=6,
            )
    focus_terms = _dedupe_queries([*anchors, *segments, *answer_choices.values()], max_items=14)
    intent_queries = _enforce_intent_query_quotas(
        stem=stem,
        answer_choices=answer_choices,
        segments=segments,
        intent_queries=intent_queries,
    )
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


def _enforce_intent_query_quotas(
    *,
    stem: str,
    answer_choices: dict[str, str],
    segments: list[str],
    intent_queries: dict[str, list[str]],
) -> dict[str, list[str]]:
    lead = segments[0] if segments else stem
    max_primary = _env_int("SPARKIT_INTENT_PRIMARY_MAX", 10, minimum=4)
    max_methods = _env_int("SPARKIT_INTENT_METHODS_MAX", 8, minimum=2)
    max_reference = _env_int("SPARKIT_INTENT_REFERENCE_MAX", 6, minimum=2)
    max_adversarial = _env_int("SPARKIT_INTENT_ADVERSARIAL_MAX", 8, minimum=2)
    max_factcheck = _env_int("SPARKIT_INTENT_FACTCHECK_MAX", 6, minimum=2)
    max_options = _env_int("SPARKIT_INTENT_OPTIONS_MAX", 8, minimum=2)

    min_primary = _env_int("SPARKIT_INTENT_PRIMARY_MIN", 3, minimum=1)
    min_methods = _env_int("SPARKIT_INTENT_METHODS_MIN", 2, minimum=1)
    min_reference = _env_int("SPARKIT_INTENT_REFERENCE_MIN", 2, minimum=1)
    min_adversarial = _env_int("SPARKIT_INTENT_ADVERSARIAL_MIN", 2, minimum=1)
    min_factcheck = _env_int("SPARKIT_INTENT_FACTCHECK_MIN", 2, minimum=1)
    min_options = _env_int("SPARKIT_INTENT_OPTIONS_MIN", 2, minimum=1) if answer_choices else 0

    primary = _dedupe_queries(intent_queries.get("primary", []), max_items=max_primary)
    methods = _dedupe_queries(intent_queries.get("methods", []), max_items=max_methods)
    reference = _dedupe_queries(intent_queries.get("reference", []), max_items=max_reference)
    adversarial = _dedupe_queries(intent_queries.get("adversarial", []), max_items=max_adversarial)
    factcheck = _dedupe_queries(intent_queries.get("factcheck", []), max_items=max_factcheck)
    options = _dedupe_queries(intent_queries.get("options", []), max_items=max_options)

    if len(primary) < min_primary:
        primary = _dedupe_queries(
            [*primary, f"{lead} primary evidence", f"{lead} consensus evidence", f"{lead} key findings"],
            max_items=max_primary,
        )
    if len(methods) < min_methods:
        methods = _dedupe_queries(
            [*methods, f"{lead} methods", f"{lead} protocol details", f"{lead} experimental design"],
            max_items=max_methods,
        )
    if len(reference) < min_reference:
        reference = _dedupe_queries(
            [*reference, f"{lead} systematic review", f"{lead} meta-analysis", f"{lead} review article"],
            max_items=max_reference,
        )
    if len(adversarial) < min_adversarial:
        adversarial = _dedupe_queries(
            [*adversarial, f"{lead} contradictory findings", f"{lead} failed replication", f"{lead} negative results"],
            max_items=max_adversarial,
        )
    if len(factcheck) < min_factcheck:
        factcheck = _dedupe_queries(
            [*factcheck, f"{lead} canonical mechanism", f"{lead} established pathway", f"{lead} textbook explanation"],
            max_items=max_factcheck,
        )
    if answer_choices and len(options) < min_options:
        options = _dedupe_queries(
            [*options, *_build_option_hypothesis_queries(stem, answer_choices, max_items=max_options)],
            max_items=max_options,
        )

    if options:
        primary = _dedupe_queries([*primary, *options], max_items=max_primary)

    return {
        "primary": primary,
        "options": options,
        "methods": methods,
        "adversarial": adversarial,
        "reference": reference,
        "factcheck": factcheck,
    }


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
        "queries_factcheck: query1 | query2\n\n"
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
        "factcheck": [],
    }
    key_map = {
        "queries_primary": "primary",
        "queries_options": "options",
        "queries_methods": "methods",
        "queries_adversarial": "adversarial",
        "queries_reference": "reference",
        "queries_factcheck": "factcheck",
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
    for intent in ("primary", "options", "methods", "adversarial", "reference", "factcheck"):
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

    merged_intents = _enforce_intent_query_quotas(
        stem=stem,
        answer_choices=answer_choices,
        segments=segments or fallback.segments,
        intent_queries=merged_intents,
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
            ("retrieval_factcheck", plan.intent_queries.get("factcheck", [])),
            ("retrieval_methods", plan.intent_queries["methods"]),
            ("retrieval_adversarial", plan.intent_queries["adversarial"]),
            ("retrieval_reference", plan.intent_queries["reference"]),
        ]
    option_queries = _dedupe_queries(plan.intent_queries.get("options", []), max_items=8)
    rounds = [
        ("retrieval_round_1", primary_queries),
        ("retrieval_round_option_hypotheses", option_queries),
        ("retrieval_round_factcheck", plan.intent_queries.get("factcheck", [])),
        ("retrieval_round_2_gap_fill", _dedupe_queries([*plan.intent_queries["methods"], *plan.intent_queries["reference"]], max_items=8)),
        ("retrieval_round_3_adversarial", plan.intent_queries["adversarial"]),
    ]
    if _env_bool("SPARKIT_ENABLE_FALSIFICATION_ROUND", True):
        rounds.append(
            (
                "retrieval_round_4_falsification",
                _build_falsification_queries(
                    stem=question,
                    answer_choices=plan.answer_choices,
                    segments=plan.segments,
                    max_items=_env_int("SPARKIT_FALSIFICATION_MAX_QUERIES", 10, minimum=4),
                ),
            )
        )
    return rounds


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
    option_evidence_packs: dict[str, object] | None = None,
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
        support = " | ".join((pack.get("support_snippets", []) if isinstance(pack, dict) else pack)[:3]) if pack else "no support snippets"
        counter = " | ".join((pack.get("counter_snippets", []) if isinstance(pack, dict) else [])[:2]) if isinstance(pack, dict) else ""
        option_pack_lines.append(f"- {label}. {text}\n  support: {support}\n  counter: {counter or 'none'}")
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
    option_evidence_packs: dict[str, object] | None = None,
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
        support = " | ".join((pack.get("support_snippets", []) if isinstance(pack, dict) else pack)[:3]) if pack else "no support snippets"
        counter = " | ".join((pack.get("counter_snippets", []) if isinstance(pack, dict) else [])[:2]) if isinstance(pack, dict) else ""
        option_pack_lines.append(f"- {label}. {text}\n  support: {support}\n  counter: {counter or 'none'}")
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


def _build_mcq_option_elimination_prompt(
    question: str,
    answer_choices: dict[str, str],
    claim_texts: list[str],
    option_dossiers: dict[str, object] | None = None,
) -> str:
    stem, _ = _split_question_and_choices(question)
    choices_block = "\n".join(f"{label}. {text}" for label, text in sorted(answer_choices.items()))
    evidence_lines = "\n".join(f"- {claim}" for claim in claim_texts[:12]) or "- No evidence lines available."
    dossier_lines: list[str] = []
    for label, text in sorted(answer_choices.items()):
        row = (option_dossiers or {}).get(label, {})
        if isinstance(row, dict):
            support = " | ".join((row.get("support_snippets") or [])[:2]) or "none"
            counter = " | ".join((row.get("counter_snippets") or [])[:2]) or "none"
            dossier_lines.append(f"- {label}. {text}\n  support: {support}\n  counter: {counter}")
    dossier_block = "\n".join(dossier_lines) or "- No option dossiers available."
    return (
        "You are a strict STEM MCQ eliminator.\n"
        "Mark each option as KEEP or ELIMINATE based only on evidence.\n"
        "If uncertain, prefer KEEP over ELIMINATE.\n"
        "Return exactly one line per option in this format:\n"
        "A: KEEP\n"
        "B: ELIMINATE\n"
        "No extra text.\n\n"
        f"Question stem: {stem}\n"
        "Answer choices:\n"
        f"{choices_block}\n\n"
        "Option dossiers:\n"
        f"{dossier_block}\n"
        "Evidence:\n"
        f"{evidence_lines}\n"
    )


def _parse_mcq_option_elimination(text: str, answer_choices: dict[str, str]) -> dict[str, str]:
    decisions: dict[str, str] = {}
    if not text.strip():
        return decisions
    for label in sorted(answer_choices.keys()):
        match = re.search(rf"(?:^|\n)\s*{re.escape(label)}\s*:\s*(KEEP|ELIMINATE)\b", text, flags=re.IGNORECASE)
        if not match:
            continue
        decisions[label] = match.group(1).upper()
    return decisions


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
    config = OrchestrationConfig.from_inputs(
        min_sources=min_sources,
        providers=providers,
        mode=mode,
        max_latency_s=max_latency_s,
        max_cost_usd=max_cost_usd,
        synthesis_max_tokens=synthesis_max_tokens,
        prompt_version=prompt_version,
        config_version=config_version,
        reproducibility=reproducibility,
    )
    mode = config.mode
    max_latency_s = config.max_latency_s
    max_cost_usd = config.max_cost_usd
    ingestion_max_chars = int(os.getenv("SPARKIT_INGESTION_MAX_CHARS", "10000"))
    observability = RunObservability(run_id=run_id)

    provider_list = config.providers
    provider_statuses = build_default_registry().resolve(provider_list)
    missing_keys = [status for status in provider_statuses if not status.configured]
    task_type = _infer_task_type(question)
    provider_plan = build_provider_plan(
        mode=config.mode,
        statuses=provider_statuses,
        requested=provider_list,
        task_type=task_type,
    )
    effort = _effort_profile(mode=config.mode, question=question, min_sources=config.min_sources)
    synthesis_token_budget = (
        config.synthesis_max_tokens if config.synthesis_max_tokens is not None else effort.synthesis_max_tokens
    )
    research_plan: ResearchPlan | None = None
    retrieval_plan: RetrievalPlan | None = None
    if config.mode == Mode.RESEARCH_MAX.value:
        research_plan = _decompose_question(question, provider_plan.planning)
    retrieval_plan = _decompose_retrieval(question, provider_plan.planning, research_plan)
    rounds = _build_round_queries_from_plan(config.mode, question, retrieval_plan)
    adaptive = _adaptive_retrieval_config(len(rounds))
    stages: list[TraceStage] = []
    _append_plan_stages(
        stages=stages,
        started=started,
        question=question,
        config=config,
        provider_plan=provider_plan,
        provider_statuses=provider_statuses,
        effort=effort,
        rounds=rounds,
        ingestion_max_chars=ingestion_max_chars,
        synthesis_token_budget=synthesis_token_budget,
        research_plan=research_plan,
        retrieval_plan=retrieval_plan,
        adaptive=adaptive,
    )
    retrieval_result = _run_retrieval_rounds(
        started=started,
        question=question,
        rounds=rounds,
        effort=effort,
        provider_plan=provider_plan,
        retrieval_plan=retrieval_plan,
        stages=stages,
        observability=observability,
        max_latency_s=max_latency_s,
        max_cost_usd=max_cost_usd,
    )
    records_by_round = retrieval_result.records_by_round
    aggregate_errors = retrieval_result.aggregate_errors
    spent_usd = retrieval_result.spent_usd
    retrieval_base_cost_usd = retrieval_result.retrieval_base_cost_usd
    retrieval_brave_cost_usd = retrieval_result.retrieval_brave_cost_usd
    brave_request_count = retrieval_result.brave_request_count
    budget_stop_reason = retrieval_result.budget_stop_reason
    live_web_result = _run_live_web_tool_loop(
        question=question,
        records=retrieval_result.all_records,
        planning_provider=provider_plan.planning,
        retrieval_provider=provider_plan.retrieval,
        max_results=max(4, effort.retrieval_min_results // 2),
        spent_usd=spent_usd,
        retrieval_base_cost_usd=retrieval_base_cost_usd,
        retrieval_brave_cost_usd=retrieval_brave_cost_usd,
        brave_request_count=brave_request_count,
        stages=stages,
        observability=observability,
    )
    spent_usd = live_web_result.spent_usd
    retrieval_base_cost_usd = live_web_result.retrieval_base_cost_usd
    retrieval_brave_cost_usd = live_web_result.retrieval_brave_cost_usd
    brave_request_count = live_web_result.brave_request_count
    aggregate_errors.update(live_web_result.aggregate_errors)

    records = _dedupe_records(live_web_result.all_records)
    selected_records = _select_records_for_ingestion(
        question=question,
        records=records,
        target_docs=effort.ingestion_target_docs,
        boost_terms=retrieval_plan.focus_terms if retrieval_plan else None,
    )
    final_semantic_rerank = _env_bool(
        "SPARKIT_ENABLE_SEMANTIC_RERANK_FINAL",
        _env_bool("SPARKIT_ENABLE_SEMANTIC_RERANK", False),
    )
    if final_semantic_rerank:
        selected_records = _semantic_rerank_records(
            question=question,
            records=selected_records,
            provider=provider_plan.retrieval,
            top_k=effort.ingestion_target_docs,
        )
        stages.append(
            TraceStage(
                name="retrieval_semantic_rerank",
                status=Status.COMPLETED,
                model=provider_plan.retrieval,
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                artifacts={
                    "enabled": True,
                    "selected_docs_after_rerank": len(selected_records),
                    "candidates_cap": _env_int("SPARKIT_SEMANTIC_RERANK_CANDIDATES", 18, minimum=3),
                },
            )
        )
    evidence_result = _assemble_evidence_and_build_claims(
        run_id=run_id,
        question=question,
        selected_records=selected_records,
        retrieval_plan=retrieval_plan,
        ingestion_max_chars=ingestion_max_chars,
        spent_usd=spent_usd,
    )
    spent_usd = evidence_result.spent_usd
    citations = evidence_result.citations
    claim_texts = evidence_result.claim_texts
    claim_evidence = evidence_result.claim_evidence
    base_claim_conf = evidence_result.base_claim_conf
    unsupported_claims = evidence_result.unsupported_claims

    adversarial_stage_name = "retrieval_adversarial" if mode == Mode.RESEARCH_MAX.value else "retrieval_round_3_adversarial"
    verifier_records = list(records_by_round.get(adversarial_stage_name, []))
    verifier_records.extend(records_by_round.get("retrieval_round_4_falsification", []))
    verifier_records = _dedupe_records(verifier_records)
    verifier_start = datetime.now(timezone.utc)
    depth = contradiction_depth_from_budget(max_cost_usd=max_cost_usd, max_latency_s=max_latency_s) + effort.contradiction_depth_bonus
    verifier_result = run_verifier(
        claim_ids=list(base_claim_conf.keys()),
        adversarial_records=verifier_records,
        depth=depth,
        top_k=5,
    )
    verifier_cost = estimate_stage_cost("verification", units=max(1, depth))
    spent_usd += verifier_cost
    observability.add_stage(
        StageMetric(
            name="verification",
            duration_ms=int((datetime.now(timezone.utc) - verifier_start).total_seconds() * 1000),
            documents_retrieved=len(verifier_records),
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
                "adversarial_stage_name": adversarial_stage_name,
                "falsification_docs_used": len(records_by_round.get("retrieval_round_4_falsification", [])),
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
    difficulty_score, difficulty_breakdown, difficulty_profile = _difficulty_signals(
        min_sources=min_sources,
        selected_records_count=len(selected_records),
        unsupported_claims=unsupported_claims,
        total_claims=len(base_claim_conf),
        contradiction_flags=verifier_result.contradiction_flags,
        retrieval_error_count=len(aggregate_errors),
    )
    synthesis_mode = mode
    if (
        mode == Mode.ROUTED.value
        and difficulty_profile == "hard"
        and _env_bool("SPARKIT_DIFFICULTY_ESCALATE_ROUTED_TO_ENSEMBLE", False)
    ):
        synthesis_mode = Mode.ENSEMBLE.value
    stages.append(
        TraceStage(
            name="difficulty_gate",
            status=Status.COMPLETED,
            model="policy",
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            artifacts={
                "difficulty_score": round(difficulty_score, 4),
                "difficulty_profile": difficulty_profile,
                "signals": difficulty_breakdown,
                "input_mode": mode,
                "synthesis_mode": synthesis_mode,
                "escalation_enabled": _env_bool("SPARKIT_DIFFICULTY_ESCALATE_ROUTED_TO_ENSEMBLE", False),
            },
        )
    )
    synthesis_prompt = _build_synthesis_prompt(
        question,
        claim_texts,
        claim_clusters=claim_clusters,
        section_summaries=section_summaries,
    )
    stem_question, answer_choices = _split_question_and_choices(question)
    synthesis_start = datetime.now(timezone.utc)
    synthesis_result = _run_synthesis_phase(
        started=started,
        mode=synthesis_mode,
        max_latency_s=max_latency_s,
        max_cost_usd=max_cost_usd,
        spent_usd=spent_usd,
        provider_plan=provider_plan,
        question=question,
        stem_question=stem_question,
        answer_choices=answer_choices,
        claim_texts=claim_texts,
        unsupported_claims=unsupported_claims,
        claim_clusters=claim_clusters,
        section_summaries=section_summaries,
        synthesis_prompt=synthesis_prompt,
        synthesis_token_budget=synthesis_token_budget,
        research_plan=research_plan,
        stages=stages,
        difficulty_profile=difficulty_profile,
    )
    spent_usd = synthesis_result.spent_usd
    synthesis_failures = synthesis_result.synthesis_failures
    draft_texts = synthesis_result.draft_texts
    draft_usage = synthesis_result.draft_usage
    ensemble_agreement = synthesis_result.ensemble_agreement

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
        revision_result = _run_synthesis_revision_pass(
            question=question,
            provider_plan=provider_plan,
            synthesis_token_budget=synthesis_token_budget,
            synthesis_prompt=synthesis_prompt,
            draft_texts=draft_texts,
            draft_usage=draft_usage,
            spent_usd=spent_usd,
            stages=stages,
        )
        spent_usd = revision_result.spent_usd
        draft_texts = revision_result.draft_texts

    confidence_retry_enabled = _env_bool("SPARKIT_ENABLE_CONFIDENCE_RETRY", True)
    confidence_retry_max_attempts = _env_int("SPARKIT_CONFIDENCE_RETRY_MAX_ATTEMPTS", 1, minimum=0)
    retry_attempt = 0
    while confidence_retry_enabled and retry_attempt < confidence_retry_max_attempts:
        provisional_conf = _estimate_answer_confidence(
            provider_statuses=provider_statuses,
            citations_count=len(citations),
            base_claim_conf_count=len(base_claim_conf),
            unsupported_claims=unsupported_claims,
            contradiction_flags=verifier_result.contradiction_flags,
            ensemble_agreement=ensemble_agreement,
            evidence_count=len(selected_records),
            adjusted_claim_conf=adjusted_claim_conf,
        )
        should_retry, retry_reasons = _should_trigger_confidence_retry(
            question=question,
            draft_texts=draft_texts,
            synthesis_failures=synthesis_failures,
            provisional_confidence=provisional_conf,
        )
        stages.append(
            TraceStage(
                name="confidence_retry_gate",
                status=Status.COMPLETED,
                model="policy",
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                artifacts={
                    "attempt": retry_attempt + 1,
                    "enabled": confidence_retry_enabled,
                    "triggered": should_retry,
                    "reasons": retry_reasons,
                    "provisional_confidence": round(provisional_conf, 4),
                },
            )
        )
        if not should_retry:
            break

        retry_queries = _build_claim_gap_queries(
            question=question,
            stage_name="confidence_retry",
            records=selected_records,
            planning_provider=provider_plan.planning,
            max_items=_env_int("SPARKIT_CONFIDENCE_RETRY_MAX_QUERIES", 4, minimum=1),
        )
        if not retry_queries:
            break

        retry_stage_start = datetime.now(timezone.utc)
        retry_records: list[LiteratureRecord] = []
        retry_errors: dict[str, str] = {}
        retry_brave_requests = 0
        for query in retry_queries:
            found, errors, stats = search_literature(query, max_results=effort.retrieval_min_results)
            retry_records.extend(found)
            for source, err in errors.items():
                retry_errors[f"{source}:{query}"] = err
            retry_brave_requests += int((stats.get("requests_by_source") or {}).get("brave_web", 0))
        deduped_retry_records = _dedupe_records(retry_records)
        aggregate_errors.update(retry_errors)
        records = _dedupe_records([*records, *deduped_retry_records])
        selected_records = _select_records_for_ingestion(
            question=question,
            records=records,
            target_docs=effort.ingestion_target_docs,
            boost_terms=retrieval_plan.focus_terms if retrieval_plan else None,
        )
        if final_semantic_rerank:
            selected_records = _semantic_rerank_records(
                question=question,
                records=selected_records,
                provider=provider_plan.retrieval,
                top_k=effort.ingestion_target_docs,
            )

        retry_base_cost = estimate_stage_cost("retrieval", units=len(retry_queries))
        retry_brave_cost = estimate_brave_search_cost(retry_brave_requests)
        retry_cost = retry_base_cost + retry_brave_cost
        spent_usd += retry_cost
        retrieval_base_cost_usd += retry_base_cost
        retrieval_brave_cost_usd += retry_brave_cost
        brave_request_count += retry_brave_requests
        observability.add_stage(
            StageMetric(
                name="retrieval_confidence_retry",
                duration_ms=int((datetime.now(timezone.utc) - retry_stage_start).total_seconds() * 1000),
                documents_retrieved=len(deduped_retry_records),
                source_errors=len(retry_errors),
                estimated_cost_usd=retry_cost,
            )
        )
        stages.append(
            TraceStage(
                name="retrieval_confidence_retry",
                status=Status.COMPLETED,
                model=provider_plan.retrieval,
                started_at=retry_stage_start,
                ended_at=datetime.now(timezone.utc),
                artifacts={
                    "attempt": retry_attempt + 1,
                    "queries": retry_queries,
                    "documents_retrieved": len(deduped_retry_records),
                    "source_errors": retry_errors,
                    "brave_requests": retry_brave_requests,
                    "estimated_cost_usd": retry_cost,
                },
            )
        )

        evidence_result = _assemble_evidence_and_build_claims(
            run_id=run_id,
            question=question,
            selected_records=selected_records,
            retrieval_plan=retrieval_plan,
            ingestion_max_chars=ingestion_max_chars,
            spent_usd=spent_usd,
        )
        spent_usd = evidence_result.spent_usd
        citations = evidence_result.citations
        claim_texts = evidence_result.claim_texts
        claim_evidence = evidence_result.claim_evidence
        base_claim_conf = evidence_result.base_claim_conf
        unsupported_claims = evidence_result.unsupported_claims
        claim_clusters = _build_claim_clusters(claim_evidence)
        section_summaries = _build_section_summaries(claim_evidence)

        adversarial_stage = "retrieval_adversarial" if mode == Mode.RESEARCH_MAX.value else "retrieval_round_3_adversarial"
        retry_records_by_round = dict(records_by_round)
        retry_records_by_round[adversarial_stage] = _dedupe_records(
            [*(retry_records_by_round.get(adversarial_stage, [])), *deduped_retry_records]
        )
        verification_result = _run_verification_and_adjust_confidence(
            mode=mode,
            max_cost_usd=max_cost_usd,
            max_latency_s=max_latency_s,
            effort=effort,
            records_by_round=retry_records_by_round,
            base_claim_conf=base_claim_conf,
            spent_usd=spent_usd,
            provider_plan=provider_plan,
            stages=stages,
            observability=observability,
        )
        spent_usd = verification_result.spent_usd
        verifier_cost += verification_result.verifier_cost
        verifier_result = verification_result.verifier_result
        adjusted_claim_conf = verification_result.adjusted_claim_conf
        records_by_round = retry_records_by_round

        synthesis_prompt = _build_synthesis_prompt(
            question,
            claim_texts,
            claim_clusters=claim_clusters,
            section_summaries=section_summaries,
        )
        retry_synth = _run_synthesis_phase(
            started=started,
            mode=synthesis_mode,
            max_latency_s=max_latency_s,
            max_cost_usd=max_cost_usd,
            spent_usd=spent_usd,
            provider_plan=provider_plan,
            question=question,
            stem_question=stem_question,
            answer_choices=answer_choices,
            claim_texts=claim_texts,
            unsupported_claims=unsupported_claims,
            claim_clusters=claim_clusters,
            section_summaries=section_summaries,
            synthesis_prompt=synthesis_prompt,
            synthesis_token_budget=synthesis_token_budget,
            research_plan=research_plan,
            stages=stages,
            difficulty_profile=difficulty_profile,
        )
        spent_usd = retry_synth.spent_usd
        draft_usage.extend(retry_synth.draft_usage)
        draft_texts = retry_synth.draft_texts
        synthesis_failures = retry_synth.synthesis_failures
        ensemble_agreement = retry_synth.ensemble_agreement
        if effort.synthesis_revision_pass and draft_texts:
            retry_revision = _run_synthesis_revision_pass(
                question=question,
                provider_plan=provider_plan,
                synthesis_token_budget=synthesis_token_budget,
                synthesis_prompt=synthesis_prompt,
                draft_texts=draft_texts,
                draft_usage=draft_usage,
                spent_usd=spent_usd,
                stages=stages,
            )
            spent_usd = retry_revision.spent_usd
            draft_texts = retry_revision.draft_texts

        retry_attempt += 1

    execution_context = ExecutionContext(
        run_id=run_id,
        question=question,
        mode=synthesis_mode,
        min_sources=min_sources,
        max_latency_s=max_latency_s,
        max_cost_usd=max_cost_usd,
        provider_plan=provider_plan,
    )
    evidence_state = EvidenceState(
        records=records,
        selected_records=selected_records,
        citations=citations,
        claim_texts=claim_texts,
        claim_evidence=claim_evidence,
        base_claim_conf=base_claim_conf,
        unsupported_claims=unsupported_claims,
        claim_clusters=claim_clusters,
        section_summaries=section_summaries,
    )
    budget_state = BudgetStateRuntime(
        spent_usd=spent_usd,
        budget_stop_reason=budget_stop_reason,
        retrieval_base_cost_usd=retrieval_base_cost_usd,
        retrieval_brave_cost_usd=retrieval_brave_cost_usd,
        brave_request_count=brave_request_count,
        verifier_cost=verifier_cost,
    )
    finalization = _finalize_answer_and_quality_gates(
        context=execution_context,
        evidence=evidence_state,
        budget=budget_state,
        provider_statuses=provider_statuses,
        missing_keys=missing_keys,
        aggregate_errors=aggregate_errors,
        synthesis_failures=synthesis_failures,
        verifier_result=verifier_result,
        adjusted_claim_conf=adjusted_claim_conf,
        ensemble_agreement=ensemble_agreement,
        draft_texts=draft_texts,
        draft_usage=draft_usage,
        stages=stages,
    )
    answer = finalization.answer
    quality_gates = finalization.quality_gates
    provider_usage = finalization.provider_usage

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
