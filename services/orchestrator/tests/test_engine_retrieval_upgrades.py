from __future__ import annotations

from dataclasses import dataclass

from services.retrieval_service.app.models import LiteratureRecord
from services.orchestrator.app.engine import (
    RetrievalPlan,
    _build_retrieval_claim_slots,
    _build_claim_gap_queries,
    _candidate_option_labels_for_falsification,
    _build_falsification_queries,
    _build_round_queries_from_plan,
    _enforce_intent_query_quotas,
    _difficulty_signals,
    _select_best_parallel_draft,
    _marginal_coverage_rerank,
    _should_trigger_confidence_retry,
    _should_inject_claim_gap,
    _semantic_rerank_enabled_for_stage,
)


def test_build_falsification_queries_mcq_prefers_option_specific_queries() -> None:
    queries = _build_falsification_queries(
        stem="Which mechanism best explains X?",
        answer_choices={"A": "pathway alpha", "B": "pathway beta"},
        segments=["segment one", "segment two"],
        max_items=10,
    )
    lowered = " | ".join(queries).lower()
    assert "evidence against pathway alpha" in lowered
    assert "why pathway beta is incorrect" in lowered


def test_build_falsification_queries_non_mcq_uses_segments() -> None:
    queries = _build_falsification_queries(
        stem="Explain catalyst behavior",
        answer_choices={},
        segments=["catalyst behavior", "reaction kinetics"],
        max_items=10,
    )
    lowered = " | ".join(queries).lower()
    assert "catalyst behavior contradictory findings" in lowered
    assert "reaction kinetics failed replication" in lowered


def test_round_plan_includes_falsification_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SPARKIT_ENABLE_FALSIFICATION_ROUND", "1")
    plan = RetrievalPlan(
        segments=["seg1", "seg2"],
        focus_terms=["seg1"],
        intent_queries={
            "primary": ["q primary"],
            "options": ["q option"],
            "methods": ["q methods"],
            "adversarial": ["q adversarial"],
            "reference": ["q ref"],
            "factcheck": ["q factcheck"],
        },
        answer_choices={"A": "option a", "B": "option b"},
    )
    rounds = _build_round_queries_from_plan("single", "question text", plan)
    stage_names = [name for name, _ in rounds]
    assert "retrieval_round_4_falsification" in stage_names


def test_candidate_option_labels_for_falsification_limits_to_top_two() -> None:
    labels = _candidate_option_labels_for_falsification(
        stem="Which pathway drives glycolysis regulation in liver cells?",
        answer_choices={
            "A": "glycolysis regulation pathway",
            "B": "liver glycolysis control",
            "C": "marine plankton fluorescence",
        },
        max_options=2,
    )
    assert len(labels) == 2
    assert "C" not in labels


def test_semantic_rerank_enabled_for_stage_default_targets() -> None:
    assert _semantic_rerank_enabled_for_stage("retrieval_round_2_gap_fill") is True
    assert _semantic_rerank_enabled_for_stage("retrieval_round_3_adversarial") is True
    assert _semantic_rerank_enabled_for_stage("retrieval_round_4_falsification") is True
    assert _semantic_rerank_enabled_for_stage("retrieval_round_1") is False


@dataclass
class _FakeResult:
    success: bool
    text: str


def test_build_claim_gap_queries_parses_planner_output(monkeypatch) -> None:
    import services.orchestrator.app.engine as engine_mod

    monkeypatch.setattr(
        engine_mod,
        "generate_text",
        lambda *_args, **_kwargs: _FakeResult(
            success=True,
            text="queries: targeted pathway replication | contradictory mechanism evidence | control benchmark assay",
        ),
    )
    records = [
        LiteratureRecord(
            title="A paper",
            url="https://example.org/a",
            source="test",
            abstract="Mechanism discussion.",
            doi=None,
            year=2024,
        )
    ]
    queries = _build_claim_gap_queries(
        question="What explains catalytic behavior?",
        stage_name="retrieval_round_1",
        records=records,
        planning_provider="openai",
        max_items=4,
    )
    assert len(queries) >= 2
    assert any("replication" in item.lower() for item in queries)


def test_build_claim_gap_queries_has_fallback_when_planner_fails(monkeypatch) -> None:
    import services.orchestrator.app.engine as engine_mod

    monkeypatch.setattr(
        engine_mod,
        "generate_text",
        lambda *_args, **_kwargs: _FakeResult(success=False, text=""),
    )
    records = [
        LiteratureRecord(
            title="B paper",
            url="https://example.org/b",
            source="test",
            abstract="Another mechanism discussion.",
            doi=None,
            year=2023,
        )
    ]
    queries = _build_claim_gap_queries(
        question="What explains catalytic behavior?",
        stage_name="retrieval_round_2_gap_fill",
        records=records,
        planning_provider="openai",
        max_items=4,
    )
    assert len(queries) >= 1
    assert any("contradictory findings" in item.lower() for item in queries)


def test_should_inject_claim_gap_requires_low_evidence_by_default() -> None:
    allowed, reason = _should_inject_claim_gap(
        stage_idx=1,
        total_stages=4,
        new_unique_docs=5,
        stage_avg_relevance=3.0,
        elapsed_s=10.0,
        spent_usd=0.2,
        max_latency_s=None,
        max_cost_usd=3.0,
    )
    assert allowed is False
    assert reason == "evidence_sufficient"


def test_should_inject_claim_gap_blocks_when_cost_headroom_low() -> None:
    allowed, reason = _should_inject_claim_gap(
        stage_idx=1,
        total_stages=4,
        new_unique_docs=0,
        stage_avg_relevance=0.5,
        elapsed_s=10.0,
        spent_usd=2.5,
        max_latency_s=None,
        max_cost_usd=3.0,
    )
    assert allowed is False
    assert reason == "cost_headroom_low"


def test_should_inject_claim_gap_allows_when_low_evidence_and_headroom() -> None:
    allowed, reason = _should_inject_claim_gap(
        stage_idx=1,
        total_stages=4,
        new_unique_docs=0,
        stage_avg_relevance=0.5,
        elapsed_s=10.0,
        spent_usd=0.2,
        max_latency_s=None,
        max_cost_usd=3.0,
    )
    assert allowed is True
    assert reason == "inject"


def test_difficulty_signals_marks_hard_when_evidence_is_sparse_and_contradictory() -> None:
    score, signals, profile = _difficulty_signals(
        min_sources=8,
        selected_records_count=2,
        unsupported_claims=3,
        total_claims=4,
        contradiction_flags=4,
        retrieval_error_count=1,
    )
    assert profile == "hard"
    assert score > 0.45
    assert signals["retrieval_error_ratio"] == 1.0


def test_difficulty_signals_marks_easy_with_strong_evidence() -> None:
    score, _signals, profile = _difficulty_signals(
        min_sources=5,
        selected_records_count=8,
        unsupported_claims=0,
        total_claims=8,
        contradiction_flags=0,
        retrieval_error_count=0,
    )
    assert profile == "easy"
    assert score < 0.2


def test_build_retrieval_claim_slots_falls_back_to_segments_and_options(monkeypatch) -> None:
    import services.orchestrator.app.engine as engine_mod

    monkeypatch.setattr(
        engine_mod,
        "generate_text",
        lambda *_args, **_kwargs: _FakeResult(success=False, text=""),
    )
    plan = RetrievalPlan(
        segments=["kinase activation mechanism", "feedback inhibition in pathway"],
        focus_terms=["kinase", "feedback"],
        intent_queries={"primary": [], "options": [], "methods": [], "adversarial": [], "reference": [], "factcheck": []},
        answer_choices={"A": "upregulation", "B": "downregulation"},
    )
    slots = _build_retrieval_claim_slots(
        question="What is correct?\nAnswer Choices:\nA. upregulation\nB. downregulation",
        retrieval_plan=plan,
        planning_provider="openai",
        max_items=6,
    )
    joined = " | ".join(slots).lower()
    assert "kinase activation mechanism" in joined
    assert "evidence for upregulation" in joined


def test_marginal_coverage_rerank_prefers_additional_slot_coverage() -> None:
    records = [
        LiteratureRecord(
            title="ERK phosphorylation evidence in MAPK signaling",
            url="https://example.org/a",
            source="test",
            abstract="ERK phosphorylation confirms downstream MAPK activation.",
            doi=None,
            year=2024,
        ),
        LiteratureRecord(
            title="MEK inhibitor data and RAF pathway suppression",
            url="https://example.org/b",
            source="test",
            abstract="MEK inhibition suppresses RAF-MEK-ERK cascade with inhibitor specificity.",
            doi=None,
            year=2023,
        ),
        LiteratureRecord(
            title="General background on cell signaling",
            url="https://example.org/c",
            source="test",
            abstract="Broad overview without mechanistic details.",
            doi=None,
            year=2022,
        ),
    ]
    ranked = _marginal_coverage_rerank(
        question="Which mechanism explains MAPK inhibition response?",
        records=records,
        claim_slots=["ERK phosphorylation evidence", "MEK inhibitor specificity"],
        top_k=2,
        boost_terms=["MAPK", "inhibition"],
    )
    assert len(ranked) == 2
    titles = " | ".join(item.title for item in ranked)
    assert "ERK phosphorylation evidence" in titles
    assert "MEK inhibitor data" in titles


def test_confidence_retry_triggers_for_low_confidence() -> None:
    should_retry, reasons = _should_trigger_confidence_retry(
        question="What is the best answer?",
        draft_texts=["A detailed answer"],
        synthesis_failures=[],
        provisional_confidence=0.2,
    )
    assert should_retry is True
    assert any("low_confidence" in reason for reason in reasons)


def test_confidence_retry_triggers_for_mcq_missing_letter() -> None:
    should_retry, reasons = _should_trigger_confidence_retry(
        question="Question stem\nAnswer choices:\nA. one\nB. two",
        draft_texts=["No xml answer tag present"],
        synthesis_failures=[],
        provisional_confidence=0.9,
    )
    assert should_retry is True
    assert "mcq_answer_letter_missing" in reasons


def test_confidence_retry_skips_for_high_confidence_clean_output() -> None:
    should_retry, reasons = _should_trigger_confidence_retry(
        question="Question stem\nAnswer choices:\nA. one\nB. two",
        draft_texts=["<answer>B</answer>"],
        synthesis_failures=[],
        provisional_confidence=0.95,
    )
    assert should_retry is False
    assert reasons == []


def test_select_best_parallel_draft_prefers_anchor_coverage() -> None:
    best = _select_best_parallel_draft(
        "Describe MAPK phosphorylation cascade in detail",
        [
            "A broad answer with little specificity.",
            "MAPK phosphorylation cascade proceeds via RAF-MEK-ERK with caveats.",
        ],
    )
    assert "MAPK phosphorylation cascade" in best


def test_enforce_intent_query_quotas_fills_missing_intents() -> None:
    quotas = _enforce_intent_query_quotas(
        stem="How does kinase inhibition change signaling?",
        answer_choices={},
        segments=["kinase inhibition signaling"],
        intent_queries={"primary": [], "options": [], "methods": [], "adversarial": [], "reference": [], "factcheck": []},
    )
    assert len(quotas["primary"]) >= 1
    assert len(quotas["methods"]) >= 1
    assert len(quotas["reference"]) >= 1
    assert len(quotas["adversarial"]) >= 1
    assert len(quotas["factcheck"]) >= 1
