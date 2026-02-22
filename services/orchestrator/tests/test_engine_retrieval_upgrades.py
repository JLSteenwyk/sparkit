from __future__ import annotations

from dataclasses import dataclass

from services.retrieval_service.app.models import LiteratureRecord
from services.orchestrator.app.engine import (
    RetrievalPlan,
    _build_claim_gap_queries,
    _candidate_option_labels_for_falsification,
    _build_falsification_queries,
    _build_round_queries_from_plan,
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
