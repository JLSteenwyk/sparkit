from __future__ import annotations

from services.orchestrator.app.engine import (
    ClaimEvidence,
    _avg_relevance,
    _abstain_reasons,
    _anchor_coverage,
    _build_mcq_option_judge_prompt,
    _build_mcq_option_elimination_prompt,
    _build_mcq_option_scoring_prompt,
    _build_claim_clusters,
    _build_option_evidence_packs,
    _build_option_dossiers,
    _build_round_queries_from_plan,
    _build_section_summaries,
    _build_synthesis_prompt,
    _evidence_consensus_profile,
    _extract_answer_letter,
    _has_discriminative_option_scores,
    _mcq_lexical_option_scores,
    _mcq_selected_option_evidence_gate,
    _parse_mcq_option_scores,
    _parse_mcq_option_elimination,
    _extract_lexical_anchors,
    _record_identity,
    _env_bool,
    _heuristic_retrieval_plan,
    _question_has_answer_choices,
    _record_relevance_score,
    _record_source_quality_score,
    _question_domain,
    _domain_mcq_guidance,
    _select_confident_blended_option,
    _select_best_section_chunk,
    _select_option_from_dossiers,
    _select_records_for_ingestion,
    _split_question_and_choices,
    _apply_mcq_option_rescue,
)
from shared.schemas.domain import Status, TraceStage
from datetime import datetime, timezone
from services.retrieval_service.app.models import LiteratureRecord


def _evidence() -> list[ClaimEvidence]:
    return [
        ClaimEvidence(
            claim_id="c1",
            claim_text="Paper A reports improved transmon coherence from materials controls.",
            title="Transmon coherence improvements from dielectric engineering",
            year=2024,
            section_name="Abstract",
            section_text="We report improved transmon coherence using low-loss interfaces. Secondary detail.",
        ),
        ClaimEvidence(
            claim_id="c2",
            claim_text="Paper B benchmarks flux-noise mitigation under pulse shaping.",
            title="Flux-noise mitigation and benchmarking in superconducting qubits",
            year=2023,
            section_name="Results",
            section_text="Benchmark experiments show reduced dephasing after pulse shaping.",
        ),
        ClaimEvidence(
            claim_id="c3",
            claim_text="Paper C discusses readout tradeoffs and limitations.",
            title="Readout fidelity tradeoffs in transmon architectures",
            year=2022,
            section_name="Discussion",
            section_text="Limitations include readout overhead and sensitivity to drift.",
        ),
    ]


def test_claim_clusters_are_built_with_labels_and_counts() -> None:
    clusters = _build_claim_clusters(_evidence(), max_clusters=3)
    assert clusters
    assert all("label" in cluster for cluster in clusters)
    assert all("count" in cluster for cluster in clusters)
    assert sum(int(cluster["count"]) for cluster in clusters) == 3


def test_section_summaries_are_bucketed_by_section_type() -> None:
    sections = _build_section_summaries(_evidence(), max_sections=4)
    names = {row["section"] for row in sections}
    assert "overview" in names
    assert "results" in names
    assert "discussion" in names


def test_synthesis_prompt_includes_clusters_and_sections() -> None:
    evidence = _evidence()
    clusters = _build_claim_clusters(evidence, max_clusters=2)
    sections = _build_section_summaries(evidence, max_sections=2)

    prompt = _build_synthesis_prompt(
        question="What drives transmon decoherence improvements?",
        claim_texts=[item.claim_text for item in evidence],
        claim_clusters=clusters,
        section_summaries=sections,
    )
    assert "Claim clusters:" in prompt
    assert "Section-aware summaries:" in prompt
    assert "Evidence:" in prompt


def test_record_relevance_prefers_overlap_in_title_and_abstract() -> None:
    question = "transmon decoherence flux noise mitigation"
    high = LiteratureRecord(
        source="arxiv",
        title="Flux noise mitigation for transmon decoherence",
        abstract="We benchmark transmon decoherence mitigation under flux noise controls.",
        year=2024,
        url="https://example.com/high",
    )
    low = LiteratureRecord(
        source="crossref",
        title="Unrelated colloid aggregation study",
        abstract="Polymer phase behavior in complex fluids.",
        year=2024,
        url="https://example.com/low",
    )
    assert _record_relevance_score(question, high) > _record_relevance_score(question, low)


def test_select_records_for_ingestion_balances_source_diversity() -> None:
    question = "gene expression perturbation benchmark methods"
    records = [
        LiteratureRecord(source="arxiv", title="A", abstract="gene expression benchmark", year=2024, url="https://x/a"),
        LiteratureRecord(source="arxiv", title="B", abstract="gene expression perturbation", year=2023, url="https://x/b"),
        LiteratureRecord(source="crossref", title="C", abstract="benchmark methods", year=2022, url="https://x/c"),
    ]
    selected = _select_records_for_ingestion(question, records, target_docs=2)
    sources = {record.source for record in selected}
    assert len(selected) == 2
    assert len(sources) == 2


def test_abstain_reasons_trigger_on_sparse_low_support_profile() -> None:
    reasons = _abstain_reasons(
        min_sources=6,
        retrieved_count=2,
        support_coverage=0.2,
        unsupported_claims=3,
        contradiction_flags=4,
        synthesis_failures=["empty output"],
        independent_hq_sources=1,
        consensus_score=0.25,
    )
    assert "retrieved_evidence_too_sparse" in reasons
    assert "citation_coverage_below_threshold" in reasons
    assert "insufficient_independent_high_quality_sources" in reasons
    assert "evidence_consensus_weak" in reasons


def test_answer_choices_detection() -> None:
    assert _question_has_answer_choices("Answer Choices:\nA. Foo\nB. Bar") is True
    assert _question_has_answer_choices("What is the product of this reaction?") is False


def test_lexical_anchor_extraction_and_coverage() -> None:
    question = "(1S,2R)-2-((tert-butyldimethylsilyl)oxy)cyclopent-1-en-1-yl ..."
    anchors = _extract_lexical_anchors(question)
    assert any("cyclopent-1-en-1-yl" in item for item in anchors)
    assert _anchor_coverage("Contains cyclopent-1-en-1-yl term", anchors) > 0.0


def test_retrieval_plan_builds_intent_queries() -> None:
    question = "What catalytic steps control enantioselective bromination in styrene derivatives?"
    plan = _heuristic_retrieval_plan(question)
    rounds = _build_round_queries_from_plan(mode="single", question=question, plan=plan)
    assert plan.intent_queries["primary"]
    assert plan.intent_queries["methods"]
    assert len(rounds) >= 4
    assert rounds[0][0] == "retrieval_round_1"


def test_select_best_section_chunk_prefers_focus_term_overlap() -> None:
    sections = [
        ("intro", "This paper provides broad context on chemistry history and theory."),
        (
            "results",
            "The key step is bromination of styrene with NBS under catalytic control and enantioselective induction.",
        ),
    ]
    heading, chunk = _select_best_section_chunk(
        question="Which catalytic mechanism explains enantioselective bromination of styrene?",
        sections=sections,
        focus_terms=["enantioselective bromination", "styrene", "catalytic mechanism"],
    )
    assert heading == "results"
    assert "bromination" in chunk.lower()


def test_split_question_and_choices_parses_mcq_structure() -> None:
    question = (
        "What happens to Braveheart expression?\n\n"
        "Answer Choices:\n"
        "A. increases in both\n"
        "B. decreases in both\n"
        "C. unchanged\n"
    )
    stem, choices = _split_question_and_choices(question)
    assert stem == "What happens to Braveheart expression?"
    assert choices["A"] == "increases in both"
    assert choices["B"] == "decreases in both"
    assert choices["C"] == "unchanged"


def test_heuristic_retrieval_plan_includes_option_queries_for_mcq() -> None:
    question = (
        "Air-stable radicals in OLED have what key disadvantage?\n\n"
        "Answer Choices:\n"
        "A. oxygen instability\n"
        "B. wide FWHM due to multiple emissions\n"
        "C. low luminance from radical quenching\n"
    )
    plan = _heuristic_retrieval_plan(question)
    assert plan.answer_choices
    assert plan.intent_queries["options"]
    joined = " ".join(plan.intent_queries["options"]).lower()
    assert "wide fwhm due to multiple emissions" in joined
    assert all("answer choices" not in segment.lower() for segment in plan.segments)
    rounds = _build_round_queries_from_plan(mode="single", question=question, plan=plan)
    assert any(name == "retrieval_round_option_hypotheses" for name, _ in rounds)


def test_extract_answer_letter_prefers_xml_tag() -> None:
    assert _extract_answer_letter("<answer>B</answer>") == "B"
    assert _extract_answer_letter("Final: D") is None
    assert _extract_answer_letter("No option") is None


def test_mcq_option_judge_prompt_contains_choices_and_evidence() -> None:
    prompt = _build_mcq_option_judge_prompt(
        question="Which choice is correct?",
        answer_choices={"A": "up", "B": "down"},
        claim_texts=["Study reports up-regulation in both cohorts."],
        claim_clusters=[{"label": "expression", "count": 1, "sample_claims": ["up-regulation"]}],
        section_summaries=[{"section": "results", "summary": "Both cohorts increased."}],
    )
    assert "Answer choices:" in prompt
    assert "A. up" in prompt
    assert "B. down" in prompt
    assert "Evidence:" in prompt
    assert "Domain guidance:" in prompt


def test_parse_mcq_option_scores_extracts_numeric_rows() -> None:
    text = "A: support=0.20, contradiction=0.80\nB: support=0.75, contradiction=0.10"
    scores = _parse_mcq_option_scores(text, {"A": "x", "B": "y"})
    assert scores["A"]["support"] == 0.2
    assert scores["A"]["contradiction"] == 0.8
    assert round(scores["B"]["net"], 2) == 0.65


def test_mcq_option_rescue_overrides_when_low_confidence_and_margin_is_strong(monkeypatch) -> None:
    monkeypatch.setenv("SPARKIT_ENABLE_MCQ_OPTION_RESCUE", "1")
    monkeypatch.setenv("SPARKIT_MCQ_RESCUE_MIN_CONFIDENCE", "0.62")
    monkeypatch.setenv("SPARKIT_MCQ_RESCUE_MIN_MARGIN", "0.04")
    question = "Which is correct?\nAnswer Choices:\nA. alpha\nB. beta\nC. gamma"
    scorer_stage = TraceStage(
        name="mcq_option_scorer",
        status=Status.COMPLETED,
        model="test",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        artifacts={
            "allowed_labels": ["A", "B", "C"],
            "blended_scores": {
                "A": {"blended": 0.01, "net": 0.02},
                "B": {"blended": 0.31, "net": 0.33},
                "C": {"blended": 0.10, "net": 0.09},
            },
        },
    )
    text, artifacts = _apply_mcq_option_rescue(
        question=question,
        final_text="<answer>A</answer>",
        answer_confidence=0.40,
        contradiction_flags=0,
        stages=[scorer_stage],
    )
    assert text == "<answer>B</answer>"
    assert artifacts["rescue_triggered"] is True
    assert artifacts["rescue_applied"] is True
    assert artifacts["selected_option"] == "B"
    assert artifacts["rescue_margin"] > 0.04


def test_mcq_option_rescue_skips_when_gate_not_triggered(monkeypatch) -> None:
    monkeypatch.setenv("SPARKIT_ENABLE_MCQ_OPTION_RESCUE", "1")
    monkeypatch.setenv("SPARKIT_MCQ_RESCUE_MIN_CONFIDENCE", "0.62")
    question = "Which is correct?\nAnswer Choices:\nA. alpha\nB. beta"
    scorer_stage = TraceStage(
        name="mcq_option_scorer",
        status=Status.COMPLETED,
        model="test",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        artifacts={
            "allowed_labels": ["A", "B"],
            "blended_scores": {
                "A": {"blended": 0.20, "net": 0.22},
                "B": {"blended": 0.10, "net": 0.08},
            },
        },
    )
    text, artifacts = _apply_mcq_option_rescue(
        question=question,
        final_text="<answer>A</answer>",
        answer_confidence=0.90,
        contradiction_flags=0,
        stages=[scorer_stage],
    )
    assert text == "<answer>A</answer>"
    assert artifacts["rescue_triggered"] is False
    assert artifacts["rescue_applied"] is False


def test_mcq_selected_option_evidence_gate_passes_with_strong_support(monkeypatch) -> None:
    monkeypatch.setenv("SPARKIT_ENABLE_MCQ_EVIDENCE_GATE", "1")
    stage = TraceStage(
        name="mcq_option_scorer",
        status=Status.COMPLETED,
        model="test",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        artifacts={
            "option_dossiers": {
                "A": {
                    "support_snippets": ["s1", "s2"],
                    "counter_snippets": [],
                    "dossier_score": 2.4,
                }
            },
            "option_scores": {"A": {"net": 0.15}},
        },
    )
    passed, artifacts = _mcq_selected_option_evidence_gate(
        question="Q?\nAnswer Choices:\nA. alpha\nB. beta",
        final_text="<answer>A</answer>",
        stages=[stage],
    )
    assert passed is True
    assert artifacts["reason"] == "ok"


def test_mcq_selected_option_evidence_gate_fails_with_weak_support(monkeypatch) -> None:
    monkeypatch.setenv("SPARKIT_ENABLE_MCQ_EVIDENCE_GATE", "1")
    stage = TraceStage(
        name="mcq_option_scorer",
        status=Status.COMPLETED,
        model="test",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        artifacts={
            "option_dossiers": {
                "A": {
                    "support_snippets": [],
                    "counter_snippets": ["c1"],
                    "dossier_score": 0.4,
                }
            },
            "option_scores": {"A": {"net": -0.2}},
        },
    )
    passed, artifacts = _mcq_selected_option_evidence_gate(
        question="Q?\nAnswer Choices:\nA. alpha\nB. beta",
        final_text="<answer>A</answer>",
        stages=[stage],
    )
    assert passed is False
    assert artifacts["reason"] == "insufficient_support_for_selected_option"


def test_mcq_option_scoring_prompt_has_required_format() -> None:
    prompt = _build_mcq_option_scoring_prompt(
        question="Which choice is correct?",
        answer_choices={"A": "up", "B": "down"},
        claim_texts=["Evidence says up in both."],
    )
    assert "A: support=0.00, contradiction=0.00" in prompt
    assert "Answer choices:" in prompt
    assert "Evidence:" in prompt
    assert "Domain guidance:" in prompt


def test_question_domain_heuristics() -> None:
    assert _question_domain("What catalyst controls enantioselective bromination mechanism?") == "chemistry"
    assert _question_domain("Which gene expression change drives immune cell differentiation?") == "biology_medicine"
    assert _question_domain("What is the best benchmark design?") == "general_stem"


def test_domain_mcq_guidance_returns_nonempty() -> None:
    text = _domain_mcq_guidance("What catalyst controls enantioselective bromination?")
    assert isinstance(text, str)
    assert len(text) > 10


def test_mcq_option_elimination_prompt_and_parser() -> None:
    prompt = _build_mcq_option_elimination_prompt(
        question="Which choice is correct?",
        answer_choices={"A": "up", "B": "down"},
        claim_texts=["Evidence supports up-regulation."],
        option_dossiers={
            "A": {"support_snippets": ["up-regulation observed"], "counter_snippets": []},
            "B": {"support_snippets": [], "counter_snippets": ["contradicted by evidence"]},
        },
    )
    assert "A: KEEP" in prompt
    assert "B: ELIMINATE" in prompt
    parsed = _parse_mcq_option_elimination("A: KEEP\nB: ELIMINATE", {"A": "x", "B": "y"})
    assert parsed["A"] == "KEEP"
    assert parsed["B"] == "ELIMINATE"


def test_option_score_discrimination_guard() -> None:
    assert _has_discriminative_option_scores({}) is False
    flat = {
        "A": {"support": 0.0, "contradiction": 0.0, "net": 0.0},
        "B": {"support": 0.0, "contradiction": 0.0, "net": 0.0},
    }
    assert _has_discriminative_option_scores(flat) is False
    varied = {
        "A": {"support": 0.2, "contradiction": 0.6, "net": -0.4},
        "B": {"support": 0.7, "contradiction": 0.1, "net": 0.6},
    }
    assert _has_discriminative_option_scores(varied) is True


def test_option_dossiers_capture_support_and_counter() -> None:
    dossiers = _build_option_dossiers(
        stem="What is the key disadvantage?",
        answer_choices={
            "A": "oxygen instability",
            "B": "wide fwhm due to multiple emissions",
            "C": "low luminance from quenching",
        },
        claim_texts=[
            "Radical OLED systems show broad FWHM and multiple emission bands.",
            "The molecule is air stable and not oxygen unstable.",
        ],
        section_summaries=[{"section": "results", "summary": "Broad FWHM dominates line shape."}],
    )
    assert dossiers["B"]["support_snippets"]
    assert isinstance(dossiers["A"]["counter_snippets"], list)


def test_select_option_from_dossiers_requires_margin() -> None:
    dossiers = {
        "A": {"dossier_score": 1.0},
        "B": {"dossier_score": 4.0},
        "C": {"dossier_score": 1.5},
    }
    assert _select_option_from_dossiers(dossiers, min_top_score=2.0, min_margin=1.0) == "B"
    assert _select_option_from_dossiers(dossiers, min_top_score=5.0, min_margin=1.0) is None


def test_record_identity_prefers_doi_then_url() -> None:
    doi_record = LiteratureRecord(
        source="crossref",
        title="DOI record",
        abstract="a",
        year=2024,
        doi="10.1000/XYZ",
        url="https://example.org/a",
    )
    url_record = LiteratureRecord(
        source="arxiv",
        title="URL record",
        abstract="b",
        year=2024,
        url="https://example.org/B",
    )
    assert _record_identity(doi_record) == "doi:10.1000/xyz"
    assert _record_identity(url_record) == "url:https://example.org/b"


def test_avg_relevance_scores_nonempty_records() -> None:
    question = "flux noise transmon mitigation"
    records = [
        LiteratureRecord(
            source="arxiv",
            title="Flux noise mitigation for transmon systems",
            abstract="We improve transmon coherence under flux noise.",
            year=2024,
            url="https://example.org/r1",
        ),
        LiteratureRecord(
            source="crossref",
            title="Unrelated topic",
            abstract="No overlap",
            year=2024,
            url="https://example.org/r2",
        ),
    ]
    assert _avg_relevance(question, records) > 0.0
    assert _avg_relevance(question, []) == 0.0


def test_source_quality_prefers_high_trust_hosts_and_methods() -> None:
    high = LiteratureRecord(
        source="crossref",
        title="Randomized prospective cohort benchmark in chemistry kinetics",
        abstract="Systematic review with replication and mechanistic analysis.",
        year=2025,
        url="https://www.nature.com/articles/example",
    )
    low = LiteratureRecord(
        source="brave_web",
        title="Personal blog opinion",
        abstract="Thoughts and commentary",
        year=2020,
        url="https://random-blog.example.com/post",
    )
    assert _record_source_quality_score(high) > _record_source_quality_score(low)


def test_evidence_consensus_profile_detects_multi_source_hq_cluster() -> None:
    records = [
        LiteratureRecord(
            source="crossref",
            title="Enantioselective bromination in styrene reaction mechanism",
            abstract="Mechanistic kinetics benchmark.",
            year=2024,
            url="https://www.nature.com/articles/a",
        ),
        LiteratureRecord(
            source="europe_pmc",
            title="Enantioselective bromination in styrene reaction mechanism",
            abstract="Replication with NMR and kinetics.",
            year=2023,
            url="https://www.ncbi.nlm.nih.gov/pubmed/b",
        ),
    ]
    profile = _evidence_consensus_profile(records)
    assert profile["independent_hq_sources"] >= 2.0
    assert profile["consensus_score"] > 0.4


def test_env_bool_parses_truthy_and_falsy(monkeypatch) -> None:
    monkeypatch.setenv("SPARKIT_TEST_BOOL", "true")
    assert _env_bool("SPARKIT_TEST_BOOL", False) is True
    monkeypatch.setenv("SPARKIT_TEST_BOOL", "0")
    assert _env_bool("SPARKIT_TEST_BOOL", True) is False


def test_mcq_lexical_option_scores_handles_fwdh_alias() -> None:
    scores = _mcq_lexical_option_scores(
        answer_choices={"B": "wide FWDH because it has multiple emissions", "D": "low EQE because excitons are quenched"},
        claim_texts=["The radical OLED shows broad FWHM and multiple emission bands."],
    )
    assert scores["B"]["lexical"] > scores["D"]["lexical"]


def test_build_option_evidence_packs_targets_choice_overlap() -> None:
    packs = _build_option_evidence_packs(
        stem="Braveheart expression in ESC and differentiating heart cells",
        answer_choices={"A": "increases in both cell types", "B": "decreases in both cell types"},
        claim_texts=[
            "Braveheart increases in embryonic stem cells and cardiomyocyte differentiation stages.",
            "Another unrelated chemistry statement.",
        ],
    )
    assert packs["A"]


def test_select_confident_blended_option_requires_margin() -> None:
    low_margin = {
        "A": {"blended": 0.10, "net": 0.0, "lexical": 0.33},
        "B": {"blended": 0.08, "net": 0.0, "lexical": 0.26},
    }
    assert _select_confident_blended_option(low_margin, min_margin=0.05, min_top_score=0.02) is None

    high_margin = {
        "A": {"blended": 0.15, "net": 0.1, "lexical": 0.30},
        "B": {"blended": 0.03, "net": 0.0, "lexical": 0.10},
    }
    assert _select_confident_blended_option(high_margin, min_margin=0.05, min_top_score=0.02) == "A"
