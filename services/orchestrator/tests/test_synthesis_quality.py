from __future__ import annotations

from services.orchestrator.app.engine import (
    ClaimEvidence,
    _abstain_reasons,
    _anchor_coverage,
    _build_claim_clusters,
    _build_section_summaries,
    _build_synthesis_prompt,
    _extract_lexical_anchors,
    _question_has_answer_choices,
    _record_relevance_score,
    _select_records_for_ingestion,
)
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
    )
    assert "retrieved_evidence_too_sparse" in reasons
    assert "citation_coverage_below_threshold" in reasons


def test_answer_choices_detection() -> None:
    assert _question_has_answer_choices("Answer Choices:\nA. Foo\nB. Bar") is True
    assert _question_has_answer_choices("What is the product of this reaction?") is False


def test_lexical_anchor_extraction_and_coverage() -> None:
    question = "(1S,2R)-2-((tert-butyldimethylsilyl)oxy)cyclopent-1-en-1-yl ..."
    anchors = _extract_lexical_anchors(question)
    assert any("cyclopent-1-en-1-yl" in item for item in anchors)
    assert _anchor_coverage("Contains cyclopent-1-en-1-yl term", anchors) > 0.0
