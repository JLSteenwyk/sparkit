from __future__ import annotations

from services.orchestrator.app.engine import (
    ClaimEvidence,
    _build_claim_clusters,
    _build_section_summaries,
    _build_synthesis_prompt,
)


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
