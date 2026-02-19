from __future__ import annotations

from services.orchestrator.app.verifier import run_verifier
from services.retrieval_service.app.models import LiteratureRecord


def test_verifier_flags_contradictions_and_ranks() -> None:
    records = [
        LiteratureRecord(
            source="semantic_scholar",
            title="A contradictory finding in GNN chemistry",
            abstract="This work reports inconsistent and negative result trends.",
            authors=["A"],
            year=2024,
            doi="10.1/x",
            url="https://example.org/1",
        ),
        LiteratureRecord(
            source="crossref",
            title="Mixed evidence for benchmark outcomes",
            abstract="Null result appears in replicated settings.",
            authors=["B"],
            year=2023,
            doi="10.1/y",
            url="https://example.org/2",
        ),
    ]
    result = run_verifier(claim_ids=["clm_1", "clm_2"], adversarial_records=records, depth=2, top_k=2)
    assert result.contradiction_flags >= 1
    assert "clm_1" in result.penalties
    assert result.notes
    assert len(result.ranked_contradictions) >= 1
