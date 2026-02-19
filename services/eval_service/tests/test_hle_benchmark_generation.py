from __future__ import annotations

from services.eval_service.app.hle_benchmark_generation import build_hle_biochem_subset


def _row(idx: int, category: str, subject: str, question: str) -> dict:
    return {
        "id": f"src_{idx}",
        "question": question,
        "category": category,
        "raw_subject": subject,
        "answer_type": "multipleChoice",
    }


def test_build_hle_subset_counts_and_domains() -> None:
    rows = []
    for i in range(12):
        rows.append(_row(i, "Biology/Medicine", "Biology", f"Biology question {i} about proteins and pathways"))
    for i in range(12, 24):
        rows.append(_row(i, "Chemistry", "Chemistry", f"Chemistry question {i} about catalysts and kinetics"))

    subset = build_hle_biochem_subset(rows, bio_count=10, chem_count=10, seed=11)
    assert len(subset) == 20
    assert sum(1 for item in subset if item["domain"] == "biology") == 10
    assert sum(1 for item in subset if item["domain"] == "chemistry") == 10
    assert all(item["required_keywords"] for item in subset)
    assert all(item["must_have_citations"] == 2 for item in subset)


def test_build_hle_subset_is_deterministic_for_seed() -> None:
    rows = []
    for i in range(15):
        rows.append(_row(i, "Biology/Medicine", "Genetics", f"Bio q {i} with genomes and variants"))
    for i in range(15, 30):
        rows.append(_row(i, "Chemistry", "Organic Chemistry", f"Chem q {i} with synthesis and reagents"))

    first = build_hle_biochem_subset(rows, bio_count=10, chem_count=10, seed=3)
    second = build_hle_biochem_subset(rows, bio_count=10, chem_count=10, seed=3)
    assert [item["source_id"] for item in first] == [item["source_id"] for item in second]

