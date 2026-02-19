from __future__ import annotations

from services.orchestrator.app.versioning import build_reproducibility_record


def test_reproducibility_record_is_deterministic_for_same_inputs() -> None:
    a = build_reproducibility_record(
        question="What is superconducting qubit decoherence?",
        mode="routed",
        providers=["openai", "anthropic"],
        constraints={"max_latency_s": 120, "max_cost_usd": 3.0, "min_sources": 5},
        prompt_version="p1",
        config_version="c1",
    )
    b = build_reproducibility_record(
        question="What is superconducting qubit decoherence?",
        mode="routed",
        providers=["anthropic", "openai"],
        constraints={"max_latency_s": 120, "max_cost_usd": 3.0, "min_sources": 5},
        prompt_version="p1",
        config_version="c1",
    )

    # Created at may differ but fingerprint should remain stable for equivalent inputs.
    assert a["question_hash"] == b["question_hash"]
    assert a["providers"] == b["providers"]
    assert a["fingerprint"] == b["fingerprint"]
