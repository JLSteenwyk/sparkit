# Architecture (V1)

Last updated: 2026-02-20

## Goal
Build a STEM-focused literature agent that answers complex questions through multi-round investigation, citation-grounded synthesis, and calibrated confidence scoring.

## Services
1. API Gateway
- Receives requests and persists run state.
- Attaches prompt/config versions and run reproducibility metadata.

2. Orchestrator
- Executes multi-round retrieval pipeline.
- Applies mode-aware provider routing (`single`, `routed`, `ensemble`).
- Runs verifier and calibration stages.
- Executes provider-backed synthesis generation with claim clustering and section-aware summaries.
- Enforces budget policy guards.

3. Retrieval Service
- Federated literature search (arXiv, Crossref, Semantic Scholar).

4. Ingestion Service
- Downloads/parses HTML and PDF artifacts to structured sections.

5. Evidence Store
- Persists documents, passages, claims, claim-evidence links.

6. Calibration Store
- Persists calibration feature vectors and calibrated answer confidence.

7. Observability Store
- Persists per-run stage telemetry, estimated costs, and budget-stop reasons.

8. Eval Service
- Runs benchmark evaluation with rubric scoring and calibration metrics (ECE/Brier).

## Current workflow
1. Plan + provider readiness analysis.
   - includes prompt/config version metadata and stable reproducibility fingerprint payload
2. Retrieval rounds:
   - broad
   - gap-fill
   - adversarial
3. Ingest and parse evidence documents.
4. Build claim/evidence graph and citation links.
5. Run verifier pass with depth-aware contradiction reranking.
6. Calibrate answer and claim confidences from quality features.
7. Run provider synthesis:
   - single/routed: one synthesis provider
   - ensemble: multiple provider drafts + adjudication
   - prompt includes claim clusters + section-aware evidence summaries
8. Enforce budget guardrails during retrieval/synthesis.
9. Persist run observability metrics.
10. Return final answer, citations, quality gates, and trace artifacts.
    - synthesis artifacts include `claim_clusters` and `section_summaries`
11. Evaluate outputs via benchmark harness (rubric + ECE/Brier).
12. Persist reproducibility metadata (prompt/config versions + stable fingerprint) with each run.
13. Capture baseline manifests and per-run predictions for configured provider modes.

Note:
- Benchmark default presets currently use `single_*` and `routed_frontier`; `ensemble_frontier` is deprecated from standard benchmark workflows.

## Guardrails
- Unsupported claims are counted in quality gates.
- Contradiction flags feed verifier penalties and calibration.
- Missing provider keys are surfaced in uncertainty reasons.
- Provider generation failures trigger deterministic fallback synthesis.
- Cost/latency budget guards can stop rounds early and report reason.
