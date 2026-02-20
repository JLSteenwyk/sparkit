# Backlog

Last updated: 2026-02-20

## Priority 0 (next)
1. Cost-accurate accounting + latency instrumentation.
Impact: High. Effort: Low-Medium.
Deliverable: exact per-provider pricing map, stage-level token+latency capture, and cost-confidence flags in reports.
2. Answerability gating.
Impact: High. Effort: Low-Medium.
Deliverable: policy gate that returns structured insufficient-evidence outcomes when retrieval/verifier quality is below threshold.
3. Complete HLE149 routed run and publish SPARKIT vs direct comparison table.
Impact: High. Effort: Low.
Deliverable: final HLE149 comparison across `single_openai`, `single_anthropic`, `routed_frontier`, and direct baselines.

## Priority 1
1. Evidence scoring/filtering before synthesis.
Impact: High. Effort: Medium.
Deliverable: relevance/trust/novelty/contradiction ranker with balanced top-k evidence pack.
2. Query planner for multi-query decomposition.
Impact: High. Effort: Medium.
Deliverable: planner stage that emits focused sub-queries by method/domain/contradiction.
3. Draft -> critic -> revise synthesis loop.
Impact: Medium-High. Effort: Medium.
Deliverable: two-pass synthesis with structured critique and constrained revision.
4. Policy-learned routed execution.
Impact: Medium-High. Effort: Medium-High.
Deliverable: data-driven provider routing policy based on benchmark history.

## Priority 2
1. Per-domain prompt packs and rubric weighting.
Impact: Medium. Effort: Medium.
Deliverable: chemistry/biology/etc. prompt templates and domain-aware scoring configuration.
2. Benchmark expansion + stratified reporting.
Impact: Medium. Effort: Low-Medium.
Deliverable: larger benchmark slices with confidence intervals and per-domain rollups.
3. Retriever index/cache layer.
Impact: Medium. Effort: High.
Deliverable: local index/cache for repeated query acceleration and hybrid retrieval.
4. Automated regression gates in CI/CD.
Impact: Medium. Effort: Low.
Deliverable: hard fail on rubric/calibration/cost regressions versus baseline manifests.
5. Add scheduled benchmark + drift checks on larger subsets in CI/nightly workflow.
Impact: Medium. Effort: Low-Medium.
Deliverable: automated recurring capture + drift reports.
6. Add per-domain drift breakdown in checker output.
Impact: Medium. Effort: Medium.
Deliverable: drift checker emits domain-level regression deltas.
7. Add alert delivery integration (Slack/email/webhook).
Impact: Medium. Effort: Medium.
Deliverable: notification hooks for drift or benchmark regression failures.

## Completed (Priority 0)
1. Define API contract for `POST /v1/ask` and run trace endpoints.
2. Define core schemas: Document, Passage, Claim, Answer, Run.
3. Scaffold repository layout for services and shared libraries.
4. Add initial run lifecycle handlers for `/v1/ask`, `/v1/runs/{id}`, and `/v1/runs/{id}/trace` (stubbed).
5. Implement retrieval adapters for arXiv, Crossref, and Semantic Scholar.
6. Connect retrieval execution into orchestrator and gateway run lifecycle flow.
7. Replace in-memory run store with Postgres-backed persistence.
8. Add Alembic migrations for run schema and wire migration commands into workflow.
9. Add provider client layer with env-based key loading for OpenAI/Anthropic/Kimi/Gemini.
10. Add retrieval integration tests with fixture data and schema validation.
11. Tighten run cancel semantics (prevent cancelling terminal runs).
12. Add CI workflow for migrations, compile checks, and tests.
13. Build ingestion pipeline (HTML/PDF fetch + parse service).
14. Add evidence store tables and persistence for documents/passages/claims/links.
15. Implement orchestrator multi-round investigation loop (broad/gap-fill/adversarial queries).
16. Add claim-to-passage citation linking and unsupported-claim checks.
17. Add verifier stage with contradiction flags and confidence penalties.
18. Add calibration pipeline and persistence (`run_calibration_features`).
19. Add routed/ensemble execution planning and provider usage tracing.
20. Implement real provider generation adapters for OpenAI/Anthropic/Kimi/Gemini with fallback synthesis.
21. Build benchmark/evaluation harness with rubric scoring and calibration metrics (ECE/Brier).
22. Add contradiction retrieval depth controls and verifier reranking heuristics.
23. Add observability metric capture and persistence (`run_observability_metrics`).
24. Add budget-aware cost/latency policy guards with early-stop behavior.
25. Introduce prompt/config versioning and reproducibility metadata per run.
26. Expand benchmark set to STEM-Exam-200 and add baseline capture workflow.
27. Improve answer synthesis quality with claim clustering and section-aware summarization.
28. Add drift checker + thresholds config + scheduled CI sample drift gate.
