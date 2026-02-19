# Backlog

Last updated: 2026-02-19

## Priority 0 (next)
1. Execute full official baseline capture (all configs, full STEM-Exam-200) and store canonical baseline manifest.
2. Finalize drift thresholds after first full official baseline run.

## Priority 1
1. Add scheduled benchmark + drift checks on larger subsets in CI/nightly workflow.
2. Add per-domain drift breakdown (physics/chemistry/biology/etc.) in checker output.
3. Add alert delivery integration (Slack/email/webhook).

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
