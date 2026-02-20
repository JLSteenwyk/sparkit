# Progress Tracker

Last updated: 2026-02-19

## Project status
- Overall phase: Advanced pipeline implementation
- Current milestone: M5 in progress (benchmarking, drift monitoring, and quality tuning)

## Milestones
1. M1: Retrieval + ingestion + evidence storage baseline
- Status: Completed

2. M2: Single-model end-to-end answer pipeline
- Status: Completed

3. M3: Verification + confidence calibration
- Status: Completed

4. M4: Routed + ensemble model execution
- Status: Completed

5. M5: Benchmark and tuning
- Status: In progress

## Current sprint focus
- Calibration and contradiction drift alerting.
- Full official baseline capture execution on frontier providers.

## Blockers
- None.

## Recent updates
- 2026-02-20: Removed `ensemble_frontier` from benchmark presets/workflows due to poor cost-performance tradeoff.
- 2026-02-20: Added retrieval improvements in aggregator: lightweight query rewriting, relevance reranking, and source-diversity balancing.
- 2026-02-20: Added benchmark parallelism controls (`--parallel-workers`, `--parallel-configs`) for faster baseline capture.
- 2026-02-20: Added repeated-slice benchmarking with CI metrics (`scripts_benchmark_repeated_slices.py`).
- 2026-02-19: Fixed direct/provider API compatibility: OpenAI GPT-5 uses `max_completion_tokens`; Kimi default temperature set to `1.0` with env override `KIMI_TEMPERATURE`.
- 2026-02-19: Added direct single-call baseline infrastructure (`scripts_capture_direct_baselines.py`) to benchmark one raw API call/provider/question outside SPARKIT orchestration.
- 2026-02-19: Added direct-call runner and tests (`services/eval_service/app/direct_call_runner.py`).
- 2026-02-19: Added HLE Gold bio/chem subset generator (`scripts_generate_hle_biochem_subset.py`) and deterministic 10-bio/10-chem sampling module.
- 2026-02-19: Added HLE subset benchmark workflow docs and Make targets (`benchmark-generate-hle-biochem-20`, `baseline-capture-hle-biochem-20`).
- 2026-02-19: Added token-based generation cost estimation hook and wired Kimi `kimi-k2.5` pricing (cache-hit/cache-miss/output) into synthesis usage accounting.
- 2026-02-19: Added drift check module and CLI (`services/eval_service/app/drift.py`, `scripts_drift_check.py`).
- 2026-02-19: Added threshold config `benchmarks/drift_thresholds.json` with absolute + regression guards.
- 2026-02-19: Extended baseline capture with contradiction/unsupported/citation quality summary metrics.
- 2026-02-19: Added scheduled CI trigger and sample drift gate (`make drift-check-sample`).
- 2026-02-19: Added drift tests and validated suite (`31 passed, 2 skipped`).
- 2026-02-19: Wrote drift triage policy (`docs/drift-policy.md`).
- 2026-02-19: Captured official manifest structure (`benchmarks/results/official_manifest_20260219T202304Z/manifest.json`) with key-gated status.
- 2026-02-19: Upgraded synthesis with deterministic claim clustering and section-aware summaries in prompt/fallback flow.
- 2026-02-19: Added synthesis trace artifacts (`claim_clusters`, `section_summaries`) for reproducible analysis.
- 2026-02-19: Bumped prompt version to `synthesis_v1.2` for reproducibility metadata.
- 2026-02-19: Added synthesis quality tests and validated suite (`27 passed, 2 skipped`).
- 2026-02-19: Expanded benchmark set to `benchmarks/stem_exam_200/questions.json` with 200 STEM questions across 10 domains.
- 2026-02-19: Added benchmark generation tool `scripts_generate_stem_exam_200.py`.
- 2026-02-19: Added baseline capture workflow `scripts_capture_baselines.py` with single/routed/ensemble provider presets.
- 2026-02-19: Added eval runner support for `max_questions` and prediction export.
- 2026-02-19: Added benchmark utility tests and validated suite (`24 passed, 2 skipped`).
- 2026-02-19: Added prompt/config versioning and reproducibility metadata per run (fingerprinted record persisted in `runs.reproducibility_json`).
- 2026-02-19: Added Alembic migration `20260219_0005_add_run_reproducibility_fields.py` for run versioning fields.
- 2026-02-19: Added reproducibility tests and fixed deterministic fingerprinting (timestamp excluded from hash basis).
- 2026-02-19: Added verifier depth controls and contradiction reranking output.
- 2026-02-19: Added observability telemetry model + persistence (`run_observability_metrics`).
- 2026-02-19: Added budget-aware policy guardrails (cost/latency early-stop checks).
- 2026-02-19: Integrated policy/observability into orchestrator stages and trace artifacts.
- 2026-02-19: Added policy/verifier tests and validated full suite (`20 passed`).
- 2026-02-19: Verified observability metrics rows are persisted for completed runs.
