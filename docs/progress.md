# Progress Tracker

Last updated: 2026-02-21

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
- Full HLE Gold bio/chem 149-question benchmark execution and comparison (SPARKIT vs direct single-call).
- Cost/usage accuracy hardening and provider-response robustness fixes.
- Active execution: HLE-gold `single_*` SPARKIT and direct-call baselines running in tmux with safe parallelization.

## Blockers
- None.

## Recent updates
- 2026-02-21: Launched HLE-gold benchmark wave in tmux:
  - `hle_single_core_20260221T004327Z` (`single_openai,single_anthropic,single_gemini,single_kimi,single_deepseek,single_grok,single_mistral`)
  - `hle_single_overrides_20260221T004327Z` (`single_openai_pro,single_anthropic_sonnet`)
  - `hle_direct_a_20260221T004327Z` (`openai,anthropic,gemini`)
  - `hle_direct_b_20260221T004327Z` (`kimi,deepseek`)
  - `hle_direct_c_20260221T004327Z` (`grok,mistral`)
  - Outputs writing to `benchmarks/results/hle_gold_single_*` and `benchmarks/results/hle_gold_direct_batch_*`.
- 2026-02-21: Added per-question failure indexing for SPARKIT benchmark runs (`failures_<config>.json`, `failure_count`, `failed_question_ids`) and direct baseline manifests (`failed_question_ids`) to support fair targeted reruns.
- 2026-02-21: Hardened direct-call reliability for DeepSeek/Grok: added direct retry/backoff (`DIRECT_CALL_MAX_ATTEMPTS`, `DIRECT_CALL_RETRY_BACKOFF_S`), provider timeout controls (`SPARKIT_PROVIDER_TIMEOUT_S`, `<PROVIDER>_TIMEOUT_S`), and DeepSeek reasoning-content fallback when `message.content` is empty.
- 2026-02-21: Added regression tests for direct-call retry behavior (timeout retry + DeepSeek reasoning-content retry) and validated suite (`14 passed`).
- 2026-02-20: Updated exact pricing coverage and model defaults: added `gemini-3.1-pro-preview`, replaced Grok pricing/defaults with `grok-4-0709`, `grok-4-fast-reasoning`, `grok-4-fast-non-reasoning`; verified exact-pricing checks in policy tests.
- 2026-02-20: Deprecated `research_max` mode from active API/benchmark paths after initial benchmark underperformance.
- 2026-02-20: Tuned `extended_effort` profile to reduce ingestion breadth (`ingestion_target_docs` from `max(min_sources+3, 8)` to `max(min_sources+2, 6)`).
- 2026-02-20: Hardened verifier contradiction logic to ignore non-marker records and scale penalties by contradiction strength.
- 2026-02-20: Updated revision pass policy: skip revision for answer-choice questions and preserve lexical anchors to avoid benchmark keyword regressions.
- 2026-02-20: Implemented evidence reranking + source-diverse record selection before ingestion/synthesis.
- 2026-02-20: Implemented answerability gate with explicit abstain behavior when evidence quality is below threshold.
- 2026-02-20: Added single-model benchmark variants `single_openai_pro` and `single_anthropic_sonnet` with per-config model overrides.
- 2026-02-20: Removed support for `kimi-k2.5`; standardized on `kimi-k2-turbo-preview`.
- 2026-02-20: Added new `extended_effort` orchestration mode (deeper retrieval profile, larger synthesis budget, optional revision pass).
- 2026-02-20: Switched default latency behavior to uncapped (`max_latency_s = null`) unless explicitly provided.
- 2026-02-20: Added per-call API latency tracking in provider clients and wired direct-call benchmark latency reporting.
- 2026-02-20: Added configurable exact model pricing via `SPARKIT_MODEL_PRICING_JSON` and exact-pricing detection in usage summaries.
- 2026-02-20: Diagnosed invalid benchmark runs caused by sandbox DNS/egress limits; re-ran benchmarks in network-enabled sessions.
- 2026-02-20: Confirmed full direct single-call run completion for OpenAI/Anthropic/Gemini/Kimi on HLE149.
- 2026-02-20: Confirmed SPARKIT `single_openai` and `single_anthropic` completed on HLE149; relaunched `routed_frontier` as dedicated long-running task.
- 2026-02-20: Fixed Kimi direct-call handling for reasoning-only responses with empty `content`.
- 2026-02-20: Updated direct-call runner to count empty parsed answers as explicit failures (`empty_answer_text`) to avoid misleading calibration metrics.
- 2026-02-20: Added regression test for empty-answer direct-call failure handling.
- 2026-02-20: Removed `ensemble_frontier` from benchmark presets/workflows due to poor cost-performance tradeoff.
- 2026-02-20: Added retrieval improvements in aggregator: lightweight query rewriting, relevance reranking, and source-diversity balancing.
- 2026-02-20: Added benchmark parallelism controls (`--parallel-workers`, `--parallel-configs`) for faster baseline capture.
- 2026-02-20: Added repeated-slice benchmarking with CI metrics (`scripts_benchmark_repeated_slices.py`).
- 2026-02-19: Fixed direct/provider API compatibility: OpenAI GPT-5 uses `max_completion_tokens`; Kimi default temperature set to `1.0` with env override `KIMI_TEMPERATURE`.
- 2026-02-19: Added direct single-call baseline infrastructure (`scripts_capture_direct_baselines.py`) to benchmark one raw API call/provider/question outside SPARKIT orchestration.
- 2026-02-19: Added direct-call runner and tests (`services/eval_service/app/direct_call_runner.py`).
- 2026-02-19: Added HLE Gold bio/chem subset generator (`scripts_generate_hle_biochem_subset.py`) and deterministic 10-bio/10-chem sampling module.
- 2026-02-19: Added HLE subset benchmark workflow docs and Make targets (`benchmark-generate-hle-biochem-20`, `baseline-capture-hle-biochem-20`).
- 2026-02-19: Added token-based generation cost estimation hook and wired Kimi `kimi-k2-turbo-preview` pricing (cache-hit/cache-miss/output) into synthesis usage accounting.
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
- 2026-02-19: Added baseline capture workflow `scripts_capture_baselines.py` with single/routed provider presets (ensemble removed later).
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
