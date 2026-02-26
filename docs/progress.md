# Progress Tracker

Last updated: 2026-02-26

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
- Align benchmark scoring with strict MCQ/exactMatch behavior and debug grading path failures.
- Establish a stable 10-question barometer subset for fast SPARKIT-vs-direct comparisons.
- Keep results directory clean and focused on current benchmark artifacts.

## Blockers
- None.

## Recent updates
- 2026-02-26: Implemented anti-collapse MCQ policy + observability:
  - Added strict MCQ option eligibility gate before scorer/judge/arbitration selection (`SPARKIT_ENABLE_MCQ_STRICT_ELIGIBILITY` + threshold knobs).
  - Removed forced deterministic letter fallback when MCQ output is unparseable; format fallback now emits `<answer>UNKNOWN</answer>` when no eligible option exists.
  - Added finalization hard block (`SPARKIT_MCQ_HARD_BLOCK_ON_WEAK_EVIDENCE`) to replace weak-evidence MCQ answers with `<answer>UNKNOWN</answer>`.
  - Added per-question `mcq_decision` provenance to benchmark predictions for scorer/judge/fallback/rescue/hard-block diagnostics.
- 2026-02-26: Incorporated PaperQA-inspired retrieval patterns:
  - Added question-conditioned candidate scoring rerank pass (`SPARKIT_ENABLE_RCS_RERANK`).
  - Added citation-traversal query injection loop (`SPARKIT_ENABLE_CITATION_TRAVERSAL_QUERIES`) to pull follow-up/replication evidence from retrieved paper titles/DOIs.
  - Added deterministic MCQ label permutation + anti-letter-prior instructions in scorer/judge/elimination/arbitration prompts.
- 2026-02-26: Hardened MCQ fallback/selection robustness:
  - Replaced heuristic parse fallback with strict arbitration fallback (`SPARKIT_ENABLE_MCQ_ARBITRATION_FALLBACK`) and non-alphabetical deterministic tie-break.
  - Added `mcq_parse_failure` trace stage for scorer/judge/arbitration parse diagnostics.
  - Added benchmark-level MCQ collapse detector in manifests (`mcq_collapse_warning` + letter distribution fields).
- 2026-02-26: Set up scholarly metadata endpoints to reduce publisher-page scrape failures:
  - Added PubMed E-utilities adapter `search_pubmed_metadata` in `services/retrieval_service/app/adapters.py`.
  - PubMed adapter now enriches metadata hits with abstracts via EFetch XML.
  - Added retrieval toggle `SPARKIT_ENABLE_PUBMED_METADATA` in aggregator.
  - Added metadata-first corpus hydrator `scripts_ingest_scholarly_metadata.py` with targeted fulltext parsing for top-k candidates per question, then metadata fallback.
- 2026-02-26: Upgraded MCQ option selection:
  - Added domain-aware MCQ prompting (`chemistry` vs `biology_medicine` guidance) across synthesis/judge/scoring prompts.
  - Added dual-scorer MCQ blending (`SPARKIT_ENABLE_MCQ_DUAL_SCORER`) with primary+adversarial score fusion and dossier-aware weighting.
  - Barometer run `barometer10_science_enhanced_mcq_dual_selector_20260226T023728Z`: score stayed `0.30`, cost/latency remained stable vs science-enhanced baseline.
- 2026-02-26: Implemented MCQ evidence-to-answer hard constraint:
  - Added `mcq_evidence_gate` policy that validates selected option support from option dossiers/scores.
  - Gate is enforced in confidence-retry phase (can trigger one retrieval retry) and finalization (confidence penalty + uncertainty reason on failure).
  - Added tests for pass/fail behavior in `services/orchestrator/tests/test_synthesis_quality.py`.
- 2026-02-26: Enabled science-enhanced retrieval policy by default:
  - Added `SPARKIT_SCIENCE_ENHANCED_MODE` (default `1`) in retrieval quality filtering.
  - Web-style sources (Brave/Exa) are now filtered to academic domains unless DOI-backed.
  - Added retrieval test coverage to verify non-academic web results are dropped in science-enhanced mode.
- 2026-02-25: Implemented source-quality + evidence-consensus gating:
  - Added source quality scoring (`_record_source_quality_score`) and priority ranking for ingestion selection.
  - Added evidence consensus profiling (`_evidence_consensus_profile`) with independent high-quality source checks.
  - Final confidence now receives explicit penalties when evidence lacks multi-source high-quality support or consensus.
  - Added `evidence_consensus_gate` trace stage and updated synthesis-quality tests.
- 2026-02-25: Expanded Exa retrieval integration:
  - Added adapters for Exa `answer`, `content`, and `research` APIs in `services/retrieval_service/app/adapters.py`.
  - Wired aggregator flags `SPARKIT_ENABLE_EXA_ANSWER`, `SPARKIT_ENABLE_EXA_CONTENT`, `SPARKIT_ENABLE_EXA_RESEARCH`.
  - Added Exa content hydration step over top retrieved URLs and request telemetry entries (`exa_answer`, `exa_research`, `exa_content`).
  - Added test coverage in `services/retrieval_service/tests/test_aggregator.py`.
- 2026-02-25: Added Exa endpoint-level cost accounting in orchestrator:
  - Retrieval now computes Exa costs from telemetry (`exa_web`, `exa_answer`, `exa_research`, `exa_content_pieces`).
  - Final `provider_usage` includes Exa rows (`search-api`, `answer-api`, `research-api`, `contents-api`) when used.
- 2026-02-25: Verified strict evaluator path is active (`multipleChoice` letter match + `exactMatch` dual LLM graders).
- 2026-02-25: Fixed OpenAI exact-match grader fallback for GPT-5-family 400/404 responses in `services/orchestrator/app/providers/clients.py`.
- 2026-02-25: Ran rephrase-then-answer A/B:
  - HLE-10: no gain vs direct; higher cost/latency.
  - HLE-20: no gain vs direct; higher cost/latency.
- 2026-02-25: Added locked barometer subset:
  - `benchmarks/hle_gold_bio_chem/questions_barometer10_direct30.json`
  - Validated direct baseline at `0.3`:
    - `benchmarks/results/barometer10_direct_openai_v3_20260225T183524Z`
- 2026-02-25: Cleaned result tracking surface:
  - Kept only active runs in `benchmarks/results/`.
  - Moved older runs to `benchmarks/results/archive_outdated_20260225/`.
- 2026-02-22: Implemented top-6 retrieval upgrade v1 in orchestrator:
  - Added explicit falsification retrieval round (`retrieval_round_4_falsification`) with option-focused query generation.
  - Added optional semantic reranking hook for retrieval and final ingestion selection.
  - Upgraded ingestion selection to relevance+novelty (MMR-style) balancing for better source diversity/dedup.
  - Verifier now uses adversarial + falsification evidence pools.
  - Added retrieval-upgrade tests: `services/orchestrator/tests/test_engine_retrieval_upgrades.py`.
- 2026-02-22: Ran HLE-5 validation for top-6 v1:
  - `benchmarks/results/hle5_single_openai_top6_v1_20260222T182915Z/`
  - Avg rubric `0.2` (unchanged vs xhigh baseline), total cost `$0.0598115`, total latency `407.081s`.
- 2026-02-22: Implemented top-6 tuning v2:
  - Falsification restricted to top candidate options (`SPARKIT_FALSIFICATION_MAX_OPTIONS=2`).
  - Stage-targeted semantic reranking defaults to gap-fill/adversarial/falsification rounds.
  - Added chemistry-entity-aware query tightening in heuristic planner.
- 2026-02-22: Ran HLE-5 validation for top-6 v2:
  - `benchmarks/results/hle5_single_openai_top6_v2_20260222T184907Z/`
  - Avg rubric `0.2` (unchanged), total cost `$0.05514425` (slightly lower), total latency `417.681s`.
- 2026-02-22: Implemented closed-loop claim-gap injection (top-6 missing piece):
  - Added `SPARKIT_ENABLE_CLAIM_GAP_LOOP` with planning-provider query generation after each retrieval stage.
  - Gap queries are merged into the next retrieval stage (`SPARKIT_CLAIM_GAP_MAX_QUERIES`, `SPARKIT_CLAIM_GAP_MAX_NEXT_QUERIES`).
  - Added trace stage `retrieval_claim_gap_loop` for auditability.
  - Added tests for claim-gap query parsing/fallback (`services/orchestrator/tests/test_engine_retrieval_upgrades.py`).
- 2026-02-22: Added claim-gap budget/evidence guardrail to avoid runaway latency/cost:
  - New gate `retrieval_claim_gap_gate` traces injection decision + reason per stage.
  - Claim-gap injection now requires weak evidence by default and sufficient cost/latency headroom.
  - Added knobs: `SPARKIT_CLAIM_GAP_REQUIRE_LOW_EVIDENCE`, `SPARKIT_CLAIM_GAP_MIN_NEW_DOCS_TRIGGER`, `SPARKIT_CLAIM_GAP_MIN_RELEVANCE_TRIGGER`, `SPARKIT_CLAIM_GAP_MAX_COST_RATIO`, `SPARKIT_CLAIM_GAP_MAX_LATENCY_RATIO`, `SPARKIT_CLAIM_GAP_FORCE`.
- 2026-02-22: Ran HLE-5 validation for top-6 v4 + claim-gap gate:
  - `benchmarks/results/hle5_single_openai_top6_v4_claimgap_gate_20260222T192351Z/`
  - Avg rubric `0.2` (unchanged), calibration (`Brier 0.618225`, `ECE 0.691`) aligned with top-6 v1.
  - Runtime/cost improved substantially vs un-gated claim-gap v3 (`457.694s` and `$0.059507` vs `923.892s` and `$0.066386`), while keeping the same score.
- 2026-02-22: Ran HLE-5 validation for top-6 v3 + claim-gap:
  - `benchmarks/results/hle5_single_openai_top6_v3_claimgap_20260222T190203Z/`
  - Avg rubric `0.2` (unchanged), calibration improved (Brier `0.54837`, ECE `0.638`), but cost/latency rose (`$0.06638625`, `923.892s`).
- 2026-02-22: Reviewed `scripts_run_claudep_proposals_runner.sh` outputs and selected the top 6 recurring improvement methods as the active retrieval upgrade pack:
  1) falsification rounds, 2) option-specific MCQ retrieval, 3) closed-loop gap-fill retrieval, 4) diversity-aware selection/dedup, 5) query decomposition planner, 6) semantic reranking.
- 2026-02-22: Added high-effort reasoning/thinking controls in SPARKIT provider clients:
  - OpenAI: `OPENAI_REASONING_EFFORT` (`low|medium|high|xhigh`) for GPT-5 models via Responses API.
  - Anthropic: `ANTHROPIC_THINKING_ENABLED`, `ANTHROPIC_THINKING_BUDGET_TOKENS` (with compatibility handling for low-token calls and thinking-mode request shape).
  - Gemini: `GEMINI_THINKING_BUDGET_TOKENS` wired to `generationConfig.thinkingConfig.thinkingBudget`.
- 2026-02-22: Ran reasoning/thinking smoke validation and documented payload compatibility:
  - `benchmarks/results/reasoning_smoke_20260222T172431Z.json`
  - `benchmarks/results/gemini_high_effort_smoke_20260222T1728Z.json`
  - `docs/reasoning-params-smoke.md`
- 2026-02-22: Ran HLE-5 SPARKIT `single_openai` with `gpt-5.2` + `OPENAI_REASONING_EFFORT=xhigh`:
  - `benchmarks/results/hle5_single_openai_gpt52_xhigh_reasoning_20260222T173533Z/`
  - Avg rubric `0.2` (1/5), total cost `$0.05798625`, total latency `390.916s`, failures `0`.
- 2026-02-22: Ran HLE-5 SPARKIT high-thinking comparison for Anthropic + Gemini:
  - `benchmarks/results/hle5_single_anthropic_gemini_high_thinking_20260222T181121Z/`
  - `single_anthropic` avg rubric `0.0`, total cost `$0.097675`, total latency `423.568s`, failures `0`.
  - `single_gemini` avg rubric `0.0`, total cost `$0.026192`, total latency `351.049s`, failures `0`.
- 2026-02-21: Implemented adaptive retrieval continuation gate in orchestrator (`SPARKIT_ADAPTIVE_*` knobs) that stops retrieval rounds when novelty/relevance gains are low; emits `retrieval_adaptive_gate` trace stage for auditability.
- 2026-02-21: Validated adaptive retrieval on HLE-5 live-tuned slice (`single_openai`, `gpt-5.2`):
  - Non-adaptive live tuned: avg score `0.2`, total cost `$0.7040`, total latency `~799.6s`.
  - Adaptive live tuned: avg score `0.2`, total cost `$0.2972`, total latency `~420.9s`.
  - Net: held quality on this slice while reducing cost by `~57.8%` and latency by `~47.4%`.
- 2026-02-21: Ran focused HLE-5 mixed slice (`q001,q002,q005,q010,q011`) to tune SPARKIT retrieval behavior. Results:
  - `direct_openai` (`gpt-5.2`): avg score `0.0`, total latency `~19.0s`, total cost `$0.0116`.
  - `single_openai` local-corpus-only (`SPARKIT_ENABLE_LIVE_RETRIEVAL=0`): avg score `0.0` (both baseline and deeper-retrieval variants), latency `~128-151s`.
  - `single_openai` live retrieval tuned (`min_sources=12`, higher retrieval floor): avg score improved to `0.2` (1/5 correct), but with large cost/latency increase (total latency `~799.6s`, total cost `$0.7040`).
  - Takeaway: real live retrieval adapters improved correctness on this slice, while local-only corpus retrieval underperformed; next tuning should prioritize retrieval precision and query/round budgeting to preserve gains without severe cost/latency blow-up.
- 2026-02-21: Added retrieval runtime switch `SPARKIT_ENABLE_LIVE_RETRIEVAL` (`1` by default) to support local-corpus-only benchmark/debug loops when needed.
- 2026-02-21: Added Brave Search retrieval cost accounting at exact per-request rate (`$5/1000` => `$0.005` each), wired from retrieval request telemetry into orchestrator run cost/provider usage.
- 2026-02-21: Added local corpus infrastructure (`corpus_documents` + `corpus_chunks`) with local-first retrieval path and broad-science corpus builder script (`scripts_build_science_corpus.py`).
- 2026-02-21: Implemented MCQ option-aware retrieval and option scoring/judging in orchestrator (`answer_choices` parsing, `options` retrieval intent, `mcq_option_scorer` trace stage with per-choice support/contradiction scores).
- 2026-02-21: Expanded retrieval federation with OpenAlex + Europe PMC and added hard HLE-domain blocking in retrieval/ingestion (`huggingface.co`, `futurehouse.org`) to avoid benchmark data leakage in downloaded evidence.
- 2026-02-21: Fixed SPARKIT run failure caused by NUL bytes in persisted text (`PostgreSQL text fields cannot contain NUL (0x00) bytes`) by adding sanitization in ingestion/evidence persistence paths.
- 2026-02-21: Improved arXiv ingestion quality by extracting abstract blocks from `arxiv.org/abs/*` pages (instead of noisy page boilerplate), improving retrieval-to-synthesis grounding.
- 2026-02-21: Removed practical synthesis token caps in SPARKIT by default (uncapped generation unless `constraints.synthesis_max_tokens` is explicitly set), and raised default retrieval/ingestion depth for stronger evidence coverage.
- 2026-02-21: Implemented retrieval quality v1 in orchestrator: LLM-guided retrieval planner (segment decomposition + intent queries), mode-aware planned retrieval rounds, and relevance-based section chunk selection during ingestion to improve grounding quality.
- 2026-02-21: Added retrieval planner/chunk-selection coverage tests and validated targeted suite (`18 passed` across orchestrator/retrieval tests).
- 2026-02-21: Added explicit Priority-0 follow-up to improve retrieval evidence quality (query expansion/reranking + quality gates) after observing off-topic evidence in difficult exact-match chemistry prompts.
- 2026-02-21: Replaced stalled `hle_gold_direct_batch_c_*` run with split direct runs to avoid provider blocking:
  - `hle_gold_direct_grok_split_20260221T024803Z`
  - `hle_gold_direct_mistral_split_20260221T024803Z`
- 2026-02-21: Pruned historical benchmark artifacts under `benchmarks/results/`; retained only active final HLE-gold suite directories (`hle_gold_*`).
- 2026-02-21: Launched HLE-gold benchmark wave in tmux:
  - `hle_single_core_20260221T004327Z` (`single_openai,single_anthropic,single_gemini,single_kimi,single_deepseek,single_grok,single_mistral`)
  - `hle_single_overrides_20260221T004327Z` (`single_openai_pro,single_anthropic_sonnet`)
  - `hle_direct_a_20260221T004327Z` (`openai,anthropic,gemini`)
  - `hle_direct_b_20260221T004327Z` (`kimi,deepseek`)
  - `hle_direct_c_20260221T004327Z` (`grok,mistral`) [cancelled/replaced by split runs above]
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
