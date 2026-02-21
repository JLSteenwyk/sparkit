# Context Snapshot

Last updated: 2026-02-21

## What we are building
A STEM literature fetching and reviewing agent that answers complex questions through multi-round investigation, citation-grounded synthesis, and confidence scoring.

## Current phase
Advanced benchmarking and quality hardening.

## Completed in this session
1. Deprecated `research_max` mode and reverted active benchmarking focus to `single`, `routed`, and `ensemble`.
2. Changed default latency behavior to uncapped (`max_latency_s` optional).
3. Added model API latency tracking and direct baseline latency aggregation.
4. Added configurable exact model pricing (`SPARKIT_MODEL_PRICING_JSON`) with pricing-detection hooks.
5. Added evidence reranking + source-diverse selection before ingestion/synthesis.
6. Added answerability gate with abstain behavior for low-quality evidence profiles.
7. Validated updated suite (`48 passed, 2 skipped`).
8. Full HLE149 routed run completed:
   - `benchmarks/results/hle_biochem_full_149_routed_tmux_20260220T061844Z/`
9. Full HLE149 benchmark dataset materialized at:
   - `benchmarks/hle_gold_bio_chem/questions_full.json`
10. Completed direct single-call HLE149 baseline run:
   - `benchmarks/results/hle_biochem_full_149_direct_escalated_20260220T021156Z/`
11. Completed SPARKIT HLE149 `single_openai` and `single_anthropic` runs:
   - `benchmarks/results/hle_biochem_full_149_escalated_20260220T021155Z/`
12. Diagnosed invalid prior runs caused by sandbox DNS/egress limitations and re-ran in network-enabled sessions.
13. Fixed Kimi direct-call anomaly:
   - `services/orchestrator/app/providers/clients.py` now treats empty `message.content` as generation failure.
   - `services/eval_service/app/direct_call_runner.py` now counts empty parsed answers as failures.
14. Added regression test:
   - `services/eval_service/tests/test_direct_call_runner.py`
15. Added direct-call retry/backoff and provider timeout controls:
   - `DIRECT_CALL_MAX_ATTEMPTS`, `DIRECT_CALL_RETRY_BACKOFF_S`
   - `SPARKIT_PROVIDER_TIMEOUT_S`, `<PROVIDER>_TIMEOUT_S` (ex: `GROK_TIMEOUT_S`, `DEEPSEEK_TIMEOUT_S`)
16. Hardened DeepSeek direct handling:
   - if `message.content` is empty but `reasoning_content` exists, DeepSeek now returns reasoning text instead of hard failure.
17. Updated Grok default model to `grok-4-0709` and expanded exact pricing map (Gemini 3.1 + Grok 4 family).
18. Added per-question failure indexing for benchmark fairness:
   - SPARKIT manifests now include `failure_count` + `failed_question_ids` per config.
   - SPARKIT output now includes `failures_<config>.json` artifacts.
   - Direct-call manifests now include `failed_question_ids`.
## Immediate next actions
1. Monitor active tmux runs and capture completion manifests:
   - `hle_single_core_20260221T004327Z`
   - `hle_single_overrides_20260221T004327Z`
   - `hle_direct_a_20260221T004327Z`
   - `hle_direct_b_20260221T004327Z`
   - `hle_direct_c_20260221T004327Z`
2. Consolidate completed manifest outputs into final comparison table (quality + calibration + cost + latency).
3. Use `failed_question_ids` / `failures_<config>.json` for targeted reruns to enforce fair comparisons.

## Session handoff checklist
- Activate env: `source venv/bin/activate`.
- Ensure provider keys are exported in shell environment.
- For long-running jobs, prefer detached `tmux` sessions.
- Run `make test` after provider/eval pipeline changes.
- Continue from `docs/backlog.md` focusing on benchmark closure + cost precision.

## Provider key note
Use provider credentials from environment variables only:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `KIMI_API_KEY`
- `DEEPSEEK_API_KEY`
- `GROK_API_KEY`
- `MISTRAL_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
