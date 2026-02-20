# Context Snapshot

Last updated: 2026-02-20

## What we are building
A STEM literature fetching and reviewing agent that answers complex questions through multi-round investigation, citation-grounded synthesis, and confidence scoring.

## Current phase
Advanced benchmarking and quality hardening.

## Completed in this session
1. Full HLE149 benchmark dataset materialized at:
   - `benchmarks/hle_gold_bio_chem/questions_full.json`
2. Completed direct single-call HLE149 baseline run:
   - `benchmarks/results/hle_biochem_full_149_direct_escalated_20260220T021156Z/`
3. Completed SPARKIT HLE149 `single_openai` and `single_anthropic` runs:
   - `benchmarks/results/hle_biochem_full_149_escalated_20260220T021155Z/`
4. Diagnosed invalid prior runs caused by sandbox DNS/egress limitations and re-ran in network-enabled sessions.
5. Fixed Kimi direct-call anomaly:
   - `services/orchestrator/app/providers/clients.py` now treats empty `message.content` as generation failure.
   - `services/eval_service/app/direct_call_runner.py` now counts empty parsed answers as failures.
6. Added regression test:
   - `services/eval_service/tests/test_direct_call_runner.py`
7. Relaunched `routed_frontier` as long-running task in detached tmux:
   - label: `hle_biochem_full_149_routed_tmux`
   - output: `benchmarks/results/hle_biochem_full_149_routed_tmux_20260220T061844Z/`

## Immediate next actions
1. Wait for `routed_frontier` completion and collect:
   - `report_routed_frontier.json`
   - `predictions_routed_frontier.json`
2. Finish patched Kimi-only rerun and compare against prior direct Kimi artifacts.
3. Publish final SPARKIT vs direct comparison table for HLE149.
4. Decide whether to extend exact token-pricing map to OpenAI/Anthropic/Gemini for accounting-grade cost reporting.

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
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
