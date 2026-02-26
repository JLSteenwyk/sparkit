# SPARKIT V2 Roadmap (Option-Graph Pipeline)

## Goal
Replace the current MCQ workflow with an option-centric, evidence-structured pipeline that outperforms direct single-call baselines on STEM subsets while remaining auditable.

## Success Gates
- Accuracy: exceed direct OpenAI baseline on barometer-10 and HLE-25 balanced slice.
- Reliability: `mcq_unknown_answers` under agreed threshold while avoiding letter-collapse.
- Evidence quality: pre-synthesis option coverage pass rate > 0.7 on barometer-10.
- Calibration: non-regressive ECE/Brier vs current SPARKIT single_openai runs.

## Phases

### Phase 0: Freeze + Contract
- [x] Freeze baseline references and metrics (`docs/v2-freeze-baseline.md`).
- [x] Define V2 contract and acceptance criteria.
- [ ] Lock benchmark protocol for A/B comparisons (same questions, env knobs, model ids).

### Phase 1: V2 Core (Parallel Path, No Breakage)
- [x] Add `option_graph_v2` orchestration mode shell (new path, legacy untouched).
- [x] Implement initial option-hypothesis retrieval planner (support/contrast/falsification rounds) for `option_graph_v2`.
- [ ] Persist structured evidence graph objects per run (claim -> passage -> source).
- [x] Implement deterministic MCQ decision layer from option score matrix only.
  - Status: complete for `option_graph_v2` MCQ path. Legacy scorer/judge/arbitration remains only on non-V2 paths pending cutover.

### Phase 2: Retrieval Quality Engine
- [ ] Replace token-overlap structured claims with LLM structured extraction over top passages.
- [ ] Add source-independence weighting + contradiction handling in option scoring.
- [ ] Add citation-follow expansion with bounded loops and transparent diagnostics.
- [ ] Add per-option retrieval quotas and stopping rules.

### Phase 3: Evaluation + Diagnostics
- [x] Add MCQ diagnostics summary (coverage, expansion rounds, parse failures, unknown rate).
- [ ] Add per-stage quality dashboard artifact for each benchmark run.
- [ ] Add regression tests for coverage-pass, coverage-recovery, coverage-fail behaviors.
- [ ] Add deterministic replay checks for MCQ decisions.

### Phase 4: Cutover + Cleanup
- [ ] Promote V2 as default MCQ path after acceptance gates pass.
- [ ] Deprecate/remove legacy MCQ selection/fallback logic.
- [ ] Remove dead flags/helpers and obsolete scripts.
- [ ] Archive stale results and keep a single canonical benchmark table.

## Immediate Next Sprint (Execution Order)
1. Build `option_graph_v2` mode shell and deterministic decision contract.
2. Implement LLM structured-claim extractor with passage-level provenance.
3. Add regression tests for coverage gate and deterministic decision matrix.
4. Run barometer-10 A/B (`direct_openai`, `single_openai legacy`, `option_graph_v2`) and review.
