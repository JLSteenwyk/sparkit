# Context Snapshot

Last updated: 2026-02-26

## What we are building
A STEM literature-fetching and reviewing agent (SPARKIT) that answers hard bio/chem questions with retrieval, synthesis, verification, and calibrated confidence.

## Current phase
V2 pipeline planning and migration setup (option-centric evidence workflow).

## Current evaluation truth
- MCQ grading is strict option-letter matching.
- Exact-match grading uses dual LLM graders (Anthropic + OpenAI).
- OpenAI exact-match grader path was fixed for GPT-5-family 400/404 fallback to `/v1/responses`.

## Current benchmark status
- Direct baseline remains stronger than SPARKIT on barometer-10.
- MCQ hard-gating now prevents unsafe guessing but currently returns many `UNKNOWN` due to weak option coverage.
- Pinned runs for V2 comparisons are documented in:
  - `docs/v2-freeze-baseline.md`

## Planning artifacts
- V2 implementation roadmap:
  - `docs/v2-roadmap.md`
- Legacy code removal/deprecation plan:
  - `docs/v2-legacy-removal-plan.md`
- Baseline freeze snapshot:
  - `docs/v2-freeze-baseline.md`

## Next immediate step
Start V2 implementation in a parallel mode path (`option_graph_v2`) and run controlled A/B against the frozen baseline set.
