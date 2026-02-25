# Context Snapshot

Last updated: 2026-02-25

## What we are building
A STEM literature-fetching and reviewing agent (SPARKIT) that answers hard bio/chem questions with retrieval, synthesis, verification, and calibrated confidence.

## Current phase
Benchmark alignment and evaluation hardening.

## Current evaluation truth
- MCQ grading is strict option-letter matching.
- Exact-match grading uses dual LLM graders (Anthropic + OpenAI).
- OpenAI exact-match grader path was fixed for GPT-5-family 400/404 fallback to `/v1/responses`.

## Current benchmark status
- Unrestricted A/B (10-question mixed set) still favors direct calls over SPARKIT.
- Rephrase-then-answer A/B did not improve direct-call score on 10 or 20 question tests.
- New barometer subset locked for quick iteration:
  - `benchmarks/hle_gold_bio_chem/questions_barometer10_direct30.json`
  - Direct baseline validated at `0.3`:
    - `benchmarks/results/barometer10_direct_openai_v3_20260225T183524Z`

## Results cleanup
- Active results kept in `benchmarks/results/`:
  - `barometer10_direct_openai_v3_20260225T183524Z`
  - `hle10mix_direct_openai_unrestricted_cmp_20260225T003550Z`
  - `hle10mix_unrestricted_single_openai_20260225T003659Z`
  - `hle20_rephrase_ab_openai_20260225T032918Z`
- Older runs moved to:
  - `benchmarks/results/archive_outdated_20260225`

## Next immediate step
Run SPARKIT `single_openai` on `questions_barometer10_direct30.json` and compare against the locked direct-30 baseline.
