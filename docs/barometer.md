# Barometer Benchmark (Current)

Last updated: 2026-02-25

## Purpose
- Maintain a small, repeatable HLE-gold subset where direct OpenAI sits near ~30% under strict scoring.
- Use this as a quick signal to judge whether SPARKIT changes are improving or regressing.

## Scoring Mode
- `multipleChoice`: strict letter match (`<answer>X</answer>` or leading letter extraction).
- `exactMatch`: dual-grader consensus (Anthropic + OpenAI exact-match graders).
- This is the default evaluator path in `services/eval_service/app/rubric.py`.

## Locked Subset
- File: `benchmarks/hle_gold_bio_chem/questions_barometer10_direct30.json`
- Size: `10` questions
- Composition: `5 biology`, `5 chemistry` (MCQ-only)
- IDs:
  - `q033`, `q001`, `q005`, `q009`, `q013`
  - `q010`, `q034`, `q020`, `q030`, `q032`

## Baseline Target
- Direct run: `benchmarks/results/barometer10_direct_openai_v3_20260225T183524Z`
- `direct_openai` (`gpt-5.2`) result:
  - `average_rubric_score = 0.3`
  - `num_questions = 10`
  - `failure_count = 0`

## Current Active Results (non-archived)
- `benchmarks/results/barometer10_direct_openai_v3_20260225T183524Z`
- `benchmarks/results/hle10mix_direct_openai_unrestricted_cmp_20260225T003550Z`
- `benchmarks/results/hle10mix_unrestricted_single_openai_20260225T003659Z`
- `benchmarks/results/hle20_rephrase_ab_openai_20260225T032918Z`

## Cleanup Note
- Older/experimental runs were moved to:
  - `benchmarks/results/archive_outdated_20260225`
