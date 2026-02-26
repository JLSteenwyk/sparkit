# V2 Freeze Baseline Snapshot

Date: 2026-02-26

## Baseline Benchmark References

### Direct baseline (barometer-10)
- Run: `benchmarks/results/barometer10_p1_direct_openai_20260225T195229Z`
- Report: `benchmarks/results/barometer10_p1_direct_openai_20260225T195229Z/report_direct_openai.json`
- Metrics:
  - `average_rubric_score`: `0.3`
  - `brier_score`: `0.25`
  - `ece`: `0.2`
  - `avg_latency_s`: `1.1087`
  - `avg_cost_usd`: `0.0004389`

### SPARKIT pre-V2 reference (metadata/no-collapse)
- Run: `benchmarks/results/barometer10_metadata_fulltext_no_collapse_20260226T064745Z`
- Report: `benchmarks/results/barometer10_metadata_fulltext_no_collapse_20260226T064745Z/report_single_openai.json`
- Metrics:
  - `average_rubric_score`: `0.2`
  - `brier_score`: `0.1664`
  - `ece`: `0.08`
  - `avg_latency_s`: `107.8446`
  - `avg_cost_usd`: `0.0456169`

### SPARKIT hard-gated MCQ pre-synthesis coverage run
- Run: `benchmarks/results/barometer10_mcq_gate_v2_timeout25_20260226T155357Z`
- Report: `benchmarks/results/barometer10_mcq_gate_v2_timeout25_20260226T155357Z/report_single_openai.json`
- Metrics:
  - `average_rubric_score`: `0.0`
  - `mcq_unknown_answers`: `10/10`
  - `pre_synthesis_coverage_pass_rate`: `0.0`
  - `avg_pre_synthesis_missing_labels_initial`: `5.8`
  - `avg_pre_synthesis_missing_labels_final`: `5.8`
  - `avg_coverage_expansion_rounds`: `2.0`
  - `avg_coverage_expansion_queries`: `8.4`
  - `avg_parse_failures_per_mcq`: `1.2`

## Interpretation
- Direct calls currently outperform SPARKIT on this barometer slice.
- New hard gating prevents unsafe guessing but exposes retrieval weakness:
  - option-discriminative evidence is not being retrieved at required coverage.

## Freeze Use
- Use this snapshot as the immutable baseline for V2 A/B decisions.
- Do not compare V2 against ad-hoc older experiments outside these pinned runs.

