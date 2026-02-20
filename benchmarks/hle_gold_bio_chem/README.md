# HLE Gold Bio/Chem Subset

This folder stores benchmark subsets sourced from `futurehouse/hle-gold-bio-chem`.

## Default subset
- `questions_bio10_chem10.json`: 20 questions total (10 biology/medicine + 10 chemistry), sampled deterministically.
- `questions_full.json`: full HLE bio/chem set (149 questions: 107 biology/medicine + 42 chemistry).

## Generate the default subset
```bash
make benchmark-generate-hle-biochem-20
```

Equivalent command:
```bash
./venv/bin/python scripts_generate_hle_biochem_subset.py \
  --output benchmarks/hle_gold_bio_chem/questions_bio10_chem10.json \
  --bio-count 10 \
  --chem-count 10 \
  --seed 7
```

## Run baseline capture on the subset
```bash
make baseline-capture-hle-biochem-20
```

Faster parallel run options:
```bash
./venv/bin/python scripts_capture_baselines.py \
  --questions benchmarks/hle_gold_bio_chem/questions_bio10_chem10.json \
  --configs single_openai,single_anthropic,routed_frontier \
  --max-questions 10 \
  --parallel-workers 2 \
  --parallel-configs 2 \
  --skip-missing-keys
```

## Run direct single-call baselines (no SPARKIT orchestration)
```bash
make baseline-capture-direct-calls-hle20
```

## Run full HLE149 benchmark
SPARKIT configs:
```bash
./venv/bin/python scripts_capture_baselines.py \
  --questions benchmarks/hle_gold_bio_chem/questions_full.json \
  --label hle_biochem_full_149 \
  --configs single_openai,single_anthropic,routed_frontier \
  --parallel-workers 2 \
  --parallel-configs 2 \
  --min-sources 1 \
  --max-latency-s 120 \
  --max-cost-usd 3.0 \
  --skip-missing-keys
```

Direct single-call baseline:
```bash
./venv/bin/python scripts_capture_direct_baselines.py \
  --questions benchmarks/hle_gold_bio_chem/questions_full.json \
  --label hle_biochem_full_149_direct \
  --providers openai,anthropic,gemini,kimi \
  --skip-missing-keys
```

Routed-only long run in `tmux` (recommended):
```bash
tmux new-session -d -s routed_frontier_hle149 \
  "cd /mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SPARKIT && \
  ./venv/bin/python scripts_capture_baselines.py \
    --questions benchmarks/hle_gold_bio_chem/questions_full.json \
    --label hle_biochem_full_149_routed_tmux \
    --configs routed_frontier \
    --parallel-workers 2 \
    --parallel-configs 1 \
    --min-sources 1 \
    --max-latency-s 120 \
    --max-cost-usd 3.0 \
    --skip-missing-keys"
```

## Repeated slices with confidence intervals
```bash
make benchmark-repeated-slices
```
