# HLE Gold Bio/Chem Subset

This folder stores benchmark subsets sourced from `futurehouse/hle-gold-bio-chem`.

## Default subset
- `questions_bio10_chem10.json`: 20 questions total (10 biology/medicine + 10 chemistry), sampled deterministically.

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

## Run direct single-call baselines (no SPARKIT orchestration)
```bash
make baseline-capture-direct-calls-hle20
```
