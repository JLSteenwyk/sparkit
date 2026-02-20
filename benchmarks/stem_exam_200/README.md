# STEM-Exam-200 Benchmark

This folder contains the expanded 200-question benchmark used for baseline capture and calibration tracking.

## Files
- `questions.json`: 200 benchmark questions across 10 STEM domains.

## Re-generate questions
```bash
./venv/bin/python scripts_generate_stem_exam_200.py
```

## Run evaluation on full set
```bash
make eval-benchmark-full
```

## Capture baseline runs
```bash
make baseline-capture
```

Official full capture (uses loaded provider keys):
```bash
make baseline-capture-official
```

Outputs are written to `benchmarks/results/<label>_<timestamp>/` and include:
- `manifest.json`
- `predictions_<config>.json`
- `report_<config>.json`

## Default baseline configs
- `single_openai`
- `single_anthropic`
- `single_gemini`
- `single_kimi`
- `routed_frontier` (`openai, anthropic, gemini`)

Use environment keys:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `KIMI_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## Drift checks
```bash
make drift-check-sample
./venv/bin/python scripts_drift_check.py --candidate-manifest benchmarks/results/<candidate>/manifest.json --baseline-manifest benchmarks/results/<baseline>/manifest.json
```
