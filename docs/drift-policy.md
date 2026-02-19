# Drift Policy

Last updated: 2026-02-19

## Goal
Detect regressions in calibration and contradiction behavior early, and enforce a clear triage flow.

## Source of truth
- Threshold config: `benchmarks/drift_thresholds.json`
- Checker: `scripts_drift_check.py`
- Baseline capture: `scripts_capture_baselines.py`

## Metrics monitored
- `average_rubric_score` (higher is better)
- `brier_score` (lower is better)
- `ece` (lower is better)
- `avg_citation_coverage` (higher is better)
- `avg_unsupported_claims` (lower is better)
- `avg_contradiction_flags` (lower is better)

## Alert rules
1. Absolute threshold breach (from `absolute` thresholds) triggers alert.
2. Regression threshold breach against baseline (from `regression` thresholds) triggers alert.
3. Any breach causes `scripts_drift_check.py` to exit non-zero.

## Triage workflow
1. Confirm signal quality:
- Re-run the same benchmark command and drift check once.
- Ensure `prompt_version`/`config_version` match intended run.

2. Localize regression:
- Compare candidate vs baseline manifests by config (`single_openai`, `routed_frontier`, `ensemble_frontier`, etc.).
- Identify whether issue is calibration (`ece`, `brier`) or evidence quality (`unsupported`, `contradiction`, citation coverage).

3. Determine likely root cause:
- Retrieval/ingestion changes: citation coverage drops or contradiction spikes.
- Synthesis changes: rubric score drops with stable retrieval metrics.
- Calibration/verifier changes: ECE/Brier worsen with stable rubric.
- Provider/runtime changes: config-specific degradation only.

4. Remediation action:
- If severe (multiple metrics breached): rollback or pin prior prompt/config version.
- If isolated: ship targeted fix and re-run drift check.
- Update `docs/decisions.md` for threshold or policy changes.

## Commands
- Sample drift check (CI-safe):
```bash
make drift-check-sample
```

- Manifest drift check:
```bash
make drift-check-manifest CANDIDATE_MANIFEST=benchmarks/results/<candidate>/manifest.json BASELINE_MANIFEST=benchmarks/results/<baseline>/manifest.json
```

- Official baseline capture (uses configured provider keys):
```bash
make baseline-capture-official
```

## Notes
- Full official baseline capture can be expensive/slow on 200 questions.
- For quick preflight, use `scripts_capture_baselines.py --max-questions N` and run full capture afterward.
