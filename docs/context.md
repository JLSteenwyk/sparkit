# Context Snapshot

Last updated: 2026-02-19

## What we are building
A STEM literature fetching and reviewing agent that answers complex questions through multi-round investigation, citation-grounded synthesis, and confidence scoring.

## Current phase
Advanced pipeline implementation.

## Completed in this session
1. Added drift checker + thresholds:
   - `services/eval_service/app/drift.py`
   - `scripts_drift_check.py`
   - `benchmarks/drift_thresholds.json`
2. Extended baseline capture output with quality summary metrics:
   - `avg_citation_coverage`
   - `avg_unsupported_claims`
   - `avg_contradiction_flags`
3. Wired CI sample drift gate and weekly schedule (`.github/workflows/ci.yml`).
4. Added drift tests and validated suite (`31 passed, 2 skipped`).
5. Added drift triage policy doc (`docs/drift-policy.md`).
6. Captured official manifest scaffold:
   - `benchmarks/results/official_manifest_20260219T202304Z/manifest.json`

## Immediate next actions
1. Run full official baseline capture with provider keys loaded:
   - `make baseline-capture-official`
2. Compare candidate runs against canonical baseline manifest:
   - `make drift-check-manifest CANDIDATE_MANIFEST=... BASELINE_MANIFEST=...`
3. Tune thresholds in `benchmarks/drift_thresholds.json` after first full official run.

## Session handoff checklist
- Activate env: `source venv/bin/activate`.
- Ensure Postgres container is running.
- Set `DATABASE_URL`.
- Run `make db-upgrade`, `make test`, `make baseline-capture-official`, `make drift-check-sample`.
- Continue from `docs/backlog.md` Priority 0.

## Provider key note
Use provider credentials from environment variables only:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `KIMI_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
