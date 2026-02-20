# SPARKIT Docs Hub

This folder tracks architecture, implementation progress, and session handoff context for the literature fetching and reviewing agent.

Last updated: 2026-02-20

## How to use this folder
- Start each work session by reading `docs/context.md` and `docs/progress.md`.
- Update `docs/progress.md` after meaningful changes.
- Record final technical choices in `docs/decisions.md`.
- Keep `docs/architecture.md` aligned with the current implementation.
- Use `docs/backlog.md` to plan the next concrete tasks.

## File index
- `docs/architecture.md`: System architecture and service boundaries.
- `docs/api-contracts.md`: Gateway API contract and response shapes.
- `docs/schemas.md`: Domain schema references and persistence expectations.
- `docs/progress.md`: Current status, milestones, blockers, and recent updates.
- `docs/backlog.md`: Prioritized implementation tasks.
- `docs/decisions.md`: Decision log with rationale and consequences.
- `docs/drift-policy.md`: Drift thresholds, alerting rules, and triage workflow.
- `docs/context.md`: Short handoff snapshot for new context windows.
- `benchmarks/stem_exam_200/README.md`: Benchmark generation and baseline capture workflow.
- `benchmarks/hle_gold_bio_chem/README.md`: HLE Gold bio/chem subset generation and benchmarking workflow.

## Update cadence
- `context.md`: end of every focused work session.
- `progress.md`: whenever a task moves status.
- `decisions.md`: whenever an architectural or policy decision is finalized.

## Current emphasis
- HLE149 benchmark tracking (SPARKIT vs direct single-call).
- Cost-accounting accuracy and token-usage transparency.
- Provider-specific response handling robustness (especially Kimi).
