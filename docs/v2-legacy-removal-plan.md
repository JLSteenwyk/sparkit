# Legacy Removal Plan (Post-V2 Acceptance)

## Scope
Remove legacy MCQ decision paths and stale experiment surfaces after `option_graph_v2` meets acceptance gates.

## Keep Until Cutover
- Current `single`, `routed`, `ensemble` modes.
- Existing benchmark/eval format files used by historical runs.
- Existing provider integrations and cost accounting paths.

## Targeted Removal/Deprecation Areas

### 1) Legacy MCQ Selection/Fallback Logic (`services/orchestrator/app/engine.py`)
- Candidates for removal after V2 default:
  - legacy scorer/judge-only pathways not backed by structured option matrix
  - arbitration-only fallback pathways
  - deterministic tie-break fallback for unresolved MCQ letters
- Replace with:
  - deterministic matrix decision from structured evidence graph.

### 2) Legacy Feature Flags
- Audit and remove flags only used by deprecated MCQ path.
- Consolidate into smaller V2 flag set:
  - retrieval coverage
  - extraction depth
  - scoring thresholds
  - contradiction weighting

### 3) Stale Benchmark Artifacts and Scripts
- Archive outdated experiment runs and ad-hoc one-off run folders.
- Keep canonical run sets:
  - frozen baseline
  - latest accepted V2 runs
- Remove obsolete helper scripts no longer used by runbook.

### 4) Docs Cleanup
- Replace V1 architecture sections with V2 default flow.
- Move legacy architecture notes to archive section.
- Keep one migration mapping from old stage names to V2 stage names.

## Removal Checklist (Do Not Start Until V2 Gate Passes)
- [ ] V2 accuracy/cost/latency/calibration gates passed on barometer-10 + HLE-25.
- [ ] V2 diagnostics stable across 3 repeated runs.
- [ ] Legacy MCQ code paths unreachable from default execution.
- [ ] Deprecated flags removed from config docs.
- [ ] Results directory pruned and archived.
- [ ] `git grep` confirms no remaining references to removed paths.

