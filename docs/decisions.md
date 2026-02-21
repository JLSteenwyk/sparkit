# Decision Log

Last updated: 2026-02-20

## ADR-0001: Multi-round investigation workflow
- Date: 2026-02-19
- Status: Accepted
- Decision:
  - Use a 3-round loop (broad retrieval, gap fill, adversarial verification) before answer finalization.
- Rationale:
  - Improves coverage and reduces unsupported conclusions.
- Consequences:
  - Higher latency/cost than single-pass generation.

## ADR-0002: Multi-provider model strategy
- Date: 2026-02-19
- Status: Accepted
- Decision:
  - Support single, routed, and ensemble modes across GPT/Claude/Gemini.
- Rationale:
  - Enables quality/cost tradeoffs and robustness under model-specific failure modes.
- Consequences:
  - Requires unified abstractions and model-specific prompt testing.

## ADR-0003: Citation-grounded guardrail
- Date: 2026-02-19
- Status: Accepted
- Decision:
  - Major claims must be linked to retrieved evidence passages.
- Rationale:
  - Reduces hallucination risk and improves auditability.
- Consequences:
  - Requires claim extraction and evidence-linking infrastructure.

## ADR-0004: Confidence calibration requirement
- Date: 2026-02-19
- Status: Accepted
- Decision:
  - Output answer-level and claim-level confidence with calibration.
- Rationale:
  - Raw model self-confidence is not reliable for scientific QA.
- Consequences:
  - Requires benchmark labels and calibration pipeline.

## ADR-0005: Run reproducibility metadata and prompt/config versioning
- Date: 2026-02-19
- Status: Accepted
- Decision:
  - Persist `prompt_version`, `config_version`, and stable reproducibility fingerprint per run.
- Rationale:
  - Enables replayability, auditability, and safe comparison across behavior changes.
- Consequences:
  - Requires strict version bumps when synthesis/orchestration logic changes.

## ADR-0006: STEM-Exam-200 benchmark as primary tuning set
- Date: 2026-02-19
- Status: Accepted
- Decision:
  - Use a generated 200-question cross-domain STEM benchmark as the default evaluation set.
- Rationale:
  - Improves representativeness over the 3-question sample and supports trend tracking.
- Consequences:
  - Increases evaluation runtime and baseline capture cost.

## ADR-0007: Section-aware synthesis and claim clustering
- Date: 2026-02-19
- Status: Accepted
- Decision:
  - Enrich synthesis prompts/fallback answers with deterministic claim clusters and section summaries.
- Rationale:
  - Improves answer structure and evidence-grounded synthesis quality.
- Consequences:
  - Requires prompt version bump (`synthesis_v1.2`) and new trace artifacts.

## ADR-0008: Drift gate policy for calibration and contradiction metrics
- Date: 2026-02-19
- Status: Accepted
- Decision:
  - Enforce threshold-based drift checks using absolute and baseline-regression limits.
- Rationale:
  - Prevent silent quality degradation in calibration and evidence-grounding behavior.
- Consequences:
  - Requires maintaining threshold config and canonical baseline manifests.

## ADR-0009: Deprecate `ensemble_frontier` from benchmark presets
- Date: 2026-02-20
- Status: Accepted
- Decision:
  - Remove `ensemble_frontier` from default benchmark capture presets and documented workflows.
- Rationale:
  - Observed cost-performance tradeoff was unfavorable versus routed/single configurations.
- Consequences:
  - Benchmark comparisons now center on `single_*` and `routed_frontier`.
  - Ensemble code path remains available in orchestrator internals but is no longer part of standard benchmark runs.

## ADR-0010: Treat empty direct-call outputs as failures
- Date: 2026-02-20
- Status: Accepted
- Decision:
  - In direct-call benchmarks, count empty parsed answer text as explicit failure.
  - For Kimi, treat empty `message.content` responses as generation failures (even when `reasoning_content` exists).
- Rationale:
  - Prevent misleadingly low/optimistic calibration metrics from blank answers classified as successful calls.
- Consequences:
  - Direct baseline failure counts better reflect real usability.
  - Historical direct Kimi runs with blank outputs should be treated as invalid and rerun.

## ADR-0011: Replace extended-effort with research-max mode (later deprecated)
- Date: 2026-02-20
- Status: Accepted
- Decision:
  - Remove `extended_effort` and add `research_max` mode for unconstrained deep reasoning.
  - `research_max` runs decomposition, bucketed retrieval, evidence graphing, dual-solver synthesis, debate adjudication, and task-aware finalization.
- Rationale:
  - A deeper mode should be architecturally different from baseline, not just “more rounds”.
- Consequences:
  - Higher cost/latency per question when `research_max` is used.
  - Requires separate benchmark tracking versus `single_*` and `routed`.
  - Follow-up: mode removed from active API/benchmark surfaces after early benchmarks underperformed.

## ADR-0012: Remove default latency cap
- Date: 2026-02-20
- Status: Accepted
- Decision:
  - `constraints.max_latency_s` is optional and defaults to `null` (no enforced latency cap).
- Rationale:
  - High-effort scientific runs should not be truncated by arbitrary default latency limits.
- Consequences:
  - Callers must explicitly set latency caps when strict SLOs are required.
