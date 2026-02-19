# Decision Log

Last updated: 2026-02-19

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
