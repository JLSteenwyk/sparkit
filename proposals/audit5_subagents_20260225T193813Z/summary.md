# SPARKIT Subagent Audit (5)

Generated from 5 parallel Claude subagent audits.

## 1. Calibration-Gated Option Rescue: Active Evidence-Confusion Detection and Per-Option Evidence Re-Ranking

- Proposal ID: `audit5-calibration-option-rescue`
- Est. Accuracy Delta: `15 pp`
- Cost Impact: `decrease`
- Latency Impact: `decrease`
- Confidence: `medium`
- Distinctive Angle: Use the calibration signal (Brier/ECE + per-option evidence balance) as an active answer-selection override — not just a reporting metric — by detecting evidence-confusion regimes and triggering a structured MCQ elimination pass before committing to a final answer.
- Top Retrieval Upgrade: No additional retrieval rounds needed — option rescue reuses already-persisted mcq_option_scorer evidence from existing retrieval runs, making this zero-retrieval-cost at rescue time.

## 2. Staged Adversarial Verification Loop (SAVL): Falsification-First Critic Orchestration

- Proposal ID: `audit5-orch-savl-001`
- Est. Accuracy Delta: `8 pp`
- Cost Impact: `increase`
- Latency Impact: `increase`
- Confidence: `medium`
- Distinctive Angle: Treat falsification as a first-class orchestration primitive rather than a post-hoc review step. A dedicated Critic agent is tasked with disproving each intermediate claim before the Orchestrator accepts it, forcing targeted re-retrieval and re-planning on any disputed node rather than only at the final answer boundary.
- Top Retrieval Upgrade: Adversarial retrieval per node: query = negation of claim + domain context, forcing the retrieval system to surface contradicting evidence rather than confirming evidence — this catches hallucinated facts that confirmatory retrieval misses

## 3. Stepwise Epistemic Decomposition with Marginal-Coverage Reranking

- Proposal ID: `audit5-retrieval-sed-001`
- Est. Accuracy Delta: `7 pp`
- Cost Impact: `increase`
- Latency Impact: `increase`
- Confidence: `medium`
- Distinctive Angle: Sequential, context-conditioned query decomposition where each sub-query is informed by what prior retrieval steps already established — eliminating redundant retrieval and targeting only the residual knowledge gap at each stage, with reranking scored on marginal information gain rather than relevance-to-query.
- Top Retrieval Upgrade: Sequential context-conditioned sub-queries improve recall of late-chain terminology-specific facts that parallel decomposition systematically misses

## 4. Parallel Speculative Sub-Problem Decomposition with Semantic Intermediate Cache

- Proposal ID: `sparkit-audit5-throughput-001`
- Est. Accuracy Delta: `7 pp`
- Cost Impact: `mixed`
- Latency Impact: `decrease`
- Confidence: `medium`
- Distinctive Angle: Hard STEM questions share latent sub-problems (e.g., unit conversions, integral forms, reaction mechanisms) across queries. Parallelizing decomposition+solving while semantically caching solved sub-problems eliminates redundant LLM calls and saturates available compute, directly increasing accuracy per token budget.
- Top Retrieval Upgrade: Embed sub-problems at decomposition time and retrieve chunked domain-specific corpora (e.g., IUPAC tables, NIST thermochemical data, ArXiv abstracts) per sub-problem rather than per full question, tightening retrieval precision for hard STEM sub-queries.

## 5. Source-Authority-Weighted Claim Triangulation with Contradiction Resolution

- Proposal ID: `audit5-evidence-trust-contradiction-v1`
- Est. Accuracy Delta: `6 pp`
- Cost Impact: `increase`
- Latency Impact: `increase`
- Confidence: `medium`
- Distinctive Angle: Instead of treating all retrieved passages as equally credible, assign explicit authority tiers to sources and run atomic-claim extraction + contradiction scoring before the answer generation step. Contradictory low-authority claims are suppressed; high-authority agreement boosts claim confidence; genuine unresolved contradictions are surfaced to the LLM as explicit uncertainty signals rather than being silently concatenated into context.
- Top Retrieval Upgrade: Re-rank retrieved passages by a composite score: 0.5 × semantic similarity + 0.3 × source authority score + 0.2 × citation count proxy (when available), replacing pure embedding similarity ranking.
