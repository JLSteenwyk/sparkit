# Federated Science RAG Plan

## Goal
Combine PaperQA2, Exa, Elicit, Consensus, and scite under one evidence contract to maximize scientific QA accuracy.

## Architecture
1. Adapter layer (one adapter per provider):
- `paperqa2`
- `exa`
- `elicit`
- `consensus`
- `scite`

2. Normalization layer:
- Map provider outputs into one schema (`EvidenceItem`).
- Preserve provenance metadata for auditing.

3. Federation layer:
- Deduplicate at DOI/PMID/title level.
- Rank by confidence x study quality x evidence role.
- Keep both supporting and contradicting evidence for calibration.

4. Synthesis layer (next implementation):
- Build answer only from ranked evidence pack.
- Force strict MCQ output: `<answer>X</answer>`.
- Include explicit contradiction notes when evidence conflicts.

## Normalized Evidence Schema
Required fields:
- `provider`
- `claim`
- `evidence_type` (`supporting|contradicting|neutral`)
- `study_type` (meta-analysis, RCT, etc.)
- `confidence` (0..1)

Optional but strongly preferred:
- `title`
- `doi`
- `pmid`
- `url`
- `year`
- `provenance`

## Scoring Policy (v0)
- Study-type weight:
  - meta-analysis > systematic-review > RCT > observational > preclinical > case-report > preprint
- Evidence-type weight:
  - supporting and contradicting are both high-value, neutral lower.
- Final rank score:
  - `confidence * study_weight * evidence_weight`

## Implementation Status
- Done:
  - Normalized schema and enums.
  - Provider interface.
  - Real provider integration: Exa search adapter.
  - Provider stubs for PaperQA2/Elicit/Consensus/scite.
  - Federation builder with dedupe and weighted ranking.
- Next:
  - Real API integrations for each provider.
  - Answer synthesis module from federated evidence.
  - Eval harness for per-provider ablations.
