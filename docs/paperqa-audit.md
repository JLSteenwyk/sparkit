# PaperQA Audit Notes

Date: 2026-02-26

## What we audited
- Repository: https://github.com/Future-House/paper-qa
- Project docs: https://futurehouse.gitbook.io/futurehouse-cookbook/paperqa

## Patterns worth borrowing
- Query-focused evidence gathering over many candidate papers (not single-pass answering).
- Question-conditioned contextual scoring/ranking of candidate papers before final synthesis.
- Iterative retrieval with follow-up searches from high-signal papers (citation-style traversal).
- Evidence-first finalization over direct prior-only completion.

## What was incorporated into SPARKIT now
- PaperQA-inspired RCS rerank pass in retrieval rounds:
  - `SPARKIT_ENABLE_RCS_RERANK=1` (default).
  - Implemented in `services/orchestrator/app/engine.py` via `_rcs_rerank_records(...)`.
- Citation-traversal query injection:
  - `SPARKIT_ENABLE_CITATION_TRAVERSAL_QUERIES=1` (default).
  - Injects DOI/title follow-up queries into the next retrieval stage.
- MCQ anti-bias prompt improvements:
  - Deterministic label permutation and explicit anti-letter-prior guidance in scorer/judge/eliminator/arbitrator prompts.

## Remaining ideas (not yet implemented)
- True bibliographic citation graph traversal (references/cited-by via dedicated scholarly graph endpoints).
- Multi-hop evidence chain building and explicit claim-to-source support graphs at answer time.
- Structured per-paper evidence summaries persisted as first-class objects for later synthesis and audit.
