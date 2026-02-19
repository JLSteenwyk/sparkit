# Schemas (V1)

Last updated: 2026-02-19

## Design goals
- Stable IDs for reproducibility and trace joins.
- Claim-centric evidence representation.
- Separation of raw ingestion artifacts from normalized answer artifacts.

## Entity: Run
```json
{
  "run_id": "run_01JT7A9J8Q6D9J4S4KB9M0Y9ZF",
  "question": "...",
  "mode": "single",
  "status": "running",
  "answer_style": "technical",
  "constraints": {
    "max_latency_s": 120,
    "max_cost_usd": 3.0,
    "min_sources": 5
  },
  "prompt_version": "synthesis_v1.2",
  "config_version": "orchestration_v1.2",
  "reproducibility": {
    "fingerprint": "sha256...",
    "question_hash": "sha256...",
    "providers": ["openai"]
  },
  "created_at": "2026-02-19T13:40:00Z",
  "updated_at": "2026-02-19T13:40:23Z"
}
```

## Entity: Document
```json
{
  "doc_id": "doc_2f3e...",
  "title": "...",
  "authors": ["A. Smith", "B. Li"],
  "year": 2024,
  "venue": "Physical Review Applied",
  "doi": "10.1103/PhysRevApplied....",
  "url": "https://...",
  "source_type": "paper",
  "peer_reviewed": true,
  "is_retracted": false,
  "quality_score": 0.87,
  "metadata": {
    "publisher": "APS",
    "open_access": true,
    "citation_count": 128
  }
}
```

## Entity: IngestionArtifact
```json
{
  "artifact_id": "art_...",
  "doc_id": "doc_2f3e...",
  "content_type": "pdf",
  "storage_uri": "s3://bucket/raw/doc_2f3e.pdf",
  "sha256": "...",
  "ingested_at": "2026-02-19T13:41:11Z",
  "parse_status": "parsed"
}
```

## Entity: Passage
```json
{
  "passage_id": "psg_...",
  "doc_id": "doc_2f3e...",
  "section": "Results",
  "text": "...",
  "offset_start": 12230,
  "offset_end": 12690,
  "embedding_model": "text-embed-x",
  "embedding_vector": [0.01, -0.02, 0.44],
  "token_count": 190
}
```

## Entity: Claim
```json
{
  "claim_id": "clm_...",
  "run_id": "run_01JT7A9J8Q6D9J4S4KB9M0Y9ZF",
  "text": "Dielectric loss in interfaces is a dominant T1 limiter in modern transmons.",
  "claim_type": "causal",
  "support_score": 0.84,
  "contradiction_score": 0.12,
  "status": "supported"
}
```

## Entity: ClaimEvidenceLink
```json
{
  "link_id": "lnk_...",
  "claim_id": "clm_...",
  "passage_id": "psg_...",
  "relation": "supports",
  "strength": 0.79,
  "rationale": "Passage reports measured T1 improvements after interface treatment."
}
```

## Entity: Answer
```json
{
  "answer_id": "ans_...",
  "run_id": "run_01JT7A9J8Q6D9J4S4KB9M0Y9ZF",
  "final_text": "...",
  "answer_confidence": 0.81,
  "uncertainty_reasons": [
    "Conflicting findings in low-frequency noise attribution"
  ],
  "insufficient_evidence": false,
  "created_at": "2026-02-19T13:41:54Z"
}
```

## Entity: ClaimConfidence
```json
{
  "claim_id": "clm_...",
  "confidence": 0.73,
  "features": {
    "support_density": 0.82,
    "source_quality": 0.77,
    "model_agreement": 0.68,
    "contradictions": 0.21,
    "coverage_gap": 0.18
  }
}
```

## Entity: ProviderUsage
```json
{
  "run_id": "run_01JT7A9J8Q6D9J4S4KB9M0Y9ZF",
  "stage": "synthesis",
  "provider": "google",
  "model": "gemini-x",
  "tokens_input": 38211,
  "tokens_output": 2301,
  "cost_usd": 0.61,
  "latency_ms": 6120
}
```

## Trace synthesis artifacts
```json
{
  "claim_clusters": [
    {
      "label": "transmon coherence",
      "count": 3,
      "sample_claims": ["..."]
    }
  ],
  "section_summaries": [
    {
      "section": "results",
      "summary": "Benchmark experiments show ..."
    }
  ]
}
```

## Enumerations
- `mode`: `single | routed | ensemble`
- `status`: `queued | running | completed | failed | cancelled | cancelling`
- `source_type`: `paper | preprint | review | thesis | web`
- `claim_type`: `fact | causal | comparative | methodological | definitional`
- `relation`: `supports | contradicts | neutral`
- `answer_style`: `exam | technical | concise`

## Minimum persistence requirements
- Store immutable run inputs.
- Store all evidence links used in final answer.
- Store provider usage and cost per stage.
- Store confidence features used for calibration auditing.
