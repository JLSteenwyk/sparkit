# API Contracts (V1)

Last updated: 2026-02-20

## Principles
- Every run is traceable and reproducible.
- Responses include confidence and citation-grounding metadata.
- Long-running tasks are asynchronous with pollable run state.

## Endpoints

### POST `/v1/ask`
Create a new research run.

Request:
```json
{
  "question": "What are the dominant causes of decoherence in superconducting transmon qubits at millikelvin temperatures?",
  "mode": "single",
  "providers": ["openai"],
  "constraints": {
    "max_latency_s": null,
    "max_cost_usd": 3.0,
    "min_sources": 5,
    "synthesis_max_tokens": 900
  },
  "answer_style": "technical",
  "include_trace": false,
  "metadata": {
    "request_id": "client-123",
    "user_id": "user-42"
  }
}
```

Validation:
- `question` required, non-empty string.
- `mode` required: `single | routed | ensemble`.
- `providers` optional in routed/ensemble modes, required in `single` mode.
- `constraints.max_latency_s` optional; when provided range is `10..7200`.
- `constraints.max_cost_usd` range: `0.05..100.0`.
- `constraints.min_sources` range: `1..50`.
- `constraints.synthesis_max_tokens` optional; when provided range is `128..4096`.
- `answer_style`: `exam | technical | concise`.

Response (`202 Accepted`):
```json
{
  "run_id": "run_01JT7A9J8Q6D9J4S4KB9M0Y9ZF",
  "status": "queued",
  "created_at": "2026-02-19T13:40:00Z",
  "poll_url": "/v1/runs/run_01JT7A9J8Q6D9J4S4KB9M0Y9ZF"
}
```

Errors:
- `400` invalid request
- `401` unauthorized
- `429` rate limited
- `500` internal error

### GET `/v1/runs/{run_id}`
Fetch run status and (when completed) final result.

Response (`200 OK`, running):
```json
{
  "run_id": "run_01JT7A9J8Q6D9J4S4KB9M0Y9ZF",
  "status": "running",
  "mode": "routed",
  "progress": {
    "stage": "verification_round",
    "percent": 72
  },
  "usage": {
    "cost_usd": 1.41,
    "latency_s": 63.8,
    "tokens_input": 128402,
    "tokens_output": 4512
  },
  "created_at": "2026-02-19T13:40:00Z",
  "updated_at": "2026-02-19T13:41:04Z"
}
```

Response (`200 OK`, completed):
```json
{
  "run_id": "run_01JT7A9J8Q6D9J4S4KB9M0Y9ZF",
  "status": "completed",
  "mode": "ensemble",
  "answer": {
    "final_text": "At millikelvin temperatures, dominant decoherence channels in transmon qubits are dielectric loss ...",
    "answer_confidence": 0.81,
    "claim_confidences": [
      {
        "claim_id": "clm_1",
        "confidence": 0.86
      },
      {
        "claim_id": "clm_2",
        "confidence": 0.73
      }
    ],
    "uncertainty_reasons": [
      "Evidence heterogeneity across fabrication stacks"
    ]
  },
  "citations": [
    {
      "claim_id": "clm_1",
      "doc_id": "doc_a",
      "passage_id": "psg_14"
    }
  ],
  "usage": {
    "cost_usd": 2.12,
    "latency_s": 94.1,
    "tokens_input": 185332,
    "tokens_output": 7231
  },
  "created_at": "2026-02-19T13:40:00Z",
  "updated_at": "2026-02-19T13:41:54Z"
}
```

Terminal statuses:
- `completed`
- `failed`
- `cancelled`

### GET `/v1/runs/{run_id}/trace`
Return explainability and audit trace for a run.

Response (`200 OK`):
```json
{
  "run_id": "run_01JT7A9J8Q6D9J4S4KB9M0Y9ZF",
  "stages": [
    {
      "name": "plan",
      "status": "completed",
      "model": "claude-x",
      "started_at": "2026-02-19T13:40:02Z",
      "ended_at": "2026-02-19T13:40:07Z",
      "artifacts": {
        "subquestions": [
          "What known loss channels dominate T1?",
          "How does flux noise impact dephasing?"
        ]
      }
    },
    {
      "name": "retrieval_round_1",
      "status": "completed",
      "artifacts": {
        "queries": ["transmon decoherence dielectric loss", "flux noise transmon dephasing"],
        "documents_retrieved": 42
      }
    }
  ],
  "provider_usage": [
    {
      "provider": "openai",
      "model": "gpt-x",
      "tokens_input": 80231,
      "tokens_output": 3221,
      "cost_usd": 0.99
    }
  ],
  "quality_gates": {
    "citation_coverage": 0.93,
    "unsupported_claims": 0,
    "contradiction_flags": 1
  }
}
```

### POST `/v1/runs/{run_id}/cancel`
Cancel an active run.

Response (`202 Accepted`):
```json
{
  "run_id": "run_01JT7A9J8Q6D9J4S4KB9M0Y9ZF",
  "status": "cancelling"
}
```

Errors:
- `404` run not found
- `409` run is already terminal (`completed|failed|cancelled`) or already `cancelling`

## Error shape (standard)
```json
{
  "error": {
    "code": "invalid_request",
    "message": "constraints.max_cost_usd must be >= 0.05",
    "details": {
      "field": "constraints.max_cost_usd"
    }
  }
}
```

## Notes
- V1 omits streaming; add later if needed.
- Trace may redact provider prompts by policy, but should preserve structural artifacts.
- For insufficient evidence outcomes, `status` remains `completed` and answer includes explicit limitation section.
- `POST /v1/ask` now records reproducibility metadata (prompt/config versions and stable fingerprint) in run storage and trace planning artifacts.
- When `constraints.max_latency_s` is omitted (`null`), no latency cap is enforced by default.
- Synthesis trace artifacts include `claim_clusters` and `section_summaries`.
