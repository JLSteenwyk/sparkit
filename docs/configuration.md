# Configuration

Last updated: 2026-02-19

## Required environment variables
Set these provider keys in the runtime environment (do not hardcode):

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `KIMI_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## Database
- `DATABASE_URL` (example: `postgresql://postgres:postgres@localhost:5432/sparkit`)

## Optional provider overrides
- `KIMI_BASE_URL` (default: `https://api.moonshot.ai`)
- `KIMI_TEMPERATURE` (default: `1.0`)

## Notes
- `GEMINI_API_KEY` and `GOOGLE_API_KEY` are treated as alternatives for Google models.
- Keep keys in environment/secrets manager only.
- Kimi cost accounting uses `kimi-k2.5` rates per 1M tokens: input cache hit `$0.10`, input cache miss `$0.60`, output `$3.00`.
- Baseline capture command: `make baseline-capture` (writes outputs to `benchmarks/results/`).
- Official baseline capture command: `make baseline-capture-official`.
- Benchmark generation command: `make benchmark-generate`.
- HLE bio/chem subset generation: `make benchmark-generate-hle-biochem-20`.
- HLE bio/chem subset baseline capture: `make baseline-capture-hle-biochem-20`.
- HLE bio/chem direct-call baselines (single raw API call per question): `make baseline-capture-direct-calls-hle20`.
- Full benchmark eval command: `make eval-benchmark-full`.
- Drift check command (sample): `make drift-check-sample`.
