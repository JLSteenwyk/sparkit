# Configuration

Last updated: 2026-02-20

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
- `OPENAI_MODEL` (default: `gpt-5.2`)
- `ANTHROPIC_MODEL` (default: `claude-opus-4-6`)
- `KIMI_MODEL` (default: `kimi-k2.5`)
- `GEMINI_MODEL` (default: `gemini-3-pro-preview`)

## Notes
- `GEMINI_API_KEY` and `GOOGLE_API_KEY` are treated as alternatives for Google models.
- Keep keys in environment/secrets manager only.
- Kimi cost accounting uses `kimi-k2.5` rates per 1M tokens: input cache hit `$0.10`, input cache miss `$0.60`, output `$3.00`.
- Cost precision note: only configured priced models use exact token pricing; other providers still use deterministic synthesis-stage heuristics.
- Direct-call quality note: empty parsed answers are now counted as failures (`empty_answer_text`).
- Kimi response note: Kimi responses with empty `message.content` are treated as failures to avoid silently accepting reasoning-only outputs.
- Baseline capture command: `make baseline-capture` (writes outputs to `benchmarks/results/`).
- Official baseline capture command: `make baseline-capture-official`.
- Benchmark generation command: `make benchmark-generate`.
- HLE bio/chem subset generation: `make benchmark-generate-hle-biochem-20`.
- HLE bio/chem subset baseline capture: `make baseline-capture-hle-biochem-20`.
- HLE bio/chem direct-call baselines (single raw API call per question): `make baseline-capture-direct-calls-hle20`.
- Repeated-slice benchmark with CI: `make benchmark-repeated-slices`.
- Full benchmark eval command: `make eval-benchmark-full`.
- Drift check command (sample): `make drift-check-sample`.
