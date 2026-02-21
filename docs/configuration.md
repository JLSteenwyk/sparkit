# Configuration

Last updated: 2026-02-21

## Required environment variables
Set these provider keys in the runtime environment (do not hardcode):

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `KIMI_API_KEY`
- `DEEPSEEK_API_KEY`
- `GROK_API_KEY`
- `MISTRAL_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## Database
- `DATABASE_URL` (example: `postgresql://postgres:postgres@localhost:5432/sparkit`)

## Optional provider overrides
- `KIMI_BASE_URL` (default: `https://api.moonshot.ai`)
- `KIMI_TEMPERATURE` (default: `1.0`)
- `OPENAI_MODEL` (default: `gpt-5.2`)
- `ANTHROPIC_MODEL` (default: `claude-opus-4-6`)
- `KIMI_MODEL` (default: `kimi-k2-turbo-preview`)
- `GEMINI_MODEL` (default: `gemini-3-pro-preview`)
  - Also supported in exact pricing: `gemini-3.1-pro-preview`
- `DEEPSEEK_MODEL` (default: `deepseek-reasoner`)
- `GROK_MODEL` (default: `grok-4-0709`)
  - Also supported in exact pricing: `grok-4-fast-reasoning`, `grok-4-fast-non-reasoning`
- `MISTRAL_MODEL` (default: `mistral-large-2512`)
- `DEEPSEEK_BASE_URL` (default: `https://api.deepseek.com`)
- `GROK_BASE_URL` (default: `https://api.x.ai`)
- `MISTRAL_BASE_URL` (default: `https://api.mistral.ai`)
- `SPARKIT_PROVIDER_TIMEOUT_S` (default: `35`)
- `<PROVIDER>_TIMEOUT_S` per-provider override (examples: `GROK_TIMEOUT_S`, `DEEPSEEK_TIMEOUT_S`)
- `DIRECT_CALL_MAX_ATTEMPTS` (default: `3`, applies to direct single-call baselines)
- `DIRECT_CALL_RETRY_BACKOFF_S` (default: `0.8`, exponential backoff base seconds)
- `SPARKIT_MODEL_PRICING_JSON` (optional per-model pricing override map)

## Benchmark model presets
- `single_openai`: provider `openai` using current `OPENAI_MODEL` (default `gpt-5.2`)
- `single_openai_pro`: provider `openai` with override `OPENAI_MODEL=gpt-5.2-pro`
- `single_anthropic`: provider `anthropic` using current `ANTHROPIC_MODEL` (default `claude-opus-4-6`)
- `single_anthropic_sonnet`: provider `anthropic` with override `ANTHROPIC_MODEL=claude-sonnet-4-6`
- `single_gemini`: provider `gemini` using current `GEMINI_MODEL` (default `gemini-3-pro-preview`)
- `single_kimi`: provider `kimi` using current `KIMI_MODEL` (default `kimi-k2-turbo-preview`)
- `single_deepseek`: provider `deepseek` using current `DEEPSEEK_MODEL` (default `deepseek-reasoner`)
- `single_grok`: provider `grok` using current `GROK_MODEL` (default `grok-4-0709`)
- `single_mistral`: provider `mistral` using current `MISTRAL_MODEL` (default `mistral-large-2512`)
- `routed_frontier_plus`: routed config over OpenAI + Anthropic + Gemini + DeepSeek + Grok + Mistral + Kimi

## Runtime tuning knobs (per request)
- `constraints.synthesis_max_tokens` overrides synthesis token budget per run.
- Current mode defaults when unset:
  - `single_*` / `routed`: `700`
  - `ensemble`: `700` (per draft)

## Notes
- `GEMINI_API_KEY` and `GOOGLE_API_KEY` are treated as alternatives for Google models.
- Keep keys in environment/secrets manager only.
- Default exact pricing map (per 1M tokens):
  - `gpt-5.2`: input cache hit `$0.175`, input cache miss `$1.75`, output `$14.00`
  - `gpt-5.2-pro`: input cache hit `$21.00`, input cache miss `$21.00`, output `$168.00`
  - `claude-opus-4-6`: input cache hit `$0.50`, input cache miss `$5.00`, output `$25.00`
  - `claude-sonnet-4-6`: input cache hit `$0.30`, input cache miss `$3.00`, output `$15.00`
  - `gemini-3-pro-preview`: input cache hit `$0.20`, input cache miss `$2.00`, output `$12.00` (prompt >200k tokens tier: `$0.40`, `$4.00`, `$18.00`)
  - `gemini-3.1-pro-preview`: input cache hit `$0.20`, input cache miss `$2.00`, output `$12.00` (prompt >200k tokens tier: `$0.40`, `$4.00`, `$18.00`)
  - `kimi-k2-turbo-preview`: input cache hit `$0.10`, input cache miss `$0.60`, output `$3.00`
  - `deepseek-reasoner`: input cache hit `$0.028`, input cache miss `$0.28`, output `$0.42`
  - `grok-4-0709`: input cache hit `$0.75`, input cache miss `$3.00`, output `$15.00`
  - `grok-4-fast-reasoning`: input cache hit `$0.20`, input cache miss `$0.20`, output `$0.50`
  - `grok-4-fast-non-reasoning`: input cache hit `$0.20`, input cache miss `$0.20`, output `$0.50`
  - `mistral-large-2512`: input cache hit `$2.00`, input cache miss `$2.00`, output `$6.00`
- Cost precision note: exact generation cost is computed when model pricing is configured (built-in defaults + `SPARKIT_MODEL_PRICING_JSON` overrides). Unknown models fall back to deterministic synthesis-stage estimates.
- `SPARKIT_MODEL_PRICING_JSON` supports either:
  - `{"provider:model":{"input_cache_hit":...,"input_cache_miss":...,"output":...}}`
  - `{"provider":{"model":{"input_cache_hit":...,"input_cache_miss":...,"output":...}}}`
- Latency policy note: default `max_latency_s` is unset (`null`), so no latency cap is applied unless explicitly set.
- Direct-call quality note: empty parsed answers are now counted as failures (`empty_answer_text`).
- DeepSeek direct-call note: when DeepSeek returns empty `message.content` but populated `reasoning_content`, provider adapter falls back to reasoning text to avoid false hard-failures.
- Timeout note: tuned provider defaults are Grok `60s`, DeepSeek `45s`, and `35s` for other providers unless overridden by env vars.
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
