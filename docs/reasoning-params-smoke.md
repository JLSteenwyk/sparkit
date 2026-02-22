# Reasoning Parameter Smoke Test Progress

## Task Progress
| Step | Status | Notes | Artifact |
|---|---|---|---|
| Add one-question cross-provider smoke runner | Completed | Added `scripts_reasoning_smoke.py` to test OpenAI/Anthropic/Gemini directly with high-effort reasoning/thinking settings. | `scripts_reasoning_smoke.py` |
| Run first smoke pass | Completed | OpenAI + Anthropic succeeded; Gemini failed due wrong field location for `thinkingConfig`. | `benchmarks/results/reasoning_smoke_20260222T172054Z.json` |
| Diagnose Gemini parameter shape | Completed | Confirmed working shape is `generationConfig.thinkingConfig.thinkingBudget`. | terminal probe |
| Patch runner with corrected Gemini thinking config | Completed | Updated payload structure in script. | `scripts_reasoning_smoke.py` |
| Re-run final smoke pass | Completed | All providers returned HTTP 200 and valid answer text. | `benchmarks/results/reasoning_smoke_20260222T172431Z.json` |

## Provider Settings Confirmed
| Provider | Model (current env default) | High-effort parameter used | Status | Evidence |
|---|---|---|---|---|
| OpenAI | `gpt-5.2` | `reasoning: {"effort":"xhigh"}` via `/v1/responses` | Works | `benchmarks/results/reasoning_smoke_20260222T172431Z.json` |
| Anthropic | `claude-opus-4-6` | `thinking: {"type":"enabled","budget_tokens":3000}` | Works | `benchmarks/results/reasoning_smoke_20260222T172431Z.json` |
| Gemini | `gemini-3-pro-preview` | `generationConfig.thinkingConfig: {"thinkingBudget":24576}` | Works | `benchmarks/results/reasoning_smoke_20260222T172431Z.json` |

## Notes
- Gemini rejects top-level `thinkingConfig` with HTTP 400 (`Unknown name "thinkingConfig"`). It must be nested under `generationConfig`.
- OpenAI validation probe also showed reasoning token accounting (`output_tokens_details.reasoning_tokens`) and `output` entries including `reasoning`.
- Anthropic validation probe returned `content` blocks with both `thinking` and `text` types.
