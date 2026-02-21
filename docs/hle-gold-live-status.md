# HLE Gold Live Status

Last updated (UTC): 2026-02-21 15:17:10Z

| Run | Method | Model | Status | N | Avg Rubric | Brier | ECE | Avg Cost/Q ($) | Total Cost ($) | Failures | Bio/Med Rubric | Chem Rubric | Chem-Bio Δ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hle_gold_single_core_v2_20260221T054301Z` | `single_grok` | `grok-4-0709` | completed | 149 | 0.636913 | 0.121356 | 0.109027 | 0.018634 | 2.776491 | 0 | 0.628972 | 0.657143 | 0.028171 |
| `hle_gold_single_core_v2_20260221T054301Z` | `single_gemini` | `gemini-3-pro-preview` | completed | 149 | 0.612458 | 0.047054 | 0.130604 | 0.019336 | 2.881072 | 0 | 0.608002 | 0.623810 | 0.015807 |
| `hle_gold_single_core_v2_20260221T054301Z` | `single_anthropic` | `claude-opus-4-6` | completed | 149 | 0.568960 | 0.298180 | 0.234329 | 0.025625 | 3.818075 | 0 | 0.512266 | 0.713393 | 0.201127 |
| `hle_gold_single_core_v2_20260221T054301Z` | `single_mistral` | `mistral-large-2512` | completed | 149 | 0.508012 | 0.328928 | 0.304161 | 0.007049 | 1.050278 | 0 | 0.474182 | 0.594196 | 0.120014 |
| `hle_gold_single_core_v2_20260221T054301Z` | `single_openai` | `gpt-5.2` | completed | 149 | 0.484564 | 0.389543 | 0.391879 | 0.008391 | 1.250305 | 0 | 0.453446 | 0.563839 | 0.110393 |
| `hle_gold_direct_batch_b_20260221T004327Z` | `direct_deepseek` | `deepseek-reasoner` | completed | 149 | 0.453398 | 0.271644 | 0.381208 | 0.000383 | 0.057013 | 0 | 0.447897 | 0.467411 | 0.019514 |
| `hle_gold_single_core_v2_20260221T054301Z` | `single_deepseek` | `deepseek-reasoner` | completed | 149 | 0.423364 | 0.467854 | 0.499765 | 0.000281 | 0.041812 | 0 | 0.405432 | 0.469048 | 0.063615 |
| `hle_gold_single_gemini31_20260221T023731Z` | `single_gemini` | `gemini-3.1-pro-preview` | completed | 149 | 0.393876 | 0.425798 | 0.480940 | 0.007330 | 1.092186 | 0 | 0.414720 | 0.340774 | -0.073946 |
| `hle_gold_single_core_v2_20260221T054301Z` | `single_kimi` | `kimi-k2-turbo-preview` | completed | 149 | 0.391904 | 0.478250 | 0.531275 | 0.001931 | 0.287683 | 0 | 0.365771 | 0.458482 | 0.092711 |
| `hle_gold_direct_anthropic_sonnet_20260221T023544Z` | `direct_anthropic` | `claude-sonnet-4-6` | completed | 149 | 0.280998 | 0.412364 | 0.615570 | 0.008775 | 1.307511 | 1 | 0.273715 | 0.299554 | 0.025839 |
| `hle_gold_direct_mistral_split_20260221T024803Z` | `direct_mistral` | `mistral-large-2512` | completed | 149 | 0.252810 | 0.801429 | 0.886242 | 0.001900 | 0.283146 | 0 | 0.235164 | 0.297768 | 0.062604 |
| `hle_gold_direct_batch_a_20260221T004327Z` | `direct_anthropic` | `claude-opus-4-6` | completed | 149 | 0.168876 | 0.431566 | 0.622886 | 0.013160 | 1.960875 | 1 | 0.128446 | 0.271875 | 0.143429 |
| `hle_gold_direct_grok4_fast_nonreason_20260221T023544Z` | `direct_grok` | `grok-4-fast-non-reasoning` | completed | 149 | 0.106963 | 0.844094 | 0.908725 | 0.000133 | 0.019772 | 0 | 0.089544 | 0.151339 | 0.061795 |
| `hle_gold_direct_openai_pro_v3_20260221T060305Z` | `direct_openai` | `gpt-5.2-pro` | completed | 149 | 0.095512 | 0.316634 | 0.434832 | 0.060737 | 9.049866 | 53 | 0.102687 | 0.077232 | -0.025455 |
| `hle_gold_direct_batch_a_20260221T004327Z` | `direct_openai` | `gpt-5.2` | completed | 149 | 0.093247 | 0.480379 | 0.676309 | 0.001820 | 0.271147 | 1 | 0.082185 | 0.121429 | 0.039244 |
| `hle_gold_direct_grok4_fast_reason_20260221T023544Z` | `direct_grok` | `grok-4-fast-reasoning` | completed | 149 | 0.057634 | 0.760839 | 0.847651 | 0.000122 | 0.018235 | 5 | 0.041706 | 0.098214 | 0.056509 |
| `hle_gold_direct_gemini31_20260221T023544Z` | `direct_gemini` | `gemini-3.1-pro-preview` | completed | 149 | 0.045554 | 0.528289 | 0.689262 | 0.000971 | 0.144622 | 1 | 0.041706 | 0.055357 | 0.013652 |
| `hle_gold_direct_batch_b_20260221T004327Z` | `direct_kimi` | `kimi-k2-turbo-preview` | completed | 149 | 0.027055 | 0.764044 | 0.866174 | 0.000313 | 0.046664 | 0 | 0.018049 | 0.050000 | 0.031951 |
| `hle_gold_direct_batch_a_20260221T004327Z` | `direct_gemini` | `gemini-3-pro-preview` | completed | 149 | 0.022651 | 0.586613 | 0.731409 | 0.000984 | 0.146674 | 0 | 0.009463 | 0.056250 | 0.046787 |

Refresh command: `./venv/bin/python scripts_update_hle_live_status.py`