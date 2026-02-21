# HLE-Gold Triplicate Runbook

Last updated: 2026-02-21

## Objective
Run HLE-gold (`benchmarks/hle_gold_bio_chem/questions_full.json`, 149 questions) in triplicate for:
- SPARKIT benchmark configs (`single_*`, `routed_frontier`, `routed_frontier_plus`)
- Direct single-call baselines

All long runs should be launched in `tmux`.

## Preflight
1. `source ~/.bashrc`
2. `source venv/bin/activate`
3. Confirm keys:
   - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`, `KIMI_API_KEY`, `DEEPSEEK_API_KEY`, `GROK_API_KEY`, `MISTRAL_API_KEY`
4. Ensure retry/timeout settings for reliability:
   - `export DIRECT_CALL_MAX_ATTEMPTS=3`
   - `export DIRECT_CALL_RETRY_BACKOFF_S=0.8`
   - `export GROK_TIMEOUT_S=90`
   - `export DEEPSEEK_TIMEOUT_S=60`

## SPARKIT triplicate (3 tmux sessions)
Use a balanced parallel profile to avoid overload:
- `--parallel-workers 4`
- `--parallel-configs 3`

```bash
tmux new-session -d -s hle_trip_sparkit_1 \
"source ~/.bashrc && source venv/bin/activate && ./venv/bin/python scripts_capture_baselines.py \
  --questions benchmarks/hle_gold_bio_chem/questions_full.json \
  --label hle_gold_triplicate_sparkit_r1 \
  --configs single_openai,single_openai_pro,single_anthropic,single_anthropic_sonnet,single_gemini,single_kimi,single_deepseek,single_grok,single_mistral,routed_frontier,routed_frontier_plus \
  --parallel-workers 4 --parallel-configs 3 --skip-missing-keys"

tmux new-session -d -s hle_trip_sparkit_2 \
"source ~/.bashrc && source venv/bin/activate && ./venv/bin/python scripts_capture_baselines.py \
  --questions benchmarks/hle_gold_bio_chem/questions_full.json \
  --label hle_gold_triplicate_sparkit_r2 \
  --configs single_openai,single_openai_pro,single_anthropic,single_anthropic_sonnet,single_gemini,single_kimi,single_deepseek,single_grok,single_mistral,routed_frontier,routed_frontier_plus \
  --parallel-workers 4 --parallel-configs 3 --skip-missing-keys"

tmux new-session -d -s hle_trip_sparkit_3 \
"source ~/.bashrc && source venv/bin/activate && ./venv/bin/python scripts_capture_baselines.py \
  --questions benchmarks/hle_gold_bio_chem/questions_full.json \
  --label hle_gold_triplicate_sparkit_r3 \
  --configs single_openai,single_openai_pro,single_anthropic,single_anthropic_sonnet,single_gemini,single_kimi,single_deepseek,single_grok,single_mistral,routed_frontier,routed_frontier_plus \
  --parallel-workers 4 --parallel-configs 3 --skip-missing-keys"
```

## Direct-call triplicate (3 tmux sessions)

```bash
tmux new-session -d -s hle_trip_direct_1 \
"source ~/.bashrc && source venv/bin/activate && DIRECT_CALL_MAX_ATTEMPTS=3 DIRECT_CALL_RETRY_BACKOFF_S=0.8 GROK_TIMEOUT_S=90 DEEPSEEK_TIMEOUT_S=60 ./venv/bin/python scripts_capture_direct_baselines.py \
  --questions benchmarks/hle_gold_bio_chem/questions_full.json \
  --label hle_gold_triplicate_direct_r1 \
  --providers openai,anthropic,gemini,kimi,deepseek,grok,mistral \
  --skip-missing-keys"

tmux new-session -d -s hle_trip_direct_2 \
"source ~/.bashrc && source venv/bin/activate && DIRECT_CALL_MAX_ATTEMPTS=3 DIRECT_CALL_RETRY_BACKOFF_S=0.8 GROK_TIMEOUT_S=90 DEEPSEEK_TIMEOUT_S=60 ./venv/bin/python scripts_capture_direct_baselines.py \
  --questions benchmarks/hle_gold_bio_chem/questions_full.json \
  --label hle_gold_triplicate_direct_r2 \
  --providers openai,anthropic,gemini,kimi,deepseek,grok,mistral \
  --skip-missing-keys"

tmux new-session -d -s hle_trip_direct_3 \
"source ~/.bashrc && source venv/bin/activate && DIRECT_CALL_MAX_ATTEMPTS=3 DIRECT_CALL_RETRY_BACKOFF_S=0.8 GROK_TIMEOUT_S=90 DEEPSEEK_TIMEOUT_S=60 ./venv/bin/python scripts_capture_direct_baselines.py \
  --questions benchmarks/hle_gold_bio_chem/questions_full.json \
  --label hle_gold_triplicate_direct_r3 \
  --providers openai,anthropic,gemini,kimi,deepseek,grok,mistral \
  --skip-missing-keys"
```

## Monitoring
```bash
tmux ls
tmux attach -t hle_trip_sparkit_1
tmux attach -t hle_trip_direct_1
```

Check manifests as they appear:
```bash
find benchmarks/results -maxdepth 2 -name manifest.json | rg "hle_gold_triplicate"
```

## Failure tracking and targeted reruns
SPARKIT baseline capture now writes per-config failure artifacts:
- `failures_<config>.json` (records `id`, `run_id`, `status`, `error`)
- `manifest.json` includes `failure_count` and `failed_question_ids` per config

Direct baseline manifests include:
- `failure_count`
- `failed_question_ids`

Build a rerun subset from failed IDs (example for one config):
```bash
./venv/bin/python - <<'PY'
import json
from pathlib import Path
run_dir=Path("benchmarks/results/hle_gold_triplicate_sparkit_r1_<timestamp>")
all_q=json.loads(Path("benchmarks/hle_gold_bio_chem/questions_full.json").read_text())
failed=json.loads((run_dir/"failures_single_grok.json").read_text())
ids={row["id"] for row in failed}
subset=[q for q in all_q if q.get("id") in ids]
out=run_dir/"questions_failed_single_grok.json"
out.write_text(json.dumps(subset, indent=2))
print("wrote", out, "n=", len(subset))
PY
```

Then rerun only those questions:
```bash
source ~/.bashrc && source venv/bin/activate
./venv/bin/python scripts_capture_baselines.py \
  --questions benchmarks/results/<run_slug>/questions_failed_single_grok.json \
  --label hle_gold_failed_rerun_single_grok \
  --configs single_grok \
  --parallel-workers 2 --parallel-configs 1 --skip-missing-keys
```

## Post-run consolidation target
Create one summary table per run group (SPARKIT/direct), then aggregate mean and variance across 3 runs for:
- rubric score
- Brier score
- ECE
- total/avg cost
- total/avg latency
