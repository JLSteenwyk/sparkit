#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  cat <<'EOF'
Usage: scripts_run_claudep_proposals.sh [--num-runs N] [--max-parallel N] [--timeout-s N] [--output-dir DIR]

Run parallel claude -p proposal generation and write JSON outputs to proposals/.

Options:
  --num-runs N       Total number of runs (default: 20)
  --max-parallel N   Maximum concurrent runs, capped at 10 (default: 10)
  --timeout-s N      Per-run timeout in seconds (default: 900)
  --output-dir DIR   Output directory (default: proposals)
  -h, --help         Show this help message
EOF
}

NUM_RUNS=20
MAX_PARALLEL=10
TIMEOUT_S=900
OUTPUT_DIR="proposals"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-runs)
      NUM_RUNS="${2:-}"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="${2:-}"
      shift 2
      ;;
    --timeout-s)
      TIMEOUT_S="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v claude >/dev/null 2>&1; then
  echo "Missing required command: claude" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Missing required command: jq" >&2
  exit 1
fi

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || (( NUM_RUNS < 1 )); then
  echo "--num-runs must be a positive integer" >&2
  exit 2
fi
if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || (( MAX_PARALLEL < 1 )); then
  echo "--max-parallel must be a positive integer" >&2
  exit 2
fi
if ! [[ "$TIMEOUT_S" =~ ^[0-9]+$ ]] || (( TIMEOUT_S < 1 )); then
  echo "--timeout-s must be a positive integer" >&2
  exit 2
fi

if (( MAX_PARALLEL > 10 )); then
  MAX_PARALLEL=10
fi

mkdir -p "$OUTPUT_DIR"

BASE_PROMPT=$(cat <<'EOF'
You are conducting a full technical audit of the SPARKIT codebase.

Goal:
- Propose high-impact ways to improve SPARKIT performance on very challenging questions.
- Focus heavily on internet retrieval quality, evidence grounding, and answer correctness.
- Be creative and distinct from common ideas.

Return STRICT JSON only (no markdown) with this exact top-level schema:
{
  "proposal_id": "string",
  "title": "string",
  "distinctive_angle": "string",
  "summary": "string",
  "reasoning": "string explaining why this proposal should improve SPARKIT for hard QA",
  "expected_impact": {
    "accuracy_delta_pct_points": "number",
    "cost_impact": "increase|decrease|mixed",
    "latency_impact": "increase|decrease|mixed",
    "confidence": "low|medium|high"
  },
  "implementation_plan": [
    {"step": "string", "owner": "string", "effort": "low|medium|high"}
  ],
  "retrieval_improvements": [
    "string"
  ],
  "evaluation_plan": [
    "string"
  ],
  "risks": [
    "string"
  ]
}

Hard constraints:
- Proposal must be specific to SPARKIT architecture and workflows.
- Include at least 3 retrieval-focused improvements.
- Include at least 3 concrete evaluation checks.
- Keep it implementable.
EOF
)

build_prompt() {
  local index="$1"
  local total="$2"
  # Intentionally ignore run index/total for prompt text so every run is identical.
  # shellcheck disable=SC2034
  index="$index"
  # shellcheck disable=SC2034
  total="$total"
  cat <<EOF
$BASE_PROMPT
EOF
}

run_one() {
  local index="$1"
  local total="$2"
  local pad
  pad=$(printf "%03d" "$index")
  local prompt
  prompt="$(build_prompt "$index" "$total")"

  local raw_file="$OUTPUT_DIR/proposal_${pad}.raw.txt"
  local json_file="$OUTPUT_DIR/proposal_${pad}.json"
  local err_file="$OUTPUT_DIR/proposal_${pad}.error.json"
  local status_file="$OUTPUT_DIR/proposal_${pad}.status.json"
  local tmp_out
  local tmp_err
  tmp_out="$(mktemp)"
  tmp_err="$(mktemp)"

  local rc=0
  echo "[start] proposal_${pad} (index=${index}/${total})" >&2
  if ! timeout "${TIMEOUT_S}s" claude -p "$prompt" >"$tmp_out" 2>"$tmp_err"; then
    rc=$?
  fi

  cat "$tmp_out" > "$raw_file"
  if [[ -s "$tmp_err" ]]; then
    {
      echo
      echo "[stderr]"
      cat "$tmp_err"
    } >> "$raw_file"
  fi

  local ok="false"
  local error_msg=""

  if [[ "$rc" -eq 0 ]] && jq -e . "$tmp_out" >/dev/null 2>&1; then
    cp "$tmp_out" "$json_file"
    ok="true"
    rm -f "$err_file"
  else
    error_msg="returncode=${rc}, json_valid=false"
    jq -n \
      --argjson index "$index" \
      --arg error "$error_msg" \
      --arg raw_path "$raw_file" \
      '{index:$index,error:$error,raw_path:$raw_path}' > "$err_file"
    rm -f "$json_file"
  fi

  jq -n \
    --argjson index "$index" \
    --argjson success "$ok" \
    --arg error "$error_msg" \
    --arg json_file "$( [[ "$ok" == "true" ]] && echo "proposal_${pad}.json" || echo "" )" \
    --arg raw_file "proposal_${pad}.raw.txt" \
    '{
      index:$index,
      success:$success,
      error:($error | select(length>0)),
      json_file:($json_file | select(length>0)),
      raw_file:$raw_file
    }' > "$status_file"

  if [[ "$ok" == "true" ]]; then
    echo "[done]  proposal_${pad} success" >&2
  else
    echo "[done]  proposal_${pad} failed (${error_msg})" >&2
  fi

  rm -f "$tmp_out" "$tmp_err"
}

for ((i=1; i<=NUM_RUNS; i++)); do
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n || true
  done
  run_one "$i" "$NUM_RUNS" &
done

# Progress monitor until all jobs finish.
while (( $(jobs -rp | wc -l) > 0 )); do
  completed="$(ls "$OUTPUT_DIR"/proposal_*.status.json 2>/dev/null | wc -l | tr -d ' ')"
  success_so_far=0
  fail_so_far=0
  if (( completed > 0 )); then
    success_so_far="$(jq -s '[.[] | select(.success == true)] | length' "$OUTPUT_DIR"/proposal_*.status.json 2>/dev/null || echo 0)"
    fail_so_far="$(jq -s '[.[] | select(.success != true)] | length' "$OUTPUT_DIR"/proposal_*.status.json 2>/dev/null || echo 0)"
  fi
  echo "[progress] completed=${completed}/${NUM_RUNS} success=${success_so_far} failed=${fail_so_far}" >&2
  sleep 2
done

wait || true

mapfile -t status_files < <(ls "$OUTPUT_DIR"/proposal_*.status.json 2>/dev/null | sort)
if (( ${#status_files[@]} == 0 )); then
  echo "No status files were produced." >&2
  exit 1
fi

success_count="$(jq -s '[.[] | select(.success == true)] | length' "${status_files[@]}")"
failure_count="$(jq -s '[.[] | select(.success != true)] | length' "${status_files[@]}")"
created_at="$(date -u +"%Y%m%dT%H%M%SZ")"
started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
completed_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

jq -n \
  --arg created_at "$created_at" \
  --arg started_at "$started_at" \
  --arg completed_at "$completed_at" \
  --argjson num_runs "$NUM_RUNS" \
  --argjson max_parallel "$MAX_PARALLEL" \
  --argjson timeout_s "$TIMEOUT_S" \
  --argjson success_count "$success_count" \
  --argjson failure_count "$failure_count" \
  --slurpfile proposals <(jq -s '.' "${status_files[@]}") \
  '{
    created_at:$created_at,
    started_at:$started_at,
    completed_at:$completed_at,
    num_runs:$num_runs,
    max_parallel:$max_parallel,
    timeout_s:$timeout_s,
    success_count:$success_count,
    failure_count:$failure_count,
    proposals:$proposals[0]
  }' > "$OUTPUT_DIR/proposals_index.json"

cat "$OUTPUT_DIR/proposals_index.json"

if (( success_count > 0 )); then
  exit 0
fi
exit 1
