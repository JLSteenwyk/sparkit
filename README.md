# SPARKIT (Reboot)

Accuracy-first scientific RAG, rebuilt around federated evidence providers.

## Current scope
This reboot currently includes:
- A normalized scientific evidence schema.
- Provider adapter interfaces.
- Provider stubs for:
  - PaperQA2
  - Elicit
  - Consensus
  - scite
- A federation orchestrator for dedupe + weighted ranking.

## Repository layout
- `src/sparkit/evidence/schema.py`: shared evidence model and enums.
- `src/sparkit/providers/`: provider adapters and stubs.
- `src/sparkit/orchestrator/federation.py`: evidence federation logic.
- `docs/federated-science-rag-plan.md`: architecture and implementation plan.
- `scripts_example_federation.py`: minimal example entrypoint.

## Quick start
1. Create venv:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install package:

```bash
pip install -e .
```

3. Optional integrations:

```bash
pip install -e .[benchmark]
pip install -e .[paperqa2]
```

PaperQA2 adapter runtime note:
- Set `PAPERQA_PAPER_DIRECTORY` to a local paper corpus/index directory before using the PaperQA2 provider.
- Bootstrap a local corpus quickly (Exa + arXiv + bioRxiv):

```bash
python scripts_build_paperqa_directory.py \
  --questions benchmarks/hle_gold/questions_full.json \
  --output-dir data/paperqa_papers \
  --max-questions 50 \
  --results-per-question 8 \
  --arxiv-per-question 3 \
  --biorxiv-per-question 3
export PAPERQA_PAPER_DIRECTORY="$(pwd)/data/paperqa_papers"
```

Notes:
- `--include-exa`, `--include-arxiv`, and `--include-biorxiv` can be used to force specific source subsets.
- If no `--include-*` flags are provided, the builder enables all three by default.

Exa adapter runtime note:
- Set `EXA_API_KEY` to enable Exa-backed retrieval.

Synthesis runtime note (MCQ answering):
- `decide_mcq_from_evidence` now defaults to LLM synthesis over top ranked evidence.
- Set `SPARKIT_SYNTH_MODEL` (default: `gpt-5.2`).
- Optional controls:
  - `SPARKIT_SYNTH_EVIDENCE_LIMIT` (default: `12`)
  - `SPARKIT_SYNTH_EVIDENCE_CHARS` (default: `900` per evidence item)
- If LLM synthesis fails, SPARKIT automatically falls back to heuristic overlap scoring.
- Optional low-latency mode: `--decision-mode fast_consensus` in `scripts_benchmark_hle_gold.py`
  - Three quick votes (heuristic, citation-weighted, tiny LLM judge) run first.
  - If at least 2 votes agree, return immediately.
  - Otherwise, fall back to full LLM synthesis.
  - Tiny judge model defaults to `SPARKIT_FAST_JUDGE_MODEL=gpt-5-nano`.
- Optional mode: `--decision-mode nano_consensus10`
  - Runs 10 tiny-LLM votes (`gpt-5-nano` by default) and returns majority answer.
  - Falls back to full synthesis on tie or no valid votes.
  - Tunables: `SPARKIT_NANO_CONSENSUS_*` env vars.

Federation runtime note (retrieval + reranking):
- `build_evidence_pack` now performs MCQ-aware multi-query retrieval by default:
  - base stem query
  - additional per-option query variants
- Retrieved evidence is deduplicated and reranked before synthesis using:
  - evidence quality score
  - lexical relevance to stem
  - option-specific overlap
  - provider prior
- `FederationConfig` knobs:
  - `enable_mcq_option_queries` (default: `True`)
  - `max_query_variants` (default: `6`)

4. Run local federation smoke:

```bash
python scripts_example_federation.py
```

## Next steps
1. Implement remaining real provider integrations.
2. Improve evidence scoring/reranking for science-specific quality.
3. Run HLE-Gold benchmark loops and compare provider ablations.

## HLE-Gold benchmark
Download HLE-Gold (FutureHouse via Hugging Face):

```bash
python scripts_download_hle_gold.py \
  --dataset futurehouse/hle-gold-bio-chem \
  --split train \
  --output benchmarks/hle_gold/questions_full.json
```

Run MVP benchmark:

```bash
python scripts_benchmark_hle_gold.py \
  --questions benchmarks/hle_gold/questions_full.json \
  --providers paperqa2,exa,elicit,consensus,scite
```
