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
- Bootstrap a local corpus quickly (using Exa search snippets):

```bash
python scripts_build_paperqa_directory.py \
  --questions benchmarks/hle_gold/questions_full.json \
  --output-dir data/paperqa_papers \
  --max-questions 50 \
  --results-per-question 8
export PAPERQA_PAPER_DIRECTORY="$(pwd)/data/paperqa_papers"
```

Exa adapter runtime note:
- Set `EXA_API_KEY` to enable Exa-backed retrieval.

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
