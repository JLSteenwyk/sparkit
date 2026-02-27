# MVP Checklist (Reboot)

## MVP definition
The reboot reaches MVP when all are true:

1. Federated scientific evidence contract implemented.
2. At least one real external provider integration live.
3. HLE-Gold downloader script available (FutureHouse HF dataset).
4. End-to-end benchmark script runs and emits predictions/report.

## Current status
- [x] Federated evidence contract
- [x] Real provider integration: PaperQA2 adapter (requires `paper-qa` + local paper directory via `PAPERQA_PAPER_DIRECTORY`)
- [x] HLE-Gold downloader script
- [x] End-to-end benchmark script

## Note
- Exa adapter is now integrated as a real web/science backfill provider.
- Elicit/Consensus/scite adapters are currently stubs pending official API integration details and credentials.
