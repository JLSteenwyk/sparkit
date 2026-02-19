# SPARKIT

STEM literature fetching and reviewing agent.

## Repository layout
- `services/api_gateway`: Public API service (`/v1/ask`, run status, trace).
- `services/orchestrator`: Multi-round workflow orchestration service.
- `services/retrieval_service`: Federated literature retrieval adapters.
- `services/ingestion_service`: Document ingestion/parsing (HTML/PDF).
- `services/eval_service`: Benchmark and calibration evaluation harness.
- `shared/schemas`: Shared Pydantic models used across services.
- `alembic`: Database migration scripts.
- `docs`: Architecture, contracts, progress, and handoff docs.

## Quick start
1. Create environment: `python -m venv venv && source venv/bin/activate`
2. Install deps: `pip install -e .`
3. Set Postgres connection string:
   - `export DATABASE_URL='postgresql://postgres:postgres@localhost:5432/sparkit'`
4. Run DB migrations:
   - `make db-upgrade`
5. Run API gateway:
   - `uvicorn services.api_gateway.app.main:app --reload --port 8000`

## Provider API keys
Set these in your environment before enabling model routing/ensemble calls:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `KIMI_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## Workflow commands
- `make db-upgrade`
- `make test-compile`
- `make test`
- `make eval-sample`
- `make eval-benchmark`
- `make benchmark-generate`
- `make eval-benchmark-full`
- `make baseline-capture`
- `make baseline-capture-official`
- `make drift-check-sample`

## Notes
- API gateway run state is persisted in Postgres (`runs` table), managed by Alembic migrations.
- Evidence tables (`documents`, `passages`, `claims`, `claim_evidence_links`) are migration-managed.
- Calibration features are persisted in `run_calibration_features`.
- Observability telemetry is persisted in `run_observability_metrics`.
- Budget-aware guards can stop expensive stages early based on `constraints.max_cost_usd` and `constraints.max_latency_s`.
