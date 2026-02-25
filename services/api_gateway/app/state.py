from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from shared.schemas.domain import (
    Answer,
    Citation,
    Progress,
    QualityGates,
    Run,
    RunTraceResponse,
    Status,
    TraceStage,
    Usage,
)

from services.orchestrator.app.engine import execute_orchestration


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RunExecutor:
    def __init__(self, store: "RunStore") -> None:
        self.store = store

    def execute_run_if_needed(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        state = self.store.get_state(run_id)
        if run is None or state is None:
            return

        if run.status in {Status.COMPLETED, Status.FAILED, Status.CANCELLED, Status.CANCELLING}:
            return

        started = _now_utc()
        state["progress"] = Progress(stage="retrieval_round", percent=30)
        running_state = self.store._serialize_state(state)  # noqa: SLF001 - store internal state serializer
        self.store._update_run_fields(  # noqa: SLF001 - delegated update
            run_id,
            {
                "status": Status.RUNNING.value,
                "updated_at": started,
                "progress_json": running_state["progress_json"],
                "usage_json": running_state["usage_json"],
                "answer_json": running_state["answer_json"],
                "citations_json": running_state["citations_json"],
                "trace_json": running_state["trace_json"],
            },
        )

        try:
            result = execute_orchestration(
                run_id=run.run_id,
                question=run.question,
                min_sources=run.constraints.min_sources,
                providers=run.providers,
                mode=run.mode.value,
                max_latency_s=run.constraints.max_latency_s,
                max_cost_usd=run.constraints.max_cost_usd,
                synthesis_max_tokens=run.constraints.synthesis_max_tokens,
                prompt_version=run.prompt_version,
                config_version=run.config_version,
                reproducibility=run.reproducibility,
            )
        except Exception as exc:  # noqa: BLE001
            ended = _now_utc()
            fail_state = {
                "progress": Progress(stage="failed", percent=100),
                "usage": Usage(cost_usd=0.0, latency_s=max(0.0, (ended - started).total_seconds())),
                "answer": None,
                "citations": None,
                "trace": RunTraceResponse(
                    run_id=run_id,
                    stages=[
                        TraceStage(
                            name="orchestration",
                            status=Status.FAILED,
                            started_at=started,
                            ended_at=ended,
                            artifacts={"error": str(exc)},
                        )
                    ],
                    provider_usage=[],
                    quality_gates=QualityGates(citation_coverage=0.0, unsupported_claims=1),
                ),
            }
            payload = self.store._serialize_state(fail_state)  # noqa: SLF001 - delegated serializer
            self.store._update_run_fields(  # noqa: SLF001 - delegated update
                run_id,
                {
                    "status": Status.FAILED.value,
                    "updated_at": ended,
                    "progress_json": payload["progress_json"],
                    "usage_json": payload["usage_json"],
                    "answer_json": payload["answer_json"],
                    "citations_json": payload["citations_json"],
                    "trace_json": payload["trace_json"],
                },
            )
            return

        ended = _now_utc()
        done_state = {
            "progress": Progress(stage="completed", percent=100),
            "usage": Usage(
                cost_usd=sum(item.cost_usd for item in result.provider_usage),
                latency_s=max(0.0, (ended - started).total_seconds()),
                tokens_input=sum(item.tokens_input for item in result.provider_usage),
                tokens_output=sum(item.tokens_output for item in result.provider_usage),
            ),
            "answer": result.answer,
            "citations": result.citations,
            "trace": RunTraceResponse(
                run_id=run_id,
                stages=result.stages,
                provider_usage=result.provider_usage,
                quality_gates=result.quality_gates,
            ),
        }
        payload = self.store._serialize_state(done_state)  # noqa: SLF001 - delegated serializer
        self.store._update_run_fields(  # noqa: SLF001 - delegated update
            run_id,
            {
                "status": Status.COMPLETED.value,
                "updated_at": ended,
                "progress_json": payload["progress_json"],
                "usage_json": payload["usage_json"],
                "answer_json": payload["answer_json"],
                "citations_json": payload["citations_json"],
                "trace_json": payload["trace_json"],
            },
        )


class RunStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/sparkit"
        )
        self._initialized = False
        self._executor = RunExecutor(self)

    def _conn(self) -> psycopg.Connection:
        connect_timeout_s = int(os.getenv("SPARKIT_DB_CONNECT_TIMEOUT_S", "3") or "3")
        return psycopg.connect(
            self.database_url,
            row_factory=dict_row,
            connect_timeout=max(1, connect_timeout_s),
        )

    def _initialize(self) -> None:
        if self._initialized:
            return

        # Schema is managed by Alembic migrations; this only verifies connectivity.
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        self._initialized = True

    def _default_state(self, run_id: str) -> dict[str, Any]:
        now = _now_utc()
        return {
            "progress": Progress(stage="queued", percent=0),
            "usage": Usage(),
            "answer": None,
            "citations": None,
            "trace": RunTraceResponse(
                run_id=run_id,
                stages=[
                    TraceStage(
                        name="plan",
                        status=Status.QUEUED,
                        started_at=now,
                        artifacts={},
                    )
                ],
                provider_usage=[],
                quality_gates=QualityGates(),
            ),
        }

    def _serialize_state(self, state: dict[str, Any]) -> dict[str, str | None]:
        progress = state["progress"]
        usage = state["usage"]
        answer = state["answer"]
        citations = state["citations"]
        trace = state["trace"]
        return {
            "progress_json": json.dumps(progress.model_dump(mode="json")),
            "usage_json": json.dumps(usage.model_dump(mode="json")),
            "answer_json": json.dumps(answer.model_dump(mode="json")) if answer else None,
            "citations_json": json.dumps([c.model_dump(mode="json") for c in citations])
            if citations
            else None,
            "trace_json": json.dumps(trace.model_dump(mode="json")),
        }

    def _deserialize_run(self, row: dict[str, Any]) -> Run:
        return Run.model_validate(
            {
                "run_id": row["run_id"],
                "question": row["question"],
                "mode": row["mode"],
                "status": row["status"],
                "constraints": json.loads(row["constraints_json"]),
                "answer_style": row["answer_style"],
                "providers": json.loads(row["providers_json"]),
                "include_trace": row["include_trace"],
                "prompt_version": row.get("prompt_version") or "synthesis_v1.2",
                "config_version": row.get("config_version") or "orchestration_v1.2",
                "reproducibility": json.loads(row.get("reproducibility_json") or "{}"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    def _deserialize_state(self, row: dict[str, Any]) -> dict[str, Any]:
        answer_json = row["answer_json"]
        citations_json = row["citations_json"]
        return {
            "progress": Progress.model_validate(json.loads(row["progress_json"])),
            "usage": Usage.model_validate(json.loads(row["usage_json"])),
            "answer": None if answer_json is None else Answer.model_validate(json.loads(answer_json)),
            "citations": None
            if citations_json is None
            else [Citation.model_validate(item) for item in json.loads(citations_json)],
            "trace": RunTraceResponse.model_validate(json.loads(row["trace_json"])),
        }

    def create_run(self, run: Run) -> str:
        self._initialize()
        state = self._default_state(run.run_id)
        state_payload = self._serialize_state(state)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runs (
                        run_id, question, mode, status, constraints_json, answer_style,
                        providers_json, include_trace, prompt_version, config_version, reproducibility_json,
                        created_at, updated_at, progress_json, usage_json, answer_json, citations_json, trace_json
                    ) VALUES (
                        %(run_id)s, %(question)s, %(mode)s, %(status)s, %(constraints_json)s,
                        %(answer_style)s, %(providers_json)s, %(include_trace)s, %(prompt_version)s,
                        %(config_version)s, %(reproducibility_json)s, %(created_at)s, %(updated_at)s,
                        %(progress_json)s, %(usage_json)s, %(answer_json)s,
                        %(citations_json)s, %(trace_json)s
                    )
                    """,
                    {
                        "run_id": run.run_id,
                        "question": run.question,
                        "mode": run.mode.value,
                        "status": run.status.value,
                        "constraints_json": json.dumps(run.constraints.model_dump(mode="json")),
                        "answer_style": run.answer_style,
                        "providers_json": json.dumps(run.providers),
                        "include_trace": run.include_trace,
                        "prompt_version": run.prompt_version,
                        "config_version": run.config_version,
                        "reproducibility_json": json.dumps(run.reproducibility),
                        "created_at": run.created_at,
                        "updated_at": run.updated_at,
                        **state_payload,
                    },
                )
            conn.commit()
        return run.run_id

    def _get_row(self, run_id: str) -> dict[str, Any] | None:
        self._initialize()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,))
                return cur.fetchone()

    def get_run(self, run_id: str) -> Run | None:
        row = self._get_row(run_id)
        if row is None:
            return None
        return self._deserialize_run(row)

    def get_state(self, run_id: str) -> dict[str, Any] | None:
        row = self._get_row(run_id)
        if row is None:
            return None
        return self._deserialize_state(row)

    def cancel_run(self, run_id: str) -> str:
        self._initialize()
        now = _now_utc()
        run = self.get_run(run_id)
        if run is None:
            return "not_found"
        if run.status in {Status.COMPLETED, Status.FAILED, Status.CANCELLED}:
            return "terminal"
        if run.status == Status.CANCELLING:
            return "already_cancelling"

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE runs
                    SET status = %s, updated_at = %s
                    WHERE run_id = %s
                    """,
                    (Status.CANCELLING.value, now, run_id),
                )
                updated = cur.rowcount
            conn.commit()
        return "cancelled" if updated > 0 else "not_found"

    def _update_run_fields(self, run_id: str, fields: dict[str, Any]) -> None:
        if not fields:
            return
        self._initialize()
        assignments = ", ".join([f"{key} = %({key})s" for key in fields])
        query = f"UPDATE runs SET {assignments} WHERE run_id = %(run_id)s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {**fields, "run_id": run_id})
            conn.commit()

    def execute_run_if_needed(self, run_id: str) -> None:
        self._executor.execute_run_if_needed(run_id)


def make_run_id() -> str:
    return f"run_{uuid4().hex}"


run_store = RunStore()
