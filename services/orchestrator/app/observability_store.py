from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import psycopg


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ObservabilityStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/sparkit"
        )

    def _conn(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    def upsert_metrics(self, run_id: str, metrics: dict) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO run_observability_metrics (
                        run_id, metrics_json, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id)
                    DO UPDATE SET
                        metrics_json = EXCLUDED.metrics_json,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (run_id, json.dumps(metrics), _now_utc(), _now_utc()),
                )
            conn.commit()
