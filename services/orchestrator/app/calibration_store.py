from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import psycopg


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class CalibrationStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/sparkit"
        )

    def _conn(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    def upsert_features(self, run_id: str, features: dict[str, float | int], answer_confidence: float) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO run_calibration_features (
                        run_id, features_json, answer_confidence, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (run_id)
                    DO UPDATE SET
                        features_json = EXCLUDED.features_json,
                        answer_confidence = EXCLUDED.answer_confidence,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (run_id, json.dumps(features), answer_confidence, _now_utc(), _now_utc()),
                )
            conn.commit()
