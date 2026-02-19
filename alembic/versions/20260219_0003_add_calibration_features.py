"""add calibration features table

Revision ID: 20260219_0003
Revises: 20260219_0002
Create Date: 2026-02-19 00:50:00
"""

from __future__ import annotations

from alembic import op

revision = "20260219_0003"
down_revision = "20260219_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS run_calibration_features (
            run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
            features_json TEXT NOT NULL,
            answer_confidence DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS run_calibration_features")
