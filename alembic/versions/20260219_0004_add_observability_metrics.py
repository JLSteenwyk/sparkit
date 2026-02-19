"""add observability metrics table

Revision ID: 20260219_0004
Revises: 20260219_0003
Create Date: 2026-02-19 01:20:00
"""

from __future__ import annotations

from alembic import op

revision = "20260219_0004"
down_revision = "20260219_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS run_observability_metrics (
            run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
            metrics_json TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS run_observability_metrics")
