"""create runs table

Revision ID: 20260219_0001
Revises: 
Create Date: 2026-02-19 00:00:00
"""

from __future__ import annotations

from alembic import op

revision = "20260219_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            constraints_json TEXT NOT NULL,
            answer_style TEXT NOT NULL,
            providers_json TEXT NOT NULL,
            include_trace BOOLEAN NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            progress_json TEXT NOT NULL,
            usage_json TEXT NOT NULL,
            answer_json TEXT,
            citations_json TEXT,
            trace_json TEXT NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_runs_status")
    op.execute("DROP TABLE IF EXISTS runs")
