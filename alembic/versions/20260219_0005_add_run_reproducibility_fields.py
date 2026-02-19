"""add run reproducibility fields

Revision ID: 20260219_0005
Revises: 20260219_0004
Create Date: 2026-02-19 01:45:00
"""

from __future__ import annotations

from alembic import op

revision = "20260219_0005"
down_revision = "20260219_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS prompt_version TEXT")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS config_version TEXT")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS reproducibility_json TEXT")

    op.execute("UPDATE runs SET prompt_version = COALESCE(prompt_version, 'synthesis_v1.1')")
    op.execute("UPDATE runs SET config_version = COALESCE(config_version, 'orchestration_v1.2')")
    op.execute("UPDATE runs SET reproducibility_json = COALESCE(reproducibility_json, '{}')")

    op.execute("ALTER TABLE runs ALTER COLUMN prompt_version SET NOT NULL")
    op.execute("ALTER TABLE runs ALTER COLUMN config_version SET NOT NULL")
    op.execute("ALTER TABLE runs ALTER COLUMN reproducibility_json SET NOT NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS reproducibility_json")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS config_version")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS prompt_version")
