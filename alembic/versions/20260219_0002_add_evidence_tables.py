"""add evidence tables

Revision ID: 20260219_0002
Revises: 20260219_0001
Create Date: 2026-02-19 00:30:00
"""

from __future__ import annotations

from alembic import op

revision = "20260219_0002"
down_revision = "20260219_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            abstract TEXT,
            authors_json TEXT NOT NULL,
            year INTEGER,
            doi TEXT,
            url TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents (doi)")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS passages (
            passage_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            section TEXT NOT NULL,
            text TEXT NOT NULL,
            offset_start INTEGER NOT NULL,
            offset_end INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_passages_doc_id ON passages (doc_id)")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            text TEXT NOT NULL,
            claim_type TEXT NOT NULL,
            support_score DOUBLE PRECISION NOT NULL,
            contradiction_score DOUBLE PRECISION NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_claims_run_id ON claims (run_id)")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS claim_evidence_links (
            link_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
            passage_id TEXT NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
            relation TEXT NOT NULL,
            strength DOUBLE PRECISION NOT NULL,
            rationale TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_claim_links_claim_id ON claim_evidence_links (claim_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_claim_links_claim_id")
    op.execute("DROP TABLE IF EXISTS claim_evidence_links")
    op.execute("DROP INDEX IF EXISTS idx_claims_run_id")
    op.execute("DROP TABLE IF EXISTS claims")
    op.execute("DROP INDEX IF EXISTS idx_passages_doc_id")
    op.execute("DROP TABLE IF EXISTS passages")
    op.execute("DROP INDEX IF EXISTS idx_documents_doi")
    op.execute("DROP TABLE IF EXISTS documents")
