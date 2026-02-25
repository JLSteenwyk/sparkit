from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import psycopg

from services.retrieval_service.app.models import LiteratureRecord


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _doc_id(record: LiteratureRecord) -> str:
    key = record.doi or record.url
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()  # noqa: S324
    return f"doc_{digest}"


def _sanitize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace("\x00", " ")


@dataclass(frozen=True)
class StoredEvidence:
    doc_id: str
    passage_id: str


class EvidenceStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/sparkit"
        )
        self._memory_fallback = _env_bool("SPARKIT_ENABLE_MEMORY_FALLBACK", True)
        self._use_memory = False

    def _conn(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    def upsert_document_with_passage(
        self,
        record: LiteratureRecord,
        section: str = "abstract",
        text: str | None = None,
    ) -> StoredEvidence:
        doc_id = _doc_id(record)
        passage_id = f"psg_{uuid4().hex}"
        passage_text = _sanitize_text(text if text is not None else (record.abstract or "")) or ""
        section_clean = _sanitize_text(section) or "abstract"
        title_clean = _sanitize_text(record.title) or record.title
        abstract_clean = _sanitize_text(record.abstract)
        authors_clean = [_sanitize_text(author) or "" for author in record.authors]

        if not self._use_memory:
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO documents (doc_id, source, title, abstract, authors_json, year, doi, url, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (doc_id)
                            DO UPDATE SET
                                source = EXCLUDED.source,
                                title = EXCLUDED.title,
                                abstract = EXCLUDED.abstract,
                                authors_json = EXCLUDED.authors_json,
                                year = EXCLUDED.year,
                                doi = EXCLUDED.doi,
                                url = EXCLUDED.url
                            """,
                            (
                                doc_id,
                                record.source,
                                title_clean,
                                abstract_clean,
                                json.dumps(authors_clean),
                                record.year,
                                record.doi,
                                record.url,
                                _now_utc(),
                            ),
                        )
                        cur.execute(
                            """
                            INSERT INTO passages (passage_id, doc_id, section, text, offset_start, offset_end, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (passage_id, doc_id, section_clean, passage_text, 0, len(passage_text), _now_utc()),
                        )
                    conn.commit()
            except psycopg.OperationalError:
                if not self._memory_fallback:
                    raise
                self._use_memory = True

        return StoredEvidence(doc_id=doc_id, passage_id=passage_id)

    def insert_claim(self, run_id: str, text: str, claim_type: str, support_score: float, status: str) -> str:
        claim_id = f"clm_{uuid4().hex}"
        clean_text = _sanitize_text(text) or ""
        clean_status = _sanitize_text(status) or status
        clean_claim_type = _sanitize_text(claim_type) or claim_type
        if not self._use_memory:
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO claims (
                                claim_id, run_id, text, claim_type, support_score,
                                contradiction_score, status, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (claim_id, run_id, clean_text, clean_claim_type, support_score, 0.0, clean_status, _now_utc()),
                        )
                    conn.commit()
            except psycopg.OperationalError:
                if not self._memory_fallback:
                    raise
                self._use_memory = True
        return claim_id

    def link_claim_to_passage(
        self, claim_id: str, passage_id: str, relation: str = "supports", strength: float = 0.8
    ) -> str:
        link_id = f"lnk_{uuid4().hex}"
        clean_relation = _sanitize_text(relation) or relation
        if not self._use_memory:
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO claim_evidence_links (
                                link_id, claim_id, passage_id, relation, strength, rationale, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                link_id,
                                claim_id,
                                passage_id,
                                clean_relation,
                                strength,
                                _sanitize_text("title/abstract support"),
                                _now_utc(),
                            ),
                        )
                    conn.commit()
            except psycopg.OperationalError:
                if not self._memory_fallback:
                    raise
                self._use_memory = True
        return link_id


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
