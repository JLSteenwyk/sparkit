from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from threading import RLock

import psycopg
from psycopg.rows import dict_row

from services.retrieval_service.app.models import LiteratureRecord

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "using",
    "used",
    "study",
    "evidence",
    "question",
    "results",
    "analysis",
    "model",
    "models",
    "answer",
    "choices",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    return [token for token in tokens if token not in _STOPWORDS]


def _chunk_text(text: str, max_chars: int = 1500, stride_chars: int = 1000) -> list[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start += stride_chars
    return chunks


def _doc_key(record: LiteratureRecord) -> str:
    key = record.doi or record.url
    return f"cdoc_{sha1(key.encode('utf-8')).hexdigest()}"  # noqa: S324


@dataclass(frozen=True)
class CorpusSearchHit:
    doc_id: str
    title: str
    abstract: str
    source: str
    year: int | None
    doi: str | None
    url: str
    score: float


_MEM_LOCK = RLock()
_MEM_DOCS: dict[str, dict[str, object]] = {}
_MEM_CHUNKS: list[dict[str, object]] = []


class LocalCorpusStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/sparkit"
        )
        self._ready = False
        self._memory_fallback = _env_bool("SPARKIT_ENABLE_MEMORY_FALLBACK", True)
        self._use_memory = False

    def _conn(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def ensure_schema(self) -> None:
        if self._ready:
            return
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS corpus_documents (
                            corpus_doc_id TEXT PRIMARY KEY,
                            source TEXT NOT NULL,
                            title TEXT NOT NULL,
                            abstract TEXT NULL,
                            authors_json TEXT NOT NULL,
                            year INTEGER NULL,
                            doi TEXT NULL,
                            url TEXT NOT NULL,
                            domain TEXT NULL,
                            subdomain TEXT NULL,
                            created_at TIMESTAMPTZ NOT NULL,
                            updated_at TIMESTAMPTZ NOT NULL
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS corpus_chunks (
                            chunk_id TEXT PRIMARY KEY,
                            corpus_doc_id TEXT NOT NULL REFERENCES corpus_documents(corpus_doc_id) ON DELETE CASCADE,
                            chunk_index INTEGER NOT NULL,
                            text TEXT NOT NULL,
                            token_count INTEGER NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL
                        )
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_corpus_documents_domain ON corpus_documents(domain)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_corpus_documents_updated_at ON corpus_documents(updated_at DESC)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_corpus_chunks_doc ON corpus_chunks(corpus_doc_id)"
                    )
                conn.commit()
        except psycopg.OperationalError:
            if not self._memory_fallback:
                raise
            self._use_memory = True
        self._ready = True

    def upsert_document(
        self,
        record: LiteratureRecord,
        text: str,
        domain: str | None = None,
        subdomain: str | None = None,
    ) -> str:
        self.ensure_schema()
        doc_id = _doc_key(record)
        chunks = _chunk_text(text)
        if not chunks:
            chunks = _chunk_text(record.abstract or "")
        now = _now()
        if self._use_memory:
            with _MEM_LOCK:
                _MEM_DOCS[doc_id] = {
                    "corpus_doc_id": doc_id,
                    "source": record.source,
                    "title": record.title,
                    "abstract": record.abstract or "",
                    "authors_json": json.dumps(record.authors),
                    "year": record.year,
                    "doi": record.doi,
                    "url": record.url,
                    "domain": domain,
                    "subdomain": subdomain,
                    "created_at": now,
                    "updated_at": now,
                }
                global _MEM_CHUNKS
                _MEM_CHUNKS = [row for row in _MEM_CHUNKS if row.get("corpus_doc_id") != doc_id]
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"cchunk_{sha1(f'{doc_id}:{idx}'.encode('utf-8')).hexdigest()}"  # noqa: S324
                    _MEM_CHUNKS.append(
                        {
                            "chunk_id": chunk_id,
                            "corpus_doc_id": doc_id,
                            "chunk_index": idx,
                            "text": chunk,
                            "token_count": len(_tokenize(chunk)),
                            "created_at": now,
                        }
                    )
            return doc_id
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO corpus_documents (
                        corpus_doc_id, source, title, abstract, authors_json, year, doi, url,
                        domain, subdomain, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (corpus_doc_id) DO UPDATE SET
                        source = EXCLUDED.source,
                        title = EXCLUDED.title,
                        abstract = EXCLUDED.abstract,
                        authors_json = EXCLUDED.authors_json,
                        year = EXCLUDED.year,
                        doi = EXCLUDED.doi,
                        url = EXCLUDED.url,
                        domain = EXCLUDED.domain,
                        subdomain = EXCLUDED.subdomain,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        doc_id,
                        record.source,
                        record.title.replace("\x00", " "),
                        (record.abstract or "").replace("\x00", " "),
                        json.dumps([author.replace("\x00", " ") for author in record.authors]),
                        record.year,
                        record.doi,
                        record.url,
                        domain,
                        subdomain,
                        now,
                        now,
                    ),
                )
                cur.execute("DELETE FROM corpus_chunks WHERE corpus_doc_id = %s", (doc_id,))
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"cchunk_{sha1(f'{doc_id}:{idx}'.encode('utf-8')).hexdigest()}"  # noqa: S324
                    cur.execute(
                        """
                        INSERT INTO corpus_chunks (chunk_id, corpus_doc_id, chunk_index, text, token_count, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (chunk_id, doc_id, idx, chunk.replace("\x00", " "), len(_tokenize(chunk)), now),
                    )
            conn.commit()
        return doc_id

    def query(self, query: str, max_results: int = 12, domain: str | None = None) -> list[LiteratureRecord]:
        self.ensure_schema()
        tokens = _tokenize(query)[:8]
        if not tokens:
            return []
        if self._use_memory:
            query_tokens = set(tokens)
            scored: dict[str, CorpusSearchHit] = {}
            with _MEM_LOCK:
                rows = list(_MEM_CHUNKS)
                docs = dict(_MEM_DOCS)
            for chunk in rows:
                doc_id = str(chunk.get("corpus_doc_id"))
                doc = docs.get(doc_id)
                if not doc:
                    continue
                if domain and str(doc.get("domain") or "") != domain:
                    continue
                text = f"{doc.get('title','')} {chunk.get('text','')}"
                hay_tokens = set(_tokenize(text))
                overlap = len(query_tokens & hay_tokens)
                if overlap == 0:
                    continue
                year = doc.get("year")
                recency = (int(year) if isinstance(year, int) else 2000) / 10000.0
                score = (2.0 * overlap) + recency
                existing = scored.get(doc_id)
                if existing and existing.score >= score:
                    continue
                scored[doc_id] = CorpusSearchHit(
                    doc_id=doc_id,
                    title=str(doc.get("title") or ""),
                    abstract=str(chunk.get("text") or str(doc.get("abstract") or ""))[:1600],
                    source="local_corpus",
                    year=year if isinstance(year, int) else None,
                    doi=str(doc.get("doi")) if doc.get("doi") else None,
                    url=str(doc.get("url") or ""),
                    score=score,
                )
            ranked = sorted(scored.values(), key=lambda item: item.score, reverse=True)[:max_results]
            return [
                LiteratureRecord(
                    source=item.source,
                    title=item.title,
                    abstract=item.abstract,
                    authors=[],
                    year=item.year,
                    doi=item.doi,
                    url=item.url,
                )
                for item in ranked
            ]
        like_clauses = " OR ".join([f"lower(c.text) LIKE %s OR lower(d.title) LIKE %s" for _ in tokens])
        domain_clause = " AND d.domain = %s" if domain else ""
        params: list[object] = []
        for token in tokens:
            pattern = f"%{token}%"
            params.extend([pattern, pattern])
        if domain:
            params.append(domain)
        params.append(max(120, max_results * 20))
        sql = f"""
            SELECT
                d.corpus_doc_id,
                d.source,
                d.title,
                d.abstract,
                d.year,
                d.doi,
                d.url,
                c.text AS chunk_text
            FROM corpus_documents d
            JOIN corpus_chunks c ON c.corpus_doc_id = d.corpus_doc_id
            WHERE ({like_clauses}) {domain_clause}
            ORDER BY d.updated_at DESC
            LIMIT %s
        """
        rows: list[dict] = []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        scored: dict[str, CorpusSearchHit] = {}
        query_tokens = set(tokens)
        for row in rows:
            hay_tokens = set(_tokenize(f"{row['title']} {row.get('chunk_text') or ''}"))
            overlap = len(query_tokens & hay_tokens)
            if overlap == 0:
                continue
            recency = (row.get("year") or 2000) / 10000.0
            score = (2.0 * overlap) + recency
            doc_id = row["corpus_doc_id"]
            existing = scored.get(doc_id)
            if existing and existing.score >= score:
                continue
            scored[doc_id] = CorpusSearchHit(
                doc_id=doc_id,
                title=row["title"],
                abstract=(row.get("chunk_text") or row.get("abstract") or "")[:1600],
                source="local_corpus",
                year=row.get("year"),
                doi=row.get("doi"),
                url=row["url"],
                score=score,
            )

        ranked = sorted(scored.values(), key=lambda item: item.score, reverse=True)[:max_results]
        return [
            LiteratureRecord(
                source=item.source,
                title=item.title,
                abstract=item.abstract,
                authors=[],
                year=item.year,
                doi=item.doi,
                url=item.url,
            )
            for item in ranked
        ]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
