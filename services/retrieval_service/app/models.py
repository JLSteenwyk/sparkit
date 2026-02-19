from __future__ import annotations

from pydantic import BaseModel, Field


class LiteratureRecord(BaseModel):
    source: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str
