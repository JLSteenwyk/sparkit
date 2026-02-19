from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    heading: str
    text: str


class IngestionRequest(BaseModel):
    url: str = Field(min_length=1)
    max_chars: int = Field(default=30000, ge=1000, le=200000)


class IngestionResponse(BaseModel):
    url: str
    content_type: str
    title: str | None = None
    sections: list[Section]
    char_count: int
