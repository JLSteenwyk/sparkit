from __future__ import annotations

import io
import re

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import IngestionResponse, Section


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_html(content: bytes, url: str, max_chars: int) -> IngestionResponse:
    soup = BeautifulSoup(content, "html.parser")
    title = _clean_text(soup.title.get_text()) if soup.title and soup.title.get_text() else None

    sections: list[Section] = []
    article = soup.find("article")
    if article is not None:
        paragraphs = [_clean_text(p.get_text(" ", strip=True)) for p in article.find_all("p")]
    else:
        paragraphs = [_clean_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]

    paragraphs = [p for p in paragraphs if p]
    joined = "\n\n".join(paragraphs)
    trimmed = joined[:max_chars]
    if trimmed:
        sections.append(Section(heading="body", text=trimmed))

    return IngestionResponse(
        url=url,
        content_type="html",
        title=title,
        sections=sections,
        char_count=len(trimmed),
    )


def _parse_pdf(content: bytes, url: str, max_chars: int) -> IngestionResponse:
    reader = PdfReader(io.BytesIO(content))
    page_chunks: list[str] = []
    for idx, page in enumerate(reader.pages):
        text = _clean_text(page.extract_text() or "")
        if not text:
            continue
        page_chunks.append(f"[Page {idx + 1}] {text}")

    joined = "\n\n".join(page_chunks)
    trimmed = joined[:max_chars]
    sections = [Section(heading="pdf_text", text=trimmed)] if trimmed else []

    return IngestionResponse(
        url=url,
        content_type="pdf",
        title=None,
        sections=sections,
        char_count=len(trimmed),
    )


def fetch_and_parse(url: str, max_chars: int = 30000, timeout_s: float = 20.0) -> IngestionResponse:
    headers = {"User-Agent": "SPARKIT/0.1"}
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()

    content_type = (response.headers.get("content-type") or "").lower()
    content = response.content

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return _parse_pdf(content, url=url, max_chars=max_chars)
    return _parse_html(content, url=url, max_chars=max_chars)
