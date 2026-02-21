from __future__ import annotations

import io
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import IngestionResponse, Section

_DISALLOWED_INGEST_DOMAINS = {
    "huggingface.co",
    "futurehouse.org",
}


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_html(content: bytes, url: str, max_chars: int) -> IngestionResponse:
    soup = BeautifulSoup(content, "html.parser")
    title = _clean_text(soup.title.get_text()) if soup.title and soup.title.get_text() else None

    parsed = urlparse(url)
    if "arxiv.org" in parsed.netloc and "/abs/" in parsed.path:
        abstract_block = soup.find("blockquote", class_=lambda value: value and "abstract" in value.lower())
        if abstract_block is not None:
            abstract_text = _clean_text(abstract_block.get_text(" ", strip=True))
            abstract_text = re.sub(r"^\s*abstract:\s*", "", abstract_text, flags=re.IGNORECASE)
            if abstract_text:
                arxiv_title = soup.find("h1", class_=lambda value: value and "title" in value.lower())
                if arxiv_title is not None:
                    parsed_title = _clean_text(arxiv_title.get_text(" ", strip=True))
                    parsed_title = re.sub(r"^\s*title:\s*", "", parsed_title, flags=re.IGNORECASE)
                    if parsed_title:
                        title = parsed_title
                trimmed = abstract_text[:max_chars]
                return IngestionResponse(
                    url=url,
                    content_type="html",
                    title=title,
                    sections=[Section(heading="abstract", text=trimmed)],
                    char_count=len(trimmed),
                )

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
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if any(host == dom or host.endswith(f".{dom}") for dom in _DISALLOWED_INGEST_DOMAINS):
        raise ValueError(f"ingestion blocked for disallowed domain: {host}")

    headers = {"User-Agent": "SPARKIT/0.1"}
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()

    content_type = (response.headers.get("content-type") or "").lower()
    content = response.content

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return _parse_pdf(content, url=url, max_chars=max_chars)
    return _parse_html(content, url=url, max_chars=max_chars)
