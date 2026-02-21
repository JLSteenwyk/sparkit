from __future__ import annotations

from services.ingestion_service.app.parser import _clean_text, _parse_html
from services.ingestion_service.app import parser


def test_parse_html_extracts_title_and_body() -> None:
    html = b"""
    <html>
      <head><title>Test Paper</title></head>
      <body>
        <article>
          <p>First paragraph.</p>
          <p>Second paragraph.</p>
        </article>
      </body>
    </html>
    """

    result = _parse_html(html, url="https://example.org/paper", max_chars=10000)

    assert result.content_type == "html"
    assert result.title == "Test Paper"
    assert len(result.sections) == 1
    assert "First paragraph." in result.sections[0].text
    assert "Second paragraph." in result.sections[0].text


def test_clean_text_removes_nul_bytes() -> None:
    assert _clean_text("alpha\x00beta") == "alpha beta"


def test_parse_arxiv_abs_prefers_abstract_block() -> None:
    html = b"""
    <html>
      <head><title>arXiv page</title></head>
      <body>
        <h1 class="title mathjax">Title: Radical OLED photophysics</h1>
        <blockquote class="abstract mathjax">Abstract: We show radical OLED limits are driven by non-radiative losses.</blockquote>
        <p>Help | Advanced Search arXivLabs framework message.</p>
      </body>
    </html>
    """

    result = _parse_html(html, url="https://arxiv.org/abs/2501.01234", max_chars=10000)
    assert result.title == "Radical OLED photophysics"
    assert result.sections[0].heading == "abstract"
    assert "non-radiative losses" in result.sections[0].text
    assert "arXivLabs" not in result.sections[0].text


def test_fetch_and_parse_blocks_hle_related_domains() -> None:
    try:
        parser.fetch_and_parse("https://huggingface.co/datasets/futurehouse/hle-gold-bio-chem")
    except ValueError as exc:
        assert "ingestion blocked" in str(exc)
    else:
        raise AssertionError("expected ValueError for blocked domain")
