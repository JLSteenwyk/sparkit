from __future__ import annotations

from services.ingestion_service.app.parser import _parse_html


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
