from app.content_extraction import extract_article_text


def test_extract_article_text_prefers_paragraphs() -> None:
    html = """
    <html>
      <head><title>Sample</title></head>
      <body>
        <article><p>First paragraph.</p><p>Second paragraph.</p></article>
      </body>
    </html>
    """

    assert extract_article_text(html) == "First paragraph. Second paragraph."
