from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.request import Request, urlopen

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency fallback
    BeautifulSoup = None

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency fallback
    httpx = None


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._chunks.append(stripped)

    def text(self) -> str:
        return " ".join(self._chunks)


def extract_article_text(html: str) -> str:
    if BeautifulSoup is None:
        article_match = re.search(r"<article[^>]*>(.*?)</article>", html, flags=re.IGNORECASE | re.DOTALL)
        if article_match:
            parser = _TextCollector()
            parser.feed(article_match.group(1))
            article_text = parser.text()
            if article_text:
                return article_text

        paragraph_matches = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
        if paragraph_matches:
            parser = _TextCollector()
            for chunk in paragraph_matches:
                parser.feed(chunk)
            paragraph_text = parser.text()
            if paragraph_text:
                return paragraph_text

        parser = _TextCollector()
        parser.feed(html)
        return parser.text()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    article = soup.find("article")
    if article is not None:
        text = article.get_text(" ", strip=True)
        if text:
            return text

    paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in soup.find_all("p")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if paragraphs:
        return " ".join(paragraphs)

    return soup.get_text(" ", strip=True)


def fetch_article_text(url: str, timeout_seconds: float = 10.0) -> str:
    if httpx is not None:
        response = httpx.get(url, timeout=timeout_seconds, follow_redirects=True)
        response.raise_for_status()
        return extract_article_text(response.text)

    request = Request(url, headers={"User-Agent": "TruthLens/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return extract_article_text(response.read().decode("utf-8", errors="ignore"))
