"""WeChat Official Account article HTML parser.

Parses public mp.weixin.qq.com article pages — zero anti-scraping risk.
Uses stdlib html.parser + re: no BeautifulSoup dependency needed since
WeChat article HTML structure is consistent.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for structured extraction
# ---------------------------------------------------------------------------

# Title: rich_media_title h1 or og:title meta
_RE_OG_TITLE = re.compile(
    r'<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
_RE_H1_TITLE = re.compile(
    r'<h1[^>]*class=["\'][^"\']*rich_media_title[^"\']*["\'][^>]*>(.*?)</h1>',
    re.IGNORECASE | re.DOTALL,
)

# Author: js_name anchor or rich_media_meta_text span
_RE_JS_NAME = re.compile(
    r'<a[^>]*id=["\']js_name["\'][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_RE_META_TEXT = re.compile(
    r'<span[^>]*class=["\'][^"\']*rich_media_meta_text[^"\']*["\'][^>]*>(.*?)</span>',
    re.IGNORECASE | re.DOTALL,
)

# Publish time: em#publish_time or JS var ct = "timestamp"
_RE_PUBLISH_TIME_EM = re.compile(
    r'<em[^>]*id=["\']publish_time["\'][^>]*>(.*?)</em>',
    re.IGNORECASE | re.DOTALL,
)
_RE_CT_VAR = re.compile(r'var\s+ct\s*=\s*["\'](\d+)["\']')

# Content body: div#js_content
_RE_JS_CONTENT = re.compile(
    r'<div[^>]*id=["\']js_content["\'][^>]*>(.*?)</div\s*>(?=\s*<div|\s*</section|\s*</body|\Z)',
    re.IGNORECASE | re.DOTALL,
)

# Images: data-src attribute (WeChat lazy-loads images)
_RE_DATA_SRC = re.compile(r'data-src=["\'](https?://[^"\']+)["\']', re.IGNORECASE)

# Audio: mpvoice tags
_RE_MPVOICE = re.compile(r'<mpvoice[^>]*voice_encode_fileid=["\']([^"\']+)["\'][^>]*/?>',
                          re.IGNORECASE)


# ---------------------------------------------------------------------------
# HTML tag stripper
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    """Lightweight tag stripper — used to extract text from HTML fragments."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self._parts).strip()


def _strip_tags(html_fragment: str) -> str:
    """Return plain text from an HTML fragment."""
    stripper = _HTMLStripper()
    stripper.feed(html_fragment)
    return stripper.text


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_title(html: str) -> str:
    """Extract article title from HTML."""
    # Try h1.rich_media_title first (most reliable)
    m = _RE_H1_TITLE.search(html)
    if m:
        return _strip_tags(m.group(1))
    # Fallback: og:title meta
    m = _RE_OG_TITLE.search(html)
    if m:
        raw = m.group(1)
        # HTML-decode common entities
        return raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()
    return ""


def _extract_author(html: str) -> str:
    """Extract official account name (公众号名称) from HTML."""
    # Try a#js_name first
    m = _RE_JS_NAME.search(html)
    if m:
        return _strip_tags(m.group(1))
    # Fallback: first span.rich_media_meta_text
    m = _RE_META_TEXT.search(html)
    if m:
        return _strip_tags(m.group(1))
    return ""


def _extract_publish_time(html: str) -> str:
    """Extract article publish timestamp as string.

    Tries em#publish_time text first, then JavaScript ``var ct`` unix timestamp.
    """
    # em#publish_time (usually human-readable date)
    m = _RE_PUBLISH_TIME_EM.search(html)
    if m:
        text = _strip_tags(m.group(1))
        if text:
            return text
    # JS var ct = "unix_timestamp"
    m = _RE_CT_VAR.search(html)
    if m:
        return m.group(1)
    return ""


def _extract_body(html: str) -> str:
    """Extract the article body HTML (div#js_content contents)."""
    m = _RE_JS_CONTENT.search(html)
    if m:
        return m.group(1).strip()
    return ""


def _extract_images(html: str) -> list[str]:
    """Extract image URLs from data-src attributes (WeChat lazy-loads images)."""
    seen: set[str] = set()
    result: list[str] = []
    for url in _RE_DATA_SRC.findall(html):
        clean = url.split("?")[0] if "?" in url else url
        # Keep original URL (with query params may be needed for auth)
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _extract_audio(html: str) -> list[str]:
    """Extract audio file IDs from mpvoice tags.

    Returns a list of voice_encode_fileid values. These are internal IDs
    and cannot be directly downloaded without WeChat API access; they are
    recorded in metadata for reference.
    """
    return _RE_MPVOICE.findall(html)


# ---------------------------------------------------------------------------
# Public parser class
# ---------------------------------------------------------------------------


class WeChatOAParser:
    """Fetches and parses a public WeChat Official Account article.

    Uses httpx for async HTTP — same dependency as the rest of the project.
    No authentication required; articles on mp.weixin.qq.com are public.

    Args:
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def fetch_article(self, url: str) -> dict:
        """GET the article page and return structured data.

        Args:
            url: Full mp.weixin.qq.com article URL.

        Returns:
            Dict with keys:
            - title (str)
            - author (str)
            - publish_time (str)
            - content_html (str) — raw HTML of body
            - images (list[str]) — image URLs from data-src
            - audio_urls (list[str]) — voice_encode_fileid values
            - source_url (str)
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text

        return self._parse(html, url)

    def parse_html(self, html: str, url: str = "") -> dict:
        """Parse article HTML directly (for testing without HTTP).

        Args:
            html: Raw HTML string of the article page.
            url: Original URL (recorded in source_url field).

        Returns:
            Same dict structure as :meth:`fetch_article`.
        """
        return self._parse(html, url)

    def _parse(self, html: str, url: str) -> dict:
        title = _extract_title(html)
        author = _extract_author(html)
        publish_time = _extract_publish_time(html)
        content_html = _extract_body(html)
        images = _extract_images(html)
        audio_urls = _extract_audio(html)

        logger.debug(
            "Parsed WeChatOA article: title=%r, author=%r, images=%d, audio=%d",
            title,
            author,
            len(images),
            len(audio_urls),
        )

        return {
            "title": title,
            "author": author,
            "publish_time": publish_time,
            "content_html": content_html,
            "images": images,
            "audio_urls": audio_urls,
            "source_url": url,
        }
