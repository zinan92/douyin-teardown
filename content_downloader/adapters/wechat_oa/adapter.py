"""WeChat Official Account (公众号) platform adapter.

Implements the PlatformAdapter protocol for downloading public WeChat OA articles.
Articles on mp.weixin.qq.com are public HTML — no authentication required.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

import httpx

from content_downloader.adapters.wechat_oa.parser import WeChatOAParser
from content_downloader.models import ContentItem, DownloadError, DownloadResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL patterns
# ---------------------------------------------------------------------------

_ARTICLE_URL_RE = re.compile(r"https?://mp\.weixin\.qq\.com/s/\S+")

_PROFILE_UNSUPPORTED_MESSAGE = (
    "WeChat Official Account profile batch download is not supported.\n"
    "WeChat does not expose a public article list page for OA profiles.\n"
    "Workaround: Copy individual article URLs and download them one at a time:\n"
    "  content-downloader download \"https://mp.weixin.qq.com/s/xxxx\""
)


# ---------------------------------------------------------------------------
# ID extraction
# ---------------------------------------------------------------------------


def _extract_article_id(url: str) -> str:
    """Extract a stable content ID from a WeChat article URL.

    WeChat article URLs follow two main patterns:
    - https://mp.weixin.qq.com/s/{base64_id}  (short form)
    - https://mp.weixin.qq.com/s?__biz=...&mid=...&idx=...  (long form)

    For the short form, the path suffix is used as the ID.
    For the long form, mid + idx is used.
    Falls back to the full URL path if no recognisable pattern is found.
    """
    parsed = urlparse(url)
    path = parsed.path  # e.g. /s/abc123

    # Short form: /s/{id}
    short_match = re.match(r"^/s/([^/?#]+)$", path)
    if short_match:
        return short_match.group(1)

    # Long form: query params __biz, mid, idx
    qs = parse_qs(parsed.query)
    mid = (qs.get("mid") or [""])[0]
    idx = (qs.get("idx") or [""])[0]
    if mid:
        return f"{mid}_{idx}" if idx else mid

    # Fallback: sanitise path
    return re.sub(r"[^A-Za-z0-9_-]", "_", path.strip("/")) or "unknown"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class WeChatOAAdapter:
    """Adapter for downloading WeChat Official Account articles.

    Downloads the public HTML article page, extracts structured metadata
    (title, author, publish time), downloads all inline images, records
    any audio file IDs, and writes the standard output directory.

    Args:
        parser: Optional :class:`WeChatOAParser` instance (injected for testing).
        image_timeout: HTTP timeout for individual image downloads.
    """

    platform = "wechat_oa"

    def __init__(
        self,
        parser: Optional[WeChatOAParser] = None,
        image_timeout: float = 20.0,
    ) -> None:
        self._parser = parser or WeChatOAParser()
        self._image_timeout = image_timeout

    # ------------------------------------------------------------------
    # PlatformAdapter protocol
    # ------------------------------------------------------------------

    def can_handle(self, url: str) -> bool:
        """Return True if the URL is a WeChat OA article URL."""
        return bool(_ARTICLE_URL_RE.match(url))

    async def download_single(
        self,
        url: str,
        output_dir: Path,
    ) -> ContentItem:
        """Download a single WeChat OA article.

        Steps:
        1. Fetch and parse article HTML.
        2. Create output directory structure.
        3. Save article HTML to ``media/article.html``.
        4. Download inline images to ``media/img_01.jpg``, etc.
        5. Record audio file IDs (not directly downloadable) in metadata.
        6. Write ``metadata.json`` (raw parsed data).
        7. Write ``content_item.json``.
        8. Return populated :class:`~content_downloader.models.ContentItem`.
        """
        article_id = _extract_article_id(url)
        parsed = await self._parser.fetch_article(url)

        author = parsed["author"] or "unknown"
        content_dir = output_dir / "wechat_oa" / author / article_id
        media_dir = content_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        media_files: list[str] = []

        # Save article HTML
        article_html_path = media_dir / "article.html"
        article_html_path.write_text(parsed["content_html"], encoding="utf-8")
        media_files.append("media/article.html")

        # Download images
        img_files = await self._download_images(parsed["images"], media_dir)
        media_files.extend(img_files)

        # Record audio IDs (cannot be directly downloaded without WeChat API)
        audio_files: list[str] = []
        if parsed["audio_urls"]:
            audio_index_path = media_dir / "audio_ids.txt"
            audio_index_path.write_text(
                "\n".join(parsed["audio_urls"]),
                encoding="utf-8",
            )
            audio_files.append("media/audio_ids.txt")
            media_files.extend(audio_files)

        # Write metadata.json (raw parsed result)
        metadata: dict = {
            "title": parsed["title"],
            "author": parsed["author"],
            "publish_time": parsed["publish_time"],
            "source_url": url,
            "images": parsed["images"],
            "audio_urls": parsed["audio_urls"],
        }
        metadata_path = content_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Build ContentItem
        cover_file: Optional[str] = img_files[0] if img_files else None
        description = _strip_html_tags(parsed["content_html"])[:200]

        item = ContentItem(
            platform="wechat_oa",
            content_id=article_id,
            content_type="article",
            title=parsed["title"],
            description=description,
            author_id=author,
            author_name=parsed["author"],
            publish_time=parsed["publish_time"],
            source_url=url,
            media_files=media_files,
            cover_file=cover_file,
            metadata_file="metadata.json",
            likes=0,
            comments=0,
            shares=0,
            collects=0,
            views=0,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )

        # Write content_item.json
        (content_dir / "content_item.json").write_text(
            item.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # Write text.txt — article title for quick reference
        # (full article text is in media/article.html)
        if item.title:
            (content_dir / "text.txt").write_text(
                item.title, encoding="utf-8",
            )

        return item

    async def download_profile(
        self,
        profile_url: str,
        output_dir: Path,
        limit: int = 0,
        since: Optional[datetime] = None,
    ) -> DownloadResult:
        """Batch-download articles from a WeChat OA profile.

        Not supported — WeChat does not expose a public article list API.
        Returns an ``unsupported`` error with workaround instructions.
        """
        logger.warning(
            "WeChatOA profile download not supported: %s", profile_url
        )
        error = DownloadError(
            content_id="",
            source_url=profile_url,
            error_type="unsupported",
            message=_PROFILE_UNSUPPORTED_MESSAGE,
            retryable=False,
        )
        return DownloadResult(
            items=[],
            errors=[error],
            total=0,
            success=0,
            failed=1,
            skipped=0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _download_images(
        self,
        image_urls: list[str],
        media_dir: Path,
    ) -> list[str]:
        """Download images and return relative file paths.

        Images are named ``img_01.jpg``, ``img_02.jpg``, etc.
        Failed downloads are logged and skipped (non-fatal).
        """
        relative_paths: list[str] = []
        async with httpx.AsyncClient(timeout=self._image_timeout, follow_redirects=True) as client:
            for idx, img_url in enumerate(image_urls):
                filename = f"img_{idx + 1:02d}.jpg"
                dest = media_dir / filename
                try:
                    response = await client.get(img_url)
                    response.raise_for_status()
                    dest.write_bytes(response.content)
                    relative_paths.append(f"media/{filename}")
                    logger.debug("Downloaded image %d: %s -> %s", idx + 1, img_url, filename)
                except Exception as exc:
                    logger.warning(
                        "Failed to download image %d (%s): %s", idx + 1, img_url, exc
                    )
        return relative_paths


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _strip_html_tags(html_fragment: str) -> str:
    """Remove HTML tags from a fragment for use as description text."""
    return re.sub(r"<[^>]+>", "", html_fragment)
