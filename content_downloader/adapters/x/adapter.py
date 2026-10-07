"""X (Twitter) platform adapter — implements PlatformAdapter protocol."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from content_downloader.adapters.x.fetcher import XFetcher, _YTDLP_NOT_INSTALLED_MESSAGE
from content_downloader.adapters.x.mapper import info_to_content_item
from content_downloader.models import ContentItem, DownloadError, DownloadResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL patterns
# ---------------------------------------------------------------------------

_TWEET_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:x|twitter)\.com/[^/]+/status/\d+"
)
_PROFILE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:x|twitter)\.com/[^/?#]+"
)

_PROFILE_UNSUPPORTED_MESSAGE = (
    "X/Twitter profile batch download is not supported.\n"
    "yt-dlp does not support batch-downloading a user's full timeline.\n"
    "Download individual tweet URLs instead:\n"
    "  content_downloader download https://x.com/user/status/123"
)


class XDownloadError(RuntimeError):
    """Exception raised by XAdapter on unrecoverable download failures.

    Attributes:
        download_error: The structured DownloadError payload.
    """

    def __init__(self, download_error: DownloadError) -> None:
        super().__init__(download_error.message)
        self.download_error = download_error


class XAdapter:
    """Adapter for downloading content from X (Twitter).

    Uses yt-dlp as an external CLI tool to fetch media + metadata.

    Args:
        fetcher: Optional XFetcher instance (injectable for testing).
    """

    platform = "x"

    def __init__(self, fetcher: Optional[XFetcher] = None) -> None:
        self._fetcher = fetcher or XFetcher()

    # ------------------------------------------------------------------
    # PlatformAdapter protocol
    # ------------------------------------------------------------------

    def can_handle(self, url: str) -> bool:
        """Return True if the URL is a supported X/Twitter tweet URL."""
        return bool(_TWEET_URL_RE.match(url))

    async def download_single(
        self,
        url: str,
        output_dir: Path,
    ) -> ContentItem:
        """Download a single tweet's media and metadata.

        Steps:
        1. Verify yt-dlp is installed.
        2. Run yt-dlp to download media + .info.json.
        3. Handle text-only tweets (no media file → media_files=[]).
        4. Map metadata to ContentItem and write output files.

        Raises:
            XDownloadError: If yt-dlp is unavailable or download fails.
        """
        # 1. Check yt-dlp availability
        if not await self._fetcher.is_available():
            raise XDownloadError(
                DownloadError(
                    content_id="",
                    source_url=url,
                    error_type="service_unavailable",
                    message=_YTDLP_NOT_INSTALLED_MESSAGE,
                    retryable=False,
                )
            )

        # 2. Determine output directory (we will refine after we know author/id)
        # Use a temp staging area first, then move to canonical path
        staging_dir = output_dir / "_x_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        # 3. Fetch via yt-dlp
        try:
            info = await self._fetcher.fetch_post(url, staging_dir)
        except FileNotFoundError:
            # Text-only tweet: yt-dlp succeeded but produced no info.json
            # This happens when yt-dlp downloads a tweet with no media
            # We treat it as a text post with empty media_files
            info = _build_text_only_info(url)
        except RuntimeError as exc:
            raise XDownloadError(
                DownloadError(
                    content_id="",
                    source_url=url,
                    error_type="network",
                    message=str(exc),
                    retryable=True,
                )
            ) from exc

        # 4. Determine canonical content directory
        content_id = str(info.get("id") or "unknown")
        author_id = str(info.get("uploader_id") or "unknown")
        content_dir = output_dir / "x" / author_id / content_id
        media_dir = content_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        # 5. Move staged media files to canonical location
        media_files: list[str] = []
        cover_file: Optional[str] = None

        staging_media = staging_dir / "media"
        if staging_media.exists():
            for src in staging_media.iterdir():
                if src.suffix == ".json":
                    # Skip .info.json files — we handle metadata separately
                    continue
                dest = media_dir / src.name
                src.rename(dest)
                if src.suffix in (".jpg", ".jpeg", ".png", ".webp"):
                    # All image files go into media_files
                    media_files.append(f"media/{dest.name}")
                    if cover_file is None:
                        # First image file also becomes the cover
                        cover_file = f"media/{dest.name}"
                elif src.suffix in (".mp4", ".mkv", ".webm", ".mov"):
                    media_files.append(f"media/{dest.name}")

            # Clean up staging directory
            try:
                staging_media.rmdir()
                staging_dir.rmdir()
            except OSError:
                pass  # Not empty — leave it, not critical

        # For image tweets: images are downloaded as thumbnails by yt-dlp
        # The thumbnail file IS the image
        if not media_files and not cover_file:
            # Check if there's a thumbnail
            thumbnail_path = _find_thumbnail(media_dir)
            if thumbnail_path:
                cover_file = f"media/{thumbnail_path.name}"

        # 6. Write metadata.json (raw yt-dlp info)
        metadata_path = content_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 7. Build ContentItem
        # Determine final content type based on what was actually downloaded
        content_type = _resolve_content_type(info, media_files, cover_file)
        item = info_to_content_item(
            info,
            source_url=url,
            media_files=media_files,
            cover_file=cover_file,
        )
        # Override content_type based on actual files
        item = item.model_copy(update={"content_type": content_type})

        # 8. Write content_item.json
        (content_dir / "content_item.json").write_text(
            item.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # 9. Write text.txt — standalone readable text file
        text_content = item.description or item.title or ""
        if text_content:
            (content_dir / "text.txt").write_text(
                text_content, encoding="utf-8",
            )

        return item

    async def download_profile(
        self,
        profile_url: str,
        output_dir: Path,
        limit: int = 0,
        since: Optional[datetime] = None,
    ) -> DownloadResult:
        """X profile batch download is not supported.

        yt-dlp does not support batch-downloading a user's full timeline.

        Returns:
            A DownloadResult with one unsupported error entry.
        """
        logger.warning("X profile download not supported: %s", profile_url)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_text_only_info(url: str) -> "dict[str, Any]":
    """Build a minimal info dict for a text-only tweet (no media)."""
    # Extract tweet ID from URL if possible
    m = re.search(r"/status/(\d+)", url)
    tweet_id = m.group(1) if m else "unknown"
    return {
        "id": tweet_id,
        "title": "",
        "description": "",
        "uploader": "",
        "uploader_id": "",
        "timestamp": None,
        "webpage_url": url,
        "ext": "na",
        "formats": [],
        "thumbnails": [],
    }


def _find_thumbnail(media_dir: Path) -> "Optional[Path]":
    """Find the first thumbnail/image file in *media_dir*."""
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        matches = list(media_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def _resolve_content_type(
    info: "dict[str, Any]",
    media_files: list[str],
    cover_file: "Optional[str]",
) -> str:
    """Determine content_type from actual downloaded files."""
    has_video = any(
        f.endswith((".mp4", ".mkv", ".webm", ".mov")) for f in media_files
    )
    if has_video:
        return "video"
    if cover_file or media_files:
        return "image"
    return "text"
