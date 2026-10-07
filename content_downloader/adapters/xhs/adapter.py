"""XHS (Xiaohongshu / 小红书) platform adapter — implements PlatformAdapter protocol."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from content_downloader.adapters.xhs.api_client import XHSAPIClient
from content_downloader.adapters.xhs.mapper import (
    extract_author_id,
    extract_download_urls,
    extract_note_id,
    note_to_content_item,
)
from content_downloader.adapters.xhs.sidecar import XHSSidecar
from content_downloader.models import ContentItem, DownloadError, DownloadResult


class XHSDownloadError(RuntimeError):
    """Exception raised by XHSAdapter on unrecoverable download failures.

    Carries a :class:`~content_downloader.models.DownloadError` payload
    with structured error information.

    Attributes:
        download_error: The structured DownloadError payload.
    """

    def __init__(self, download_error: DownloadError) -> None:
        super().__init__(download_error.message)
        self.download_error = download_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL patterns
# ---------------------------------------------------------------------------

_NOTE_EXPLORE_RE = re.compile(
    r"https?://(?:www\.)?xiaohongshu\.com/explore/([A-Za-z0-9]+)"
)
_NOTE_DISCOVERY_RE = re.compile(
    r"https?://(?:www\.)?xiaohongshu\.com/discovery/(?:item/)?([A-Za-z0-9]+)"
)
_SHORT_LINK_RE = re.compile(r"https?://xhslink\.com/\S+")
_PROFILE_URL_RE = re.compile(
    r"https?://(?:www\.)?xiaohongshu\.com/user/profile/([A-Za-z0-9]+)"
)

_NOTE_PATTERNS = [_NOTE_EXPLORE_RE, _NOTE_DISCOVERY_RE, _SHORT_LINK_RE]

_SERVICE_UNAVAILABLE_MESSAGE = (
    "XHS-Downloader is not running.\n"
    "Start it with:\n"
    "  cd /path/to/XHS-Downloader\n"
    "  python main.py api\n"
    "It will start on http://127.0.0.1:5556"
)

_PROFILE_UNSUPPORTED_MESSAGE = (
    "Profile batch download is not supported via the XHS-Downloader HTTP API.\n"
    "Use the XHS-Downloader CLI in creator mode instead:\n"
    "  cd /path/to/XHS-Downloader\n"
    "  python main.py --type creator\n"
    "Then enter the profile URL when prompted."
)


class XHSAdapter:
    """Adapter for downloading content from Xiaohongshu (小红书 / XHS).

    Communicates with a locally running XHS-Downloader instance via its
    HTTP API (sidecar mode).  Does NOT embed XHS-Downloader source code.

    Args:
        base_url: Base URL of the XHS-Downloader sidecar.
    """

    platform = "xhs"

    def __init__(self, base_url: str = "http://127.0.0.1:5556") -> None:
        self._base_url = base_url
        self._sidecar = XHSSidecar(base_url=base_url)

    # ------------------------------------------------------------------
    # PlatformAdapter protocol
    # ------------------------------------------------------------------

    def can_handle(self, url: str) -> bool:
        """Return True if the URL is a supported XHS note or profile URL."""
        return any(p.search(url) for p in [*_NOTE_PATTERNS, _PROFILE_URL_RE])

    async def download_single(
        self,
        url: str,
        output_dir: Path,
    ) -> ContentItem:
        """Download a single XHS note (image gallery or video).

        Steps:
        1. Check that the XHS-Downloader sidecar is running.
        2. Fetch note detail via ``POST /xhs/detail``.
        3. Download all media files to ``output_dir/xhs/{author_id}/{note_id}/media/``.
        4. Write ``metadata.json`` (raw API response) and ``content_item.json``.
        5. Return a fully populated :class:`~content_downloader.models.ContentItem`.

        Raises:
            DownloadError: As a *raised* exception when the sidecar is unavailable
                           or the API returns no data.
        """
        async with XHSAPIClient(base_url=self._base_url) as client:
            # 1. Ensure sidecar is running (auto-install + auto-start)
            if not await client.is_available():
                logger.info("XHS-Downloader not running, attempting auto-start...")
                if not await self._sidecar.ensure_running():
                    raise XHSDownloadError(
                        DownloadError(
                            content_id="",
                            source_url=url,
                            error_type="service_unavailable",
                            message=(
                                "Failed to auto-start XHS-Downloader.\n"
                                "Manual install: pip install xhs-downloader\n"
                                "Then: python -m xhs_downloader api"
                            ),
                            retryable=False,
                        )
                    )

            # 2. Fetch note detail
            note_data = await client.get_note_detail(url)

        if not note_data:
            raise XHSDownloadError(
                DownloadError(
                    content_id="",
                    source_url=url,
                    error_type="not_found",
                    message=f"XHS-Downloader returned empty response for URL: {url!r}",
                    retryable=False,
                )
            )

        return await self._save_note(note_data, url, output_dir)

    async def download_profile(
        self,
        profile_url: str,
        output_dir: Path,
        limit: int = 0,
        since: Optional[datetime] = None,
    ) -> DownloadResult:
        """Batch-download content from an XHS creator profile.

        The XHS-Downloader HTTP API does not expose a profile/creator batch
        endpoint.  This method raises an ``unsupported`` error with clear
        workaround instructions.

        Returns:
            A :class:`~content_downloader.models.DownloadResult` with one error
            entry describing the unsupported operation.
        """
        user_id = ""
        m = _PROFILE_URL_RE.search(profile_url)
        if m:
            user_id = m.group(1)

        error = DownloadError(
            content_id=user_id,
            source_url=profile_url,
            error_type="unsupported",
            message=_PROFILE_UNSUPPORTED_MESSAGE,
            retryable=False,
        )
        logger.warning(
            "XHS profile download not supported via HTTP API: %s", profile_url
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

    async def _save_note(
        self,
        note_data: dict[str, Any],
        source_url: str,
        output_dir: Path,
    ) -> ContentItem:
        """Download media files and write all output files for a single note.

        XHS-Downloader returns ``{"data": {中文字段..., "下载地址": [urls]}}``
        """
        note_id = extract_note_id(note_data)
        author_id = extract_author_id(note_data)
        download_urls = extract_download_urls(note_data)

        content_dir = output_dir / "xhs" / author_id / note_id
        media_dir = content_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        media_files: list[str] = []
        cover_file: Optional[str] = None

        # Download all media from 下载地址 list
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
            for idx, url in enumerate(download_urls):
                if not url:
                    continue
                # Detect file type from URL
                if any(ext in url for ext in [".mp4", ".mov", ".webm"]):
                    filename = f"video_{idx + 1:02d}.mp4" if len(download_urls) > 1 else "video.mp4"
                else:
                    filename = f"img_{idx + 1:02d}.jpg"

                dest = media_dir / filename
                try:
                    resp = await http.get(url)
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                    media_files.append(f"media/{filename}")
                    logger.info("Downloaded %s -> %s", url[:80], dest.name)
                except Exception as exc:
                    logger.warning("Failed to download %s: %s", url[:80], exc)

        # Set cover_file to first downloaded image (not video)
        for mf in media_files:
            if not any(mf.endswith(ext) for ext in (".mp4", ".mov", ".webm")):
                cover_file = mf
                break

        # Write metadata.json (raw API response)
        metadata_path = content_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(note_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Build ContentItem
        item = note_to_content_item(
            note_data,
            source_url=source_url,
            media_files=media_files,
            cover_file=cover_file,
        )

        # Write content_item.json
        (content_dir / "content_item.json").write_text(
            item.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # Write text.txt — standalone readable text file
        text_parts = [item.title, item.description]
        text_content = "\n\n".join(p for p in text_parts if p)
        if text_content:
            (content_dir / "text.txt").write_text(
                text_content, encoding="utf-8",
            )

        return item
