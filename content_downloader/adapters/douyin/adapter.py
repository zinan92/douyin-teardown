"""Douyin platform adapter — implements PlatformAdapter protocol."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from content_downloader.adapters.douyin.api_client import DouyinAPIClient
from content_downloader.adapters.douyin.mapper import aweme_to_content_item
from content_downloader.models import ContentItem, DownloadError, DownloadResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL patterns
# ---------------------------------------------------------------------------

_VIDEO_URL_RE = re.compile(r"https?://(?:www\.)?douyin\.com/video/(\d+)")
_USER_URL_RE = re.compile(r"https?://(?:www\.)?douyin\.com/user/([A-Za-z0-9_\-]+)")
_SHORT_URL_RE = re.compile(r"https?://v\.douyin\.com/")
_MODAL_URL_RE = re.compile(r"modal_id=(\d+)")


def _extract_aweme_id(url: str) -> Optional[str]:
    """Extract aweme_id from a canonical douyin.com/video/... URL."""
    m = _VIDEO_URL_RE.search(url)
    if m:
        return m.group(1)
    m = _MODAL_URL_RE.search(url)
    if m:
        return m.group(1)
    return None


def _extract_sec_uid(url: str) -> Optional[str]:
    """Extract sec_uid from a douyin.com/user/... URL.

    Returns None for pseudo-paths like /user/self which are not real sec_uids.
    """
    m = _USER_URL_RE.search(url)
    if m:
        uid = m.group(1)
        # /user/self is the logged-in user's own profile — not a real sec_uid
        if uid.lower() == "self":
            return None
        return uid
    return None


# ---------------------------------------------------------------------------
# No-watermark URL extraction (migrated from downloader_base.py lines 482-527)
# ---------------------------------------------------------------------------

def _build_no_watermark_url(
    aweme_data: Dict[str, Any],
    api_client: DouyinAPIClient,
) -> Optional[str]:
    """Extract the best no-watermark video URL from aweme data.

    Priority:
    1. download_addr CDN URLs (direct download, most reliable)
    2. play_addr CDN URLs with 'watermark=0'
    3. Other CDN URLs from play_addr
    4. Constructed from video URI via /aweme/v1/play/ (least reliable)
    """
    video = aweme_data.get("video") or {}

    # Priority 1: download_addr CDN URLs (most reliable for full download)
    dl_addr = video.get("download_addr") or {}
    dl_urls = [u for u in (dl_addr.get("url_list") or []) if u]
    for candidate in dl_urls:
        parsed = urlparse(candidate)
        if not parsed.netloc.endswith("douyin.com"):
            return candidate  # CDN URL, use directly

    # Priority 2-3: play_addr URLs
    play_addr = video.get("play_addr") or {}
    url_candidates = [c for c in (play_addr.get("url_list") or []) if c]
    url_candidates.sort(key=lambda u: 0 if "watermark=0" in u else 1)

    fallback_candidate: Optional[str] = None

    for candidate in url_candidates:
        parsed = urlparse(candidate)
        if not parsed.netloc.endswith("douyin.com"):
            fallback_candidate = candidate
        elif "X-Bogus=" not in candidate:
            signed_url, _ua = api_client.sign_url(candidate)
            return signed_url
        else:
            return candidate

    if fallback_candidate:
        return fallback_candidate

    # Priority 4: Build from video URI (least reliable, may get rate-limited)
    uri = (
        play_addr.get("uri")
        or video.get("vid")
        or dl_addr.get("uri")
    )
    if uri:
        params = {
            "video_id": uri,
            "ratio": "1080p",
            "line": "0",
            "is_play_url": "1",
            "watermark": "0",
            "source": "PackSourceEnum_PUBLISH",
        }
        signed_url, _ua = api_client.build_signed_path("/aweme/v1/play/", params)
        return signed_url

    return fallback_candidate


# ---------------------------------------------------------------------------
# DouyinAdapter
# ---------------------------------------------------------------------------

class DouyinAdapter:
    """Adapter for downloading content from Douyin (抖音)."""

    platform = "douyin"

    def __init__(
        self,
        cookies: Optional[Dict[str, str]] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self._cookies = cookies or {}
        self._proxy = proxy

    @property
    def has_cookies(self) -> bool:
        """Return whether the adapter has any loaded login cookies."""
        return bool(self._cookies)

    def can_handle(self, url: str) -> bool:
        """Return True if this adapter can handle the URL."""
        return bool(
            _VIDEO_URL_RE.search(url)
            or _USER_URL_RE.search(url)
            or _SHORT_URL_RE.search(url)
        )

    def _make_client(self) -> DouyinAPIClient:
        return DouyinAPIClient(cookies=self._cookies, proxy=self._proxy)

    async def download_single(
        self,
        url: str,
        output_dir: Path,
    ) -> ContentItem:
        """Download a single Douyin video (or gallery) from url.

        Steps:
        1. Resolve short link if needed
        2. Extract aweme_id from URL
        3. Fetch aweme detail from API
        4. Extract no-watermark video URL
        5. Download video + cover to output_dir
        6. Write metadata.json and content_item.json
        7. Return ContentItem

        Args:
            url: A douyin.com or v.douyin.com URL.
            output_dir: Root output directory.

        Returns:
            Populated ContentItem.

        Raises:
            ValueError: If the URL cannot be parsed.
            RuntimeError: If the API call fails.
        """
        async with self._make_client() as client:
            # 1. Resolve short link
            resolved_url = url
            if _SHORT_URL_RE.search(url):
                resolved_url = await client.resolve_short_url(url) or url
                logger.debug("Resolved short URL %s -> %s", url, resolved_url)

            # 2. Extract aweme_id
            aweme_id = _extract_aweme_id(resolved_url)
            if not aweme_id:
                raise ValueError(
                    f"Cannot extract aweme_id from URL: {url!r} (resolved: {resolved_url!r})"
                )

            # 3. Fetch aweme detail
            aweme_data = await client.get_video_detail(aweme_id)
            if not aweme_data:
                raise RuntimeError(f"Failed to fetch video detail for aweme_id={aweme_id} (the API refused and the web page does not open it: private, removed or restricted?)")

            return await self._save_aweme(aweme_data, url, output_dir, client)

    async def _save_aweme(
        self,
        aweme_data: Dict[str, Any],
        source_url: str,
        output_dir: Path,
        client: DouyinAPIClient,
    ) -> ContentItem:
        """Download media files and write all output files for a single aweme.

        Args:
            aweme_data: Raw aweme dict from the Douyin API.
            source_url: The original URL (for ContentItem.source_url fallback).
            output_dir: Root output directory.
            client: Open DouyinAPIClient to use for HTTP downloads.

        Returns:
            Populated ContentItem.
        """
        author = aweme_data.get("author") or {}
        author_id = str(author.get("uid") or "unknown")
        aweme_id = str(aweme_data.get("aweme_id") or "unknown")

        content_dir = output_dir / "douyin" / author_id / aweme_id
        media_dir = content_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        media_files: List[str] = []
        cover_file: Optional[str] = None

        # Detect gallery vs video
        # Some gallery posts use "image_post_info" instead of "images"
        gallery_items = _iter_gallery_items(aweme_data)
        if gallery_items is not None:
            # Gallery: download each image
            for idx, img_item in enumerate(gallery_items):
                img_url = _pick_first_url(img_item)
                if img_url:
                    filename = f"image_{idx + 1:03d}.jpg"
                    dest = media_dir / filename
                    await _download_file(client, img_url, dest)
                    media_files.append(f"media/{filename}")
            # Set cover_file to first image for gallery posts
            if media_files:
                cover_file = media_files[0]
        else:
            # Video: extract no-watermark URL
            video_url = _build_no_watermark_url(aweme_data, client)
            if video_url:
                video_dest = media_dir / "video.mp4"
                await _download_file(client, video_url, video_dest)
                media_files.append("media/video.mp4")

        # Cover image (only download API cover for video posts; gallery uses first image)
        if cover_file is None:
            cover_url = _extract_cover_url(aweme_data)
            if cover_url:
                cover_dest = media_dir / "cover.jpg"
                await _download_file(client, cover_url, cover_dest)
                cover_file = "media/cover.jpg"

        # Write metadata.json
        metadata_path = content_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(aweme_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Build ContentItem
        item = aweme_to_content_item(
            aweme_data,
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
        """Batch-download videos from a Douyin creator profile.

        Args:
            profile_url: URL to the creator's profile page.
            output_dir: Root output directory.
            limit: Max number of items to download (0 = no limit).
            since: Only download items published after this datetime.

        Returns:
            DownloadResult with items, errors, and counts.
        """
        sec_uid = _extract_sec_uid(profile_url)
        if not sec_uid:
            raise ValueError(f"Cannot extract sec_uid from profile URL: {profile_url!r}")

        since_ts = since.timestamp() if since else None

        items: List[ContentItem] = []
        errors: List[DownloadError] = []

        async with self._make_client() as client:
            max_cursor = 0
            has_more = True

            while has_more:
                page_data = await client.get_user_post(sec_uid, max_cursor, 20)
                aweme_list: List[Dict[str, Any]] = page_data.get("aweme_list") or []

                if not aweme_list:
                    # Empty page — stop paging
                    break

                for aweme in aweme_list:
                    if limit > 0 and len(items) + len(errors) >= limit:
                        has_more = False
                        break

                    aweme_id = str(aweme.get("aweme_id") or "")
                    aweme_url = aweme.get("share_url") or profile_url

                    # since filter
                    if since_ts is not None:
                        create_time = int(aweme.get("create_time") or 0)
                        if create_time <= since_ts:
                            continue

                    try:
                        item = await self._save_aweme(aweme, aweme_url, output_dir, client)
                        items.append(item)
                    except Exception as exc:
                        logger.error(
                            "Failed to download aweme_id=%s: %s", aweme_id, exc
                        )
                        errors.append(
                            DownloadError(
                                content_id=aweme_id,
                                source_url=aweme_url,
                                error_type="network",
                                message=str(exc),
                                retryable=True,
                            )
                        )

                prev_cursor = max_cursor
                has_more = bool(page_data.get("has_more", False))
                max_cursor = int(page_data.get("max_cursor") or 0)

                # Stagnation guard: cursor did not advance
                if has_more and max_cursor == prev_cursor:
                    logger.warning(
                        "max_cursor did not advance (%s), stopping pagination to avoid loop",
                        max_cursor,
                    )
                    break

        total = len(items) + len(errors)
        return DownloadResult(
            items=items,
            errors=errors,
            total=total,
            success=len(items),
            failed=len(errors),
            skipped=0,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STALL_THRESHOLD = 4096  # bytes — less than this per attempt counts as stalled
_MAX_STALL = 5  # consecutive stalled attempts before giving up
_MAX_RESUME_ATTEMPTS = 40  # hard ceiling on total reconnects


async def _download_file(
    client: DouyinAPIClient,
    url: str,
    dest: Path,
) -> None:
    """Download a file from url to dest with resume-on-partial support.

    Douyin's CDN can force-close HTTP/2 streams after delivering a fixed chunk
    (1-8 MB observed). When this happens, we reconnect with a ``Range`` header
    from the current file offset and keep appending until the full
    ``Content-Length`` is satisfied.

    Falls back to a single-pass download when the server does not advertise
    ``Content-Length`` on the first response (e.g. chunked transfer).
    """
    clean_url = url.encode("utf-8").decode("unicode_escape") if "\\u" in url else url
    base_headers = {**client.headers, "Referer": "https://www.douyin.com/"}

    dest.parent.mkdir(parents=True, exist_ok=True)

    expected: int | None = None
    stall_count = 0

    async with httpx.AsyncClient(
        cookies=client.cookies,
        timeout=httpx.Timeout(300.0, connect=30.0),
        follow_redirects=True,
    ) as http:
        for attempt in range(1, _MAX_RESUME_ATTEMPTS + 1):
            pos = dest.stat().st_size if dest.exists() else 0

            # Already complete?
            if expected is not None and pos >= expected:
                break

            headers = dict(base_headers)
            if pos > 0:
                headers["Range"] = f"bytes={pos}-"

            before = pos
            try:
                async with http.stream("GET", clean_url, headers=headers) as resp:
                    # On the first attempt, record expected total size.
                    if attempt == 1 and expected is None:
                        cl = resp.headers.get("content-length")
                        if cl:
                            expected = int(cl)

                    if resp.status_code not in (200, 206):
                        resp.raise_for_status()

                    with open(dest, "ab") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)

            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout):
                # Expected — CDN closed on us. Will retry with Range.
                pass
            except httpx.HTTPStatusError:
                logger.warning("Failed to download %s -> %s: HTTP %s",
                               clean_url, dest, resp.status_code)
                raise

            after = dest.stat().st_size if dest.exists() else 0

            # Done?
            if expected is not None and after >= expected:
                break

            # No Content-Length → can't resume; treat as single-pass failure.
            if expected is None:
                if after == before:
                    logger.warning("Failed to download %s -> %s: no data received "
                                   "and no Content-Length for resume", clean_url, dest)
                    raise DownloadError(
                        f"Download failed (no Content-Length) for {clean_url}"
                    )
                break

            # Stall detection
            delta = after - before
            if delta < _STALL_THRESHOLD:
                stall_count += 1
                if stall_count >= _MAX_STALL:
                    logger.warning(
                        "Download stalled at %d/%d after %d low-progress attempts: %s",
                        after, expected, stall_count, clean_url,
                    )
                    raise DownloadError(
                        f"Download stalled at {after}/{expected} for {clean_url}"
                    )
            else:
                stall_count = 0
        else:
            after = dest.stat().st_size if dest.exists() else 0
            logger.warning(
                "Exhausted %d resume attempts at %d/%s: %s",
                _MAX_RESUME_ATTEMPTS, after, expected, clean_url,
            )
            raise DownloadError(
                f"Exhausted {_MAX_RESUME_ATTEMPTS} resume attempts for {clean_url}"
            )


def _pick_first_url(item: Any) -> Optional[str]:
    """Pick the first non-empty URL from an image item dict."""
    if not isinstance(item, dict):
        return None
    for key in ("url_list", "url_list_webp"):
        urls = item.get(key)
        if isinstance(urls, list):
            for u in urls:
                if u:
                    return u
    url = item.get("url")
    return url if url else None


def _iter_gallery_items(aweme_data: Dict[str, Any]) -> Optional[List[Any]]:
    """Extract gallery image items from aweme data.

    Checks both ``image_post_info`` (newer API format) and ``images`` (older format).

    Returns:
        A non-empty list of image item dicts if this is a gallery post,
        or None if it is not a gallery post.
    """
    image_post = aweme_data.get("image_post_info")
    image_post_images = (
        image_post.get("images") if isinstance(image_post, dict) else None
    )
    images = image_post_images or aweme_data.get("images")
    if isinstance(images, list) and len(images) > 0:
        return images
    # Key present but empty list, or key present via image_post_info with empty images
    # → this is a gallery post with no usable images, return empty list (not None)
    if image_post is not None or aweme_data.get("images") is not None:
        return []
    return None


def _extract_cover_url(aweme_data: Dict[str, Any]) -> Optional[str]:
    """Extract the cover/thumbnail URL from aweme data."""
    video = aweme_data.get("video") or {}
    for key in ("cover", "origin_cover", "dynamic_cover", "static_cover"):
        cover = video.get(key)
        if isinstance(cover, dict):
            url_list = cover.get("url_list") or []
            for u in url_list:
                if u:
                    return u
    return None
