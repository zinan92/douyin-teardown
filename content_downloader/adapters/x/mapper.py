"""Maps yt-dlp info.json dict to ContentItem model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from content_downloader.models import ContentItem


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(raw: Any) -> str:
    """Convert a Unix timestamp (seconds) to ISO 8601 string."""
    if raw is None:
        return _now_iso()
    try:
        ts = float(raw)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except (ValueError, OSError, OverflowError):
        return _now_iso()


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int, returning default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _detect_content_type(info: dict[str, Any]) -> str:
    """Detect content type from yt-dlp info dict.

    yt-dlp marks video tweets as ext='mp4' / has formats with vcodec.
    Image tweets have thumbnails but no video stream.
    Text-only tweets have neither.
    """
    # yt-dlp sets _type='video' even for image tweets sometimes,
    # but we can check for actual video format presence
    ext = info.get("ext", "")
    formats = info.get("formats") or []

    has_video = ext in ("mp4", "mkv", "webm", "mov") or any(
        f.get("vcodec") not in (None, "none") for f in formats
    )
    if has_video:
        return "video"

    # Image tweet: thumbnails exist but no video
    thumbnails = info.get("thumbnails") or []
    thumbnail = info.get("thumbnail")
    if thumbnails or thumbnail:
        return "image"

    return "text"


def info_to_content_item(
    info: dict[str, Any],
    source_url: str,
    media_files: list[str] | None = None,
    cover_file: str | None = None,
) -> ContentItem:
    """Convert a yt-dlp info.json dict to a ContentItem.

    yt-dlp produces these relevant fields for X/Twitter:
    - id: tweet ID
    - title / description: tweet text
    - uploader: display name
    - uploader_id: @handle
    - timestamp: Unix seconds
    - like_count, repost_count, comment_count, view_count
    - webpage_url: canonical URL

    Args:
        info: Parsed yt-dlp .info.json dict.
        source_url: Original URL passed by the caller (fallback for webpage_url).
        media_files: Relative paths to downloaded media files.
        cover_file: Relative path to the cover/thumbnail image.

    Returns:
        A fully populated ContentItem.
    """
    content_id = str(info.get("id") or "unknown")
    title = str(info.get("title") or info.get("description") or "")[:100]
    description = str(info.get("description") or info.get("title") or "")
    author_name = str(info.get("uploader") or "")
    author_id = str(info.get("uploader_id") or "")
    publish_time = _parse_timestamp(info.get("timestamp"))
    url = str(info.get("webpage_url") or source_url)
    content_type = _detect_content_type(info)

    return ContentItem(
        platform="x",
        content_id=content_id,
        content_type=content_type,
        title=title,
        description=description,
        author_id=author_id,
        author_name=author_name,
        publish_time=publish_time,
        source_url=url,
        media_files=media_files or [],
        cover_file=cover_file,
        metadata_file="metadata.json",
        likes=_safe_int(info.get("like_count")),
        comments=_safe_int(info.get("comment_count")),
        shares=_safe_int(info.get("repost_count")),
        collects=0,
        views=_safe_int(info.get("view_count")),
        downloaded_at=_now_iso(),
    )
