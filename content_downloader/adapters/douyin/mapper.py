"""Maps raw Douyin aweme_data dicts to ContentItem models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from content_downloader.models import ContentItem


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def aweme_to_content_item(
    aweme_data: Dict[str, Any],
    source_url: str,
    media_files: Optional[List[str]] = None,
    cover_file: Optional[str] = None,
) -> ContentItem:
    """Convert raw Douyin aweme response dict to a ContentItem.

    Args:
        aweme_data: Raw aweme dict from the Douyin API.
        source_url: The original URL used to request this content.
        media_files: Relative paths to downloaded media files (e.g. ['media/video.mp4']).
        cover_file: Relative path to cover image (e.g. 'media/cover.jpg').

    Returns:
        Populated ContentItem instance.
    """
    author = aweme_data.get("author") or {}
    stats = aweme_data.get("statistics") or {}
    create_time = aweme_data.get("create_time", 0)

    try:
        publish_time = datetime.fromtimestamp(int(create_time), tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        publish_time = datetime.now(timezone.utc).isoformat()

    # Detect content type: image posts have an images field
    if aweme_data.get("images") and isinstance(aweme_data["images"], list):
        content_type = "gallery"
    else:
        content_type = "video"

    desc = aweme_data.get("desc") or ""

    return ContentItem(
        platform="douyin",
        content_id=str(aweme_data.get("aweme_id") or ""),
        content_type=content_type,
        title=desc[:100],
        description=desc,
        author_id=str(author.get("uid") or ""),
        author_name=str(author.get("nickname") or ""),
        publish_time=publish_time,
        source_url=aweme_data.get("share_url") or source_url,
        media_files=media_files or [],
        cover_file=cover_file,
        metadata_file="metadata.json",
        likes=int(stats.get("digg_count") or 0),
        comments=int(stats.get("comment_count") or 0),
        shares=int(stats.get("share_count") or 0),
        collects=int(stats.get("collect_count") or 0),
        views=int(stats.get("play_count") or 0),
        downloaded_at=_now_iso(),
    )
