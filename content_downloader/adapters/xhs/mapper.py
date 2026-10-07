"""Maps raw XHS-Downloader API response dicts to ContentItem models.

XHS-Downloader returns Chinese field names in ``response["data"]``:
- 作品ID, 作品标题, 作品描述, 作品类型 (视频/图文)
- 作者ID, 作者昵称, 作者链接
- 点赞数量, 评论数量, 分享数量, 收藏数量
- 发布时间, 时间戳
- 下载地址 (list of URLs), 动图地址
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from content_downloader.models import ContentItem


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_xhs_time(raw: Any) -> str:
    """Convert XHS timestamp to ISO 8601.

    Handles:
    - float/int epoch seconds (时间戳 field)
    - string like "2025-10-20_18:29:20" (发布时间 field)
    """
    if raw is None:
        return _now_iso()
    # Epoch seconds (float or int)
    if isinstance(raw, (int, float)):
        try:
            dt = datetime.fromtimestamp(raw, tz=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except (OSError, OverflowError):
            return _now_iso()
    # String format "2025-10-20_18:29:20"
    if isinstance(raw, str):
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    return _now_iso()


def _parse_count(raw: Any) -> int:
    """Parse Chinese count strings like '3.8万', '1220', '3.1万' to int."""
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).strip()
    if not s:
        return 0
    # Handle 万 (10,000) suffix
    m = re.match(r"^([\d.]+)\s*万$", s)
    if m:
        return int(float(m.group(1)) * 10000)
    # Handle 亿 (100,000,000) suffix
    m = re.match(r"^([\d.]+)\s*亿$", s)
    if m:
        return int(float(m.group(1)) * 100000000)
    try:
        return int(s)
    except ValueError:
        return 0


def _extract_note_data(response: dict[str, Any]) -> dict[str, Any]:
    """Extract the actual note data from XHS-Downloader response.

    Response format: {"message": "...", "params": {...}, "data": {中文字段...}}
    """
    return response.get("data", response)


def note_to_content_item(
    response: dict[str, Any],
    source_url: str,
    media_files: list[str] | None = None,
    cover_file: str | None = None,
) -> ContentItem:
    """Convert XHS-Downloader API response to ContentItem.

    Args:
        response: Full response dict from POST /xhs/detail.
        source_url: Original URL.
        media_files: Relative paths to downloaded media files.
        cover_file: Relative path to cover image.
    """
    d = _extract_note_data(response)

    note_id = str(d.get("作品ID") or d.get("note_id") or d.get("id") or "")
    author_id = str(d.get("作者ID") or d.get("user_id") or "")
    author_name = str(d.get("作者昵称") or d.get("nickname") or "")

    # Type: "视频" → video, anything else → gallery
    raw_type = str(d.get("作品类型") or d.get("type") or "")
    content_type = "video" if "视频" in raw_type or raw_type == "video" else "gallery"

    # Timestamps: prefer 时间戳 (epoch), fallback to 发布时间 (string)
    publish_time = _parse_xhs_time(d.get("时间戳") or d.get("发布时间") or d.get("time"))

    return ContentItem(
        platform="xhs",
        content_id=note_id,
        content_type=content_type,
        title=str(d.get("作品标题") or d.get("title") or ""),
        description=str(d.get("作品描述") or d.get("desc") or ""),
        author_id=author_id,
        author_name=author_name,
        publish_time=publish_time,
        source_url=str(d.get("作品链接") or d.get("note_url") or source_url),
        media_files=media_files or [],
        cover_file=cover_file,
        metadata_file="metadata.json",
        likes=_parse_count(d.get("点赞数量") or d.get("liked_count")),
        comments=_parse_count(d.get("评论数量") or d.get("comment_count")),
        shares=_parse_count(d.get("分享数量") or d.get("share_count")),
        collects=_parse_count(d.get("收藏数量") or d.get("collected_count")),
        views=0,
        downloaded_at=_now_iso(),
    )


def extract_download_urls(response: dict[str, Any]) -> list[str]:
    """Extract media download URLs from XHS-Downloader response."""
    d = _extract_note_data(response)
    urls = d.get("下载地址") or d.get("download_url") or []
    if isinstance(urls, str):
        return [urls] if urls else []
    return [u for u in urls if u]


def extract_note_id(response: dict[str, Any]) -> str:
    d = _extract_note_data(response)
    return str(d.get("作品ID") or d.get("note_id") or d.get("id") or "unknown")


def extract_author_id(response: dict[str, Any]) -> str:
    d = _extract_note_data(response)
    return str(d.get("作者ID") or d.get("user_id") or "unknown")
