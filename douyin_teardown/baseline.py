from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable


BaselineFn = Callable[[dict[str, Any]], "dict[str, Any] | None"]


class BaselineError(RuntimeError):
    """The author's recent posts could not be fetched."""


def summarize_posts(posts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Median likes over an author's non-pinned recent posts."""
    likes = [
        int((post.get("statistics") or {}).get("digg_count") or 0)
        for post in posts
        if not post.get("is_top")
    ]
    likes = [value for value in likes if value >= 0]
    if not likes:
        return None
    return {
        "post_count": len(likes),
        "median_likes": statistics.median(likes),
        "excludes_pinned": True,
    }


async def _fetch_posts(sec_uid: str, cookies: dict[str, str], pages: int, delay_seconds: float) -> list[dict]:
    try:
        from content_downloader.adapters.douyin.api_client import DouyinAPIClient
    except ImportError as exc:
        raise BaselineError("content-downloader is not importable; cannot fetch author posts") from exc

    posts: list[dict] = []
    cursor = 0
    async with DouyinAPIClient(cookies=cookies) as client:
        for page in range(pages):
            data = await client.get_user_post(sec_uid, max_cursor=cursor, count=20)
            risk = data.get("risk_flags") or {}
            if risk.get("verify_page"):
                raise BaselineError("Douyin returned a verification page; stopped fetching author posts")
            posts.extend(data.get("items") or [])
            if not data.get("has_more"):
                break
            cursor = int(data.get("max_cursor") or 0)
            if page + 1 < pages:
                await asyncio.sleep(delay_seconds)
    return posts


def douyin_baseline(
    sec_uid: str,
    *,
    cookies: dict[str, str],
    cache_dir: Path,
    pages: int = 3,
    delay_seconds: float = 1.5,
    max_age_seconds: float = 24 * 3600,
) -> dict[str, Any] | None:
    """Fetch (or reuse a fresh cached) like-count baseline for one author."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sec_uid}.json"
    if cache_path.is_file() and time.time() - cache_path.stat().st_mtime < max_age_seconds:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    posts = asyncio.run(_fetch_posts(sec_uid, cookies, pages, delay_seconds))
    summary = summarize_posts(posts)
    if summary is None:
        return None
    summary["fetched_at"] = datetime.now(timezone.utc).isoformat()
    cache_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cache_path.chmod(0o600)
    return summary
