"""Stub adapter for platforms not yet implemented."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from content_downloader.models import ContentItem, DownloadResult


class StubAdapter:
    """Placeholder adapter that raises NotImplementedError for unimplemented platforms.

    Registered in the router so `platforms` command can list all platforms.
    Real implementations will replace this in Phase 2+.
    """

    def __init__(self, platform: str) -> None:
        self.platform = platform

    def can_handle(self, url: str) -> bool:
        return False  # Never selected by the router directly

    async def download_single(self, url: str, output_dir: Path) -> ContentItem:
        raise NotImplementedError(
            f"Platform '{self.platform}' adapter is not yet implemented. "
            f"Phase 2+ will add real adapters for douyin, xhs, wechat_oa, and x."
        )

    async def download_profile(
        self,
        profile_url: str,
        output_dir: Path,
        limit: int = 0,
        since: datetime | None = None,
    ) -> DownloadResult:
        raise NotImplementedError(
            f"Platform '{self.platform}' adapter is not yet implemented."
        )
