"""Base protocol for all platform adapters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from content_downloader.models import ContentItem, DownloadResult


@runtime_checkable
class PlatformAdapter(Protocol):
    """Protocol all platform adapters must satisfy.

    Each adapter handles URL detection, download orchestration, and file writing
    for a single platform. Adapters are stateless by convention; auth credentials
    are injected at construction time.
    """

    platform: str
    """Unique platform identifier, e.g. 'douyin', 'xhs', 'fixture'."""

    async def download_single(
        self,
        url: str,
        output_dir: Path,
    ) -> ContentItem:
        """Download a single content item from `url`.

        Args:
            url: The canonical content URL.
            output_dir: Root output directory; the adapter creates
                ``output_dir/{platform}/{author_id}/{content_id}/`` internally.

        Returns:
            Fully populated ContentItem with all local paths resolved.

        Raises:
            DownloadError (as exception) on unrecoverable failure.
        """
        ...

    async def download_profile(
        self,
        profile_url: str,
        output_dir: Path,
        limit: int = 0,
        since: datetime | None = None,
    ) -> DownloadResult:
        """Batch-download content from a creator profile.

        Args:
            profile_url: URL to the creator's profile / feed page.
            output_dir: Root output directory.
            limit: Maximum number of items to download. 0 = no limit.
            since: Only download items published after this timestamp.

        Returns:
            DownloadResult aggregating successes, errors, and skip counts.
        """
        ...

    def can_handle(self, url: str) -> bool:
        """Return True if this adapter can process `url`."""
        ...
