"""Output manager — creates standardized directory structure for downloaded content."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from content_downloader.models import ContentItem


class OutputManager:
    """Manages writing ContentItem data to the standardized output directory structure.

    Directory layout::

        {output_dir}/{platform}/{author_id}/{content_id}/
            media/          ← media files (created by adapters)
            metadata.json   ← raw platform response
            content_item.json ← standardized ContentItem
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def content_dir(self, item: ContentItem) -> Path:
        """Return the canonical content directory path for a ContentItem."""
        return self._output_dir / item.platform / item.author_id / item.content_id

    def media_dir(self, item: ContentItem) -> Path:
        """Return the media subdirectory path for a ContentItem."""
        return self.content_dir(item) / "media"

    def ensure_dirs(self, item: ContentItem) -> tuple[Path, Path]:
        """Create content and media directories if they don't exist.

        Returns:
            Tuple of (content_dir, media_dir).
        """
        c_dir = self.content_dir(item)
        m_dir = self.media_dir(item)
        c_dir.mkdir(parents=True, exist_ok=True)
        m_dir.mkdir(parents=True, exist_ok=True)
        return c_dir, m_dir

    def write_metadata(self, item: ContentItem, raw_metadata: dict[str, Any]) -> Path:
        """Write raw platform metadata JSON to ``metadata.json``.

        Args:
            item: The ContentItem (used to resolve directory).
            raw_metadata: Arbitrary dict of platform-native data.

        Returns:
            Path to the written file.
        """
        c_dir, _ = self.ensure_dirs(item)
        metadata_path = c_dir / item.metadata_file
        metadata_path.write_text(
            json.dumps(raw_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata_path

    def write_content_item(self, item: ContentItem) -> Path:
        """Write the standardized ContentItem to ``content_item.json``.

        Args:
            item: The ContentItem to persist.

        Returns:
            Path to the written file.
        """
        c_dir, _ = self.ensure_dirs(item)
        item_path = c_dir / "content_item.json"
        item_path.write_text(item.model_dump_json(indent=2), encoding="utf-8")
        return item_path

    def write_all(
        self, item: ContentItem, raw_metadata: dict[str, Any]
    ) -> dict[str, Path]:
        """Write both metadata.json and content_item.json.

        Args:
            item: The ContentItem to persist.
            raw_metadata: Platform-native metadata dict.

        Returns:
            Dict with keys ``'content_dir'``, ``'media_dir'``,
            ``'metadata_path'``, ``'item_path'``.
        """
        c_dir, m_dir = self.ensure_dirs(item)
        metadata_path = self.write_metadata(item, raw_metadata)
        item_path = self.write_content_item(item)
        return {
            "content_dir": c_dir,
            "media_dir": m_dir,
            "metadata_path": metadata_path,
            "item_path": item_path,
        }

    def exists(self, item: ContentItem) -> bool:
        """Return True if content_item.json already exists for this item."""
        return (self.content_dir(item) / "content_item.json").exists()
