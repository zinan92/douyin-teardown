"""Manifest manager — append-only JSONL index of downloaded content."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock

from content_downloader.models import ContentItem

_MANIFEST_FILENAME = "manifest.jsonl"
_LOCK_SUFFIX = ".lock"


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / _MANIFEST_FILENAME


def _lock_path(output_dir: Path) -> Path:
    return output_dir / (_MANIFEST_FILENAME + _LOCK_SUFFIX)


class ManifestManager:
    """Thread-safe, file-locked JSONL manifest for downloaded content.

    The manifest lives at ``{output_dir}/manifest.jsonl``.
    Each line is a JSON object containing a summary of a ContentItem.

    The manager is safe for concurrent use from multiple processes because
    all writes acquire a file lock before appending.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._manifest = _manifest_path(output_dir)
        self._lock = FileLock(str(_lock_path(output_dir)))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, item: ContentItem) -> None:
        """Append a ContentItem summary line to the manifest.

        Acquires a file lock before writing so concurrent callers are safe.

        Args:
            item: The ContentItem to record.
        """
        record = _item_to_record(item)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self._manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def contains(self, content_id: str) -> bool:
        """Return True if the manifest already has an entry for ``content_id``."""
        for record in self._iter_records():
            if record.get("content_id") == content_id:
                return True
        return False

    def all_records(self) -> list[dict[str, Any]]:
        """Return all manifest records as a list of dicts."""
        return list(self._iter_records())

    def filter_by_platform(self, platform: str) -> list[dict[str, Any]]:
        """Return all manifest records for a given platform."""
        return [r for r in self._iter_records() if r.get("platform") == platform]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield parsed manifest records, skipping malformed lines."""
        if not self._manifest.exists():
            return
        with self._manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Skip corrupt lines rather than crashing
                    continue


def _item_to_record(item: ContentItem) -> dict[str, Any]:
    """Convert a ContentItem into a compact manifest record."""
    return {
        "content_id": item.content_id,
        "platform": item.platform,
        "content_type": item.content_type,
        "title": item.title,
        "author_id": item.author_id,
        "author_name": item.author_name,
        "source_url": item.source_url,
        "publish_time": item.publish_time,
        "downloaded_at": item.downloaded_at,
        "likes": item.likes,
        "comments": item.comments,
        "shares": item.shares,
        "collects": item.collects,
        "views": item.views,
    }
