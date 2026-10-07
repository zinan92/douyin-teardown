"""Fixture adapter — deterministic fake adapter for testing and CI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from content_downloader.models import ContentItem, DownloadError, DownloadResult

# Fixture domain
FIXTURE_DOMAIN = "fixture.test"

# Dummy video bytes — minimal valid-looking binary blob for testing
_DUMMY_VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100
_DUMMY_IMAGE_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xd9"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class FixtureAdapter:
    """Deterministic adapter that creates real directory structures without network access.

    URL patterns:
      - ``https://fixture.test/video/<content_id>``  → single video
      - ``https://fixture.test/image/<content_id>``  → single image
      - ``https://fixture.test/user/<author_id>``    → profile (N items)
    """

    platform = "fixture"

    def can_handle(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.netloc == FIXTURE_DOMAIN
        except Exception:
            return False

    async def download_single(
        self,
        url: str,
        output_dir: Path,
    ) -> ContentItem:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.strip("/").split("/") if p]

        if len(parts) < 2:
            raise ValueError(f"Unrecognised fixture URL: {url}")

        url_type = parts[0]  # "video" | "image"
        content_id = parts[1]

        if url_type == "video":
            return self._make_video(content_id, url, output_dir)
        elif url_type == "image":
            return self._make_image(content_id, url, output_dir)
        else:
            raise ValueError(f"Unsupported fixture content type: {url_type}")

    async def download_profile(
        self,
        profile_url: str,
        output_dir: Path,
        limit: int = 0,
        since: datetime | None = None,
    ) -> DownloadResult:
        parsed = urlparse(profile_url)
        parts = [p for p in parsed.path.strip("/").split("/") if p]

        if len(parts) < 2 or parts[0] != "user":
            raise ValueError(f"Unrecognised fixture profile URL: {profile_url}")

        author_id = parts[1]

        # Default to 5 items if no limit specified
        count = limit if limit > 0 else 5
        items: list[ContentItem] = []
        errors: list[DownloadError] = []

        for i in range(count):
            content_id = f"{author_id}-vid-{i + 1:03d}"
            item_url = f"https://fixture.test/video/{content_id}"
            try:
                item = self._make_video(content_id, item_url, output_dir, author_id=author_id)
                items.append(item)
            except Exception as exc:
                errors.append(
                    DownloadError(
                        content_id=content_id,
                        source_url=item_url,
                        error_type="network",
                        message=str(exc),
                        retryable=True,
                    )
                )

        return DownloadResult(
            items=items,
            errors=errors,
            total=count,
            success=len(items),
            failed=len(errors),
            skipped=0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_video(
        self,
        content_id: str,
        url: str,
        output_dir: Path,
        author_id: str = "test-author",
    ) -> ContentItem:
        content_dir = self._ensure_content_dir(output_dir, author_id, content_id)
        media_dir = content_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        # Write dummy media files
        video_path = media_dir / "video.mp4"
        video_path.write_bytes(_DUMMY_VIDEO_BYTES)

        cover_path = media_dir / "cover.jpg"
        cover_path.write_bytes(_DUMMY_IMAGE_BYTES)

        # Write raw platform metadata
        raw_metadata = {
            "platform": "fixture",
            "id": content_id,
            "type": "video",
            "title": f"Fixture Video {content_id}",
            "author": {"id": author_id, "name": "Test Author"},
            "publish_time": "2026-03-01T12:00:00Z",
            "stats": {"likes": 100, "comments": 10, "shares": 5, "collects": 20, "views": 1000},
        }
        (content_dir / "metadata.json").write_text(
            json.dumps(raw_metadata, ensure_ascii=False, indent=2)
        )

        item = ContentItem(
            platform="fixture",
            content_id=content_id,
            content_type="video",
            title=f"Fixture Video {content_id}",
            description=f"Auto-generated fixture video for {content_id}",
            author_id=author_id,
            author_name="Test Author",
            publish_time="2026-03-01T12:00:00Z",
            source_url=url,
            media_files=["media/video.mp4"],
            cover_file="media/cover.jpg",
            metadata_file="metadata.json",
            likes=100,
            comments=10,
            shares=5,
            collects=20,
            views=1000,
            downloaded_at=_now_iso(),
        )

        # Write content_item.json
        (content_dir / "content_item.json").write_text(
            item.model_dump_json(indent=2)
        )

        return item

    def _make_image(
        self,
        content_id: str,
        url: str,
        output_dir: Path,
        author_id: str = "test-author",
    ) -> ContentItem:
        content_dir = self._ensure_content_dir(output_dir, author_id, content_id)
        media_dir = content_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        image_path = media_dir / "image.jpg"
        image_path.write_bytes(_DUMMY_IMAGE_BYTES)

        cover_path = media_dir / "cover.jpg"
        cover_path.write_bytes(_DUMMY_IMAGE_BYTES)

        raw_metadata = {
            "platform": "fixture",
            "id": content_id,
            "type": "image",
            "title": f"Fixture Image {content_id}",
            "author": {"id": author_id, "name": "Test Author"},
            "publish_time": "2026-03-01T12:00:00Z",
            "stats": {"likes": 50, "comments": 5, "shares": 2, "collects": 10, "views": 500},
        }
        (content_dir / "metadata.json").write_text(
            json.dumps(raw_metadata, ensure_ascii=False, indent=2)
        )

        item = ContentItem(
            platform="fixture",
            content_id=content_id,
            content_type="image",
            title=f"Fixture Image {content_id}",
            description=f"Auto-generated fixture image for {content_id}",
            author_id=author_id,
            author_name="Test Author",
            publish_time="2026-03-01T12:00:00Z",
            source_url=url,
            media_files=["media/image.jpg"],
            cover_file="media/cover.jpg",
            metadata_file="metadata.json",
            likes=50,
            comments=5,
            shares=2,
            collects=10,
            views=500,
            downloaded_at=_now_iso(),
        )

        (content_dir / "content_item.json").write_text(
            item.model_dump_json(indent=2)
        )

        return item

    def _ensure_content_dir(
        self, output_dir: Path, author_id: str, content_id: str
    ) -> Path:
        content_dir = output_dir / "fixture" / author_id / content_id
        content_dir.mkdir(parents=True, exist_ok=True)
        return content_dir
