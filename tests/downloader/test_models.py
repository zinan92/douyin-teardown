"""Tests for data models — JSON round-trip and field validation."""

import json
import pytest
from datetime import datetime, timezone

from content_downloader.models import ContentItem, DownloadError, DownloadResult


def make_content_item(**overrides) -> ContentItem:
    defaults = dict(
        platform="fixture",
        content_id="abc123",
        content_type="video",
        title="Test Video",
        description="A test video",
        author_id="test-author",
        author_name="Test Author",
        publish_time="2026-03-01T12:00:00Z",
        source_url="https://fixture.test/video/abc123",
        media_files=["media/video.mp4"],
        cover_file="media/cover.jpg",
        metadata_file="metadata.json",
        likes=100,
        comments=10,
        shares=5,
        collects=20,
        views=1000,
        downloaded_at="2026-03-29T09:00:00Z",
    )
    defaults.update(overrides)
    return ContentItem(**defaults)


class TestContentItem:
    def test_create_basic(self):
        item = make_content_item()
        assert item.platform == "fixture"
        assert item.content_id == "abc123"

    def test_json_round_trip(self):
        item = make_content_item()
        serialized = item.model_dump_json()
        restored = ContentItem.model_validate_json(serialized)
        assert restored == item

    def test_dict_round_trip(self):
        item = make_content_item()
        d = item.model_dump()
        restored = ContentItem(**d)
        assert restored == item

    def test_frozen_immutability(self):
        item = make_content_item()
        with pytest.raises(Exception):
            item.title = "Modified"  # type: ignore[misc]

    def test_media_files_default_empty(self):
        item = make_content_item(media_files=[])
        assert item.media_files == []

    def test_cover_file_optional(self):
        item = make_content_item(cover_file=None)
        assert item.cover_file is None

    def test_engagement_defaults_zero(self):
        item = make_content_item(likes=0, comments=0, shares=0, collects=0, views=0)
        assert item.likes == 0
        assert item.views == 0

    def test_multiple_media_files(self):
        item = make_content_item(media_files=["media/img1.jpg", "media/img2.jpg"])
        assert len(item.media_files) == 2


class TestDownloadError:
    def test_create_and_round_trip(self):
        err = DownloadError(
            content_id="xyz",
            source_url="https://fixture.test/video/xyz",
            error_type="not_found",
            message="Content not found",
            retryable=False,
        )
        restored = DownloadError.model_validate_json(err.model_dump_json())
        assert restored == err

    def test_retryable_network_error(self):
        err = DownloadError(
            content_id="xyz",
            source_url="https://fixture.test/video/xyz",
            error_type="network",
            message="Connection timeout",
            retryable=True,
        )
        assert err.retryable is True

    def test_error_types(self):
        for error_type in ("auth", "rate_limit", "not_found", "network", "unsupported"):
            err = DownloadError(
                content_id="x",
                source_url="http://x",
                error_type=error_type,
                message="msg",
                retryable=False,
            )
            assert err.error_type == error_type


class TestDownloadResult:
    def test_empty_result(self):
        result = DownloadResult()
        assert result.items == []
        assert result.errors == []
        assert result.total == 0

    def test_with_items(self):
        item = make_content_item()
        result = DownloadResult(items=[item], total=1, success=1)
        assert len(result.items) == 1
        assert result.success == 1

    def test_json_round_trip(self):
        item = make_content_item()
        err = DownloadError(
            content_id="y",
            source_url="https://x",
            error_type="auth",
            message="Unauthorized",
            retryable=False,
        )
        result = DownloadResult(items=[item], errors=[err], total=2, success=1, failed=1)
        restored = DownloadResult.model_validate_json(result.model_dump_json())
        assert restored == result
