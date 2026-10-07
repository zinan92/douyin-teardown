"""Tests for Douyin aweme -> ContentItem mapper."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from content_downloader.adapters.douyin.mapper import aweme_to_content_item

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def aweme_detail() -> dict:
    with open(FIXTURES_DIR / "aweme_detail.json") as f:
        data = json.load(f)
    return data["aweme_detail"]


class TestAwemeToContentItem:
    def test_basic_fields(self, aweme_detail: dict) -> None:
        item = aweme_to_content_item(
            aweme_detail,
            source_url="https://www.douyin.com/video/7380308675841297704",
            media_files=["media/video.mp4"],
            cover_file="media/cover.jpg",
        )
        assert item.platform == "douyin"
        assert item.content_id == "7380308675841297704"
        assert item.content_type == "video"
        assert item.author_id == "12345678"
        assert item.author_name == "测试用户"
        assert "media/video.mp4" in item.media_files
        assert item.cover_file == "media/cover.jpg"
        assert item.metadata_file == "metadata.json"

    def test_stats_fields(self, aweme_detail: dict) -> None:
        item = aweme_to_content_item(aweme_detail, source_url="https://test")
        assert item.likes == 1234
        assert item.comments == 56
        assert item.shares == 78
        assert item.collects == 90
        assert item.views == 9999

    def test_title_truncated_to_100_chars(self) -> None:
        aweme = {
            "aweme_id": "111",
            "desc": "A" * 200,
            "create_time": 1700000000,
            "author": {"uid": "1", "nickname": "x"},
            "statistics": {},
        }
        item = aweme_to_content_item(aweme, source_url="https://test")
        assert len(item.title) == 100
        assert len(item.description) == 200

    def test_publish_time_from_create_time(self, aweme_detail: dict) -> None:
        item = aweme_to_content_item(aweme_detail, source_url="https://test")
        expected_dt = datetime.fromtimestamp(1700000000, tz=timezone.utc)
        # Check it parses back to the same datetime
        parsed = datetime.fromisoformat(item.publish_time)
        assert parsed.replace(tzinfo=timezone.utc) == expected_dt.replace(tzinfo=timezone.utc)

    def test_share_url_used_as_source_url(self, aweme_detail: dict) -> None:
        item = aweme_to_content_item(aweme_detail, source_url="https://original")
        # aweme_detail has share_url set
        assert item.source_url == "https://www.douyin.com/video/7380308675841297704"

    def test_fallback_source_url_when_no_share_url(self) -> None:
        aweme = {
            "aweme_id": "222",
            "desc": "no share url",
            "create_time": 1700000000,
            "author": {"uid": "2", "nickname": "y"},
            "statistics": {},
        }
        item = aweme_to_content_item(aweme, source_url="https://fallback-url")
        assert item.source_url == "https://fallback-url"

    def test_gallery_content_type(self) -> None:
        aweme = {
            "aweme_id": "333",
            "desc": "图集",
            "create_time": 1700000000,
            "author": {"uid": "3", "nickname": "z"},
            "statistics": {},
            "images": [
                {"url_list": ["https://example.com/img1.jpg"]},
                {"url_list": ["https://example.com/img2.jpg"]},
            ],
        }
        item = aweme_to_content_item(aweme, source_url="https://test")
        assert item.content_type == "gallery"

    def test_empty_stats_default_zero(self) -> None:
        aweme = {
            "aweme_id": "444",
            "desc": "",
            "create_time": 1700000000,
            "author": {"uid": "4", "nickname": "w"},
            "statistics": {},
        }
        item = aweme_to_content_item(aweme, source_url="https://test")
        assert item.likes == 0
        assert item.comments == 0
        assert item.shares == 0
        assert item.collects == 0
        assert item.views == 0

    def test_downloaded_at_is_set(self, aweme_detail: dict) -> None:
        item = aweme_to_content_item(aweme_detail, source_url="https://test")
        assert item.downloaded_at
        # Should be a valid ISO 8601 string
        datetime.fromisoformat(item.downloaded_at.replace("Z", "+00:00"))
