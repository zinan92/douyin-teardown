"""Tests for X adapter mapper — info.json dict to ContentItem."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from content_downloader.adapters.x.mapper import (
    _detect_content_type,
    _parse_timestamp,
    _safe_int,
    info_to_content_item,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------


def test_parse_timestamp_unix_seconds() -> None:
    # 2024-03-29 00:00:00 UTC
    result = _parse_timestamp(1711670400)
    assert "2024-03-29" in result


def test_parse_timestamp_none_returns_now() -> None:
    result = _parse_timestamp(None)
    assert result.endswith("Z") or "+" in result


def test_parse_timestamp_invalid_string_returns_now() -> None:
    result = _parse_timestamp("not-a-timestamp")
    # Should not raise, should return a valid ISO string
    assert len(result) > 0


# ---------------------------------------------------------------------------
# _safe_int
# ---------------------------------------------------------------------------


def test_safe_int_converts_int() -> None:
    assert _safe_int(42) == 42


def test_safe_int_converts_string() -> None:
    assert _safe_int("100") == 100


def test_safe_int_returns_default_on_none() -> None:
    assert _safe_int(None) == 0


def test_safe_int_returns_default_on_non_numeric() -> None:
    assert _safe_int("abc", default=5) == 5


# ---------------------------------------------------------------------------
# _detect_content_type
# ---------------------------------------------------------------------------


def test_detect_content_type_video() -> None:
    info = {"ext": "mp4", "formats": [{"vcodec": "h264"}]}
    assert _detect_content_type(info) == "video"


def test_detect_content_type_image_via_thumbnail() -> None:
    info = {
        "ext": "jpg",
        "formats": [],
        "thumbnails": [{"url": "https://pbs.twimg.com/img.jpg"}],
        "thumbnail": "https://pbs.twimg.com/img.jpg",
    }
    assert _detect_content_type(info) == "image"


def test_detect_content_type_text_no_media() -> None:
    info = {"ext": "na", "formats": [], "thumbnails": []}
    assert _detect_content_type(info) == "text"


def test_detect_content_type_webm_is_video() -> None:
    info = {"ext": "webm", "formats": [{"vcodec": "vp9"}]}
    assert _detect_content_type(info) == "video"


# ---------------------------------------------------------------------------
# info_to_content_item — video fixture
# ---------------------------------------------------------------------------


def test_info_to_content_item_video_fixture() -> None:
    info = load_fixture("sample_info.json")
    url = "https://x.com/testuser/status/1234567890"
    item = info_to_content_item(
        info,
        source_url=url,
        media_files=["media/1234567890.mp4"],
        cover_file="media/thumb.jpg",
    )

    assert item.platform == "x"
    assert item.content_id == "1234567890"
    assert item.author_id == "testuser"
    assert item.author_name == "Test User"
    assert item.likes == 42
    assert item.shares == 7
    assert item.comments == 3
    assert item.views == 1500
    assert item.media_files == ["media/1234567890.mp4"]
    assert item.cover_file == "media/thumb.jpg"
    assert item.metadata_file == "metadata.json"
    assert "2024" in item.publish_time


def test_info_to_content_item_title_truncated_to_100_chars() -> None:
    long_title = "A" * 200
    info = {"id": "1", "title": long_title, "description": long_title}
    item = info_to_content_item(info, source_url="https://x.com/u/status/1")
    assert len(item.title) == 100


def test_info_to_content_item_missing_fields_use_defaults() -> None:
    info = {"id": "42"}
    item = info_to_content_item(info, source_url="https://x.com/u/status/42")

    assert item.platform == "x"
    assert item.content_id == "42"
    assert item.title == ""
    assert item.description == ""
    assert item.author_id == ""
    assert item.author_name == ""
    assert item.likes == 0
    assert item.shares == 0
    assert item.comments == 0
    assert item.views == 0
    assert item.collects == 0


def test_info_to_content_item_image_fixture() -> None:
    info = load_fixture("sample_image_info.json")
    url = "https://x.com/photouser/status/9876543210"
    item = info_to_content_item(
        info,
        source_url=url,
        media_files=[],
        cover_file="media/image123.jpg",
    )

    assert item.content_id == "9876543210"
    assert item.author_id == "photouser"
    assert item.likes == 100
    assert item.cover_file == "media/image123.jpg"
    assert item.media_files == []


def test_info_to_content_item_source_url_fallback() -> None:
    info = {"id": "99"}
    fallback_url = "https://x.com/u/status/99"
    item = info_to_content_item(info, source_url=fallback_url)
    assert item.source_url == fallback_url


def test_info_to_content_item_uses_webpage_url_over_source() -> None:
    canonical = "https://x.com/canonical/status/100"
    info = {"id": "100", "webpage_url": canonical}
    item = info_to_content_item(info, source_url="https://twitter.com/canonical/status/100")
    assert item.source_url == canonical
