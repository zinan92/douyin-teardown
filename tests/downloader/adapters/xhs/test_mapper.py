"""Unit tests for XHS note → ContentItem mapper (Chinese field names)."""

from __future__ import annotations

import pytest

from content_downloader.adapters.xhs.mapper import (
    _parse_count,
    extract_download_urls,
    extract_note_id,
    extract_author_id,
    note_to_content_item,
)


# ---------------------------------------------------------------------------
# _parse_count — handles "3.8万", "1220", "3.1万", etc.
# ---------------------------------------------------------------------------


class TestParseCount:
    def test_plain_int(self):
        assert _parse_count("1220") == 1220

    def test_wan_suffix(self):
        assert _parse_count("3.8万") == 38000

    def test_wan_integer(self):
        assert _parse_count("1万") == 10000

    def test_yi_suffix(self):
        assert _parse_count("1.2亿") == 120000000

    def test_int_input(self):
        assert _parse_count(1220) == 1220

    def test_none(self):
        assert _parse_count(None) == 0

    def test_empty_string(self):
        assert _parse_count("") == 0


# ---------------------------------------------------------------------------
# extract helpers
# ---------------------------------------------------------------------------


def _make_response(**data_fields) -> dict:
    return {"message": "ok", "data": data_fields}


class TestExtractors:
    def test_extract_note_id(self):
        resp = _make_response(**{"作品ID": "abc123"})
        assert extract_note_id(resp) == "abc123"

    def test_extract_note_id_fallback(self):
        resp = _make_response(note_id="fallback")
        assert extract_note_id(resp) == "fallback"

    def test_extract_author_id(self):
        resp = _make_response(**{"作者ID": "user456"})
        assert extract_author_id(resp) == "user456"

    def test_extract_download_urls_video(self):
        resp = _make_response(**{"下载地址": ["http://example.com/video.mp4"]})
        assert extract_download_urls(resp) == ["http://example.com/video.mp4"]

    def test_extract_download_urls_empty(self):
        resp = _make_response()
        assert extract_download_urls(resp) == []

    def test_extract_download_urls_filters_none(self):
        resp = _make_response(**{"下载地址": [None, "http://a.com/b.jpg", None]})
        assert extract_download_urls(resp) == ["http://a.com/b.jpg"]


# ---------------------------------------------------------------------------
# note_to_content_item — full mapping
# ---------------------------------------------------------------------------


def _make_video_response() -> dict:
    return {
        "message": "获取小红书作品数据成功",
        "data": {
            "作品ID": "68f60f0000000000040230eb",
            "作品标题": "运营干货全公开",
            "作品描述": "#小红书官方",
            "作品类型": "视频",
            "作者ID": "63fc8880000000001f0329db",
            "作者昵称": "dontbesilent",
            "作者链接": "https://www.xiaohongshu.com/user/profile/63fc8880000000001f0329db",
            "作品链接": "https://www.xiaohongshu.com/explore/68f60f0000000000040230eb",
            "发布时间": "2025-10-20_18:29:20",
            "时间戳": 1760956160.0,
            "点赞数量": "3.8万",
            "评论数量": "1220",
            "分享数量": "1万",
            "收藏数量": "3.1万",
            "下载地址": ["http://example.com/video.mp4"],
            "动图地址": [None],
        },
    }


class TestNoteToContentItem:
    def test_video_note_basic_fields(self):
        item = note_to_content_item(
            _make_video_response(),
            source_url="https://xhs.com/test",
        )
        assert item.platform == "xhs"
        assert item.content_id == "68f60f0000000000040230eb"
        assert item.content_type == "video"
        assert item.title == "运营干货全公开"
        assert item.author_id == "63fc8880000000001f0329db"
        assert item.author_name == "dontbesilent"

    def test_video_note_engagement(self):
        item = note_to_content_item(
            _make_video_response(),
            source_url="https://xhs.com/test",
        )
        assert item.likes == 38000
        assert item.comments == 1220
        assert item.shares == 10000
        assert item.collects == 31000

    def test_video_note_publish_time(self):
        item = note_to_content_item(
            _make_video_response(),
            source_url="https://xhs.com/test",
        )
        assert "2025" in item.publish_time

    def test_video_note_source_url(self):
        item = note_to_content_item(
            _make_video_response(),
            source_url="https://xhs.com/test",
        )
        assert "xiaohongshu.com" in item.source_url

    def test_media_files_passed_through(self):
        item = note_to_content_item(
            _make_video_response(),
            source_url="https://xhs.com/test",
            media_files=["media/video.mp4"],
            cover_file="media/cover.jpg",
        )
        assert item.media_files == ["media/video.mp4"]
        assert item.cover_file == "media/cover.jpg"

    def test_gallery_note(self):
        resp = {
            "message": "ok",
            "data": {
                "作品ID": "img123",
                "作品类型": "图文",
                "作者ID": "user1",
                "作者昵称": "test",
            },
        }
        item = note_to_content_item(resp, source_url="https://xhs.com/img")
        assert item.content_type == "gallery"

    def test_fallback_to_source_url(self):
        resp = {"data": {"作品ID": "x"}}
        item = note_to_content_item(resp, source_url="https://fallback.com")
        assert item.source_url == "https://fallback.com"

    def test_missing_fields_default_gracefully(self):
        resp = {"data": {}}
        item = note_to_content_item(resp, source_url="https://x.com")
        assert item.content_id == ""
        assert item.likes == 0
        assert item.title == ""

    def test_downloaded_at_is_iso(self):
        item = note_to_content_item(
            _make_video_response(),
            source_url="https://xhs.com/test",
        )
        assert "T" in item.downloaded_at
