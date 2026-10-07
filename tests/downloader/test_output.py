"""Tests for the output manager."""

import json
import pytest
from pathlib import Path

from content_downloader.output import OutputManager
from content_downloader.models import ContentItem


def make_item(**overrides) -> ContentItem:
    defaults = dict(
        platform="fixture",
        content_id="abc123",
        content_type="video",
        title="Test Video",
        description="Desc",
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


@pytest.fixture
def manager(tmp_output_dir) -> OutputManager:
    return OutputManager(tmp_output_dir)


class TestOutputManagerPaths:
    def test_content_dir_path(self, manager, tmp_output_dir):
        item = make_item()
        expected = tmp_output_dir / "fixture" / "test-author" / "abc123"
        assert manager.content_dir(item) == expected

    def test_media_dir_path(self, manager, tmp_output_dir):
        item = make_item()
        expected = tmp_output_dir / "fixture" / "test-author" / "abc123" / "media"
        assert manager.media_dir(item) == expected


class TestEnsureDirs:
    def test_creates_content_and_media_dirs(self, manager, tmp_output_dir):
        item = make_item()
        c_dir, m_dir = manager.ensure_dirs(item)
        assert c_dir.is_dir()
        assert m_dir.is_dir()

    def test_idempotent(self, manager, tmp_output_dir):
        item = make_item()
        manager.ensure_dirs(item)
        manager.ensure_dirs(item)  # Should not raise
        assert manager.content_dir(item).is_dir()


class TestWriteMetadata:
    def test_writes_metadata_json(self, manager, tmp_output_dir):
        item = make_item()
        raw = {"platform": "fixture", "id": "abc123", "title": "Test"}
        path = manager.write_metadata(item, raw)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["platform"] == "fixture"

    def test_uses_item_metadata_file_name(self, manager, tmp_output_dir):
        item = make_item(metadata_file="raw.json")
        raw = {"id": "abc123"}
        path = manager.write_metadata(item, raw)
        assert path.name == "raw.json"

    def test_unicode_preserved(self, manager, tmp_output_dir):
        item = make_item()
        raw = {"title": "你好世界 🎬"}
        path = manager.write_metadata(item, raw)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["title"] == "你好世界 🎬"


class TestWriteContentItem:
    def test_writes_content_item_json(self, manager, tmp_output_dir):
        item = make_item()
        path = manager.write_content_item(item)
        assert path.name == "content_item.json"
        assert path.exists()

    def test_json_is_valid_content_item(self, manager, tmp_output_dir):
        item = make_item()
        path = manager.write_content_item(item)
        restored = ContentItem.model_validate_json(path.read_text())
        assert restored == item


class TestWriteAll:
    def test_returns_all_paths(self, manager, tmp_output_dir):
        item = make_item()
        raw = {"id": "abc123"}
        result = manager.write_all(item, raw)
        assert "content_dir" in result
        assert "media_dir" in result
        assert "metadata_path" in result
        assert "item_path" in result

    def test_all_files_exist(self, manager, tmp_output_dir):
        item = make_item()
        raw = {"id": "abc123"}
        result = manager.write_all(item, raw)
        assert result["content_dir"].is_dir()
        assert result["media_dir"].is_dir()
        assert result["metadata_path"].exists()
        assert result["item_path"].exists()

    def test_different_items_get_separate_dirs(self, manager, tmp_output_dir):
        item1 = make_item(content_id="vid001")
        item2 = make_item(content_id="vid002")
        manager.write_all(item1, {"id": "vid001"})
        manager.write_all(item2, {"id": "vid002"})
        assert manager.content_dir(item1) != manager.content_dir(item2)
        assert manager.content_dir(item1).is_dir()
        assert manager.content_dir(item2).is_dir()


class TestExists:
    def test_false_when_not_written(self, manager, tmp_output_dir):
        item = make_item()
        assert manager.exists(item) is False

    def test_true_after_write(self, manager, tmp_output_dir):
        item = make_item()
        manager.write_content_item(item)
        assert manager.exists(item) is True
