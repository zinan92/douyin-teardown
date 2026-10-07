"""Tests for the manifest manager."""

import json
import threading
import pytest
from pathlib import Path

from content_downloader.manifest import ManifestManager
from content_downloader.models import ContentItem


def make_item(content_id: str = "abc123", platform: str = "fixture", **overrides) -> ContentItem:
    defaults = dict(
        platform=platform,
        content_id=content_id,
        content_type="video",
        title=f"Title {content_id}",
        description="Desc",
        author_id="test-author",
        author_name="Test Author",
        publish_time="2026-03-01T12:00:00Z",
        source_url=f"https://fixture.test/video/{content_id}",
        media_files=["media/video.mp4"],
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
def manager(tmp_output_dir) -> ManifestManager:
    return ManifestManager(tmp_output_dir)


class TestAppend:
    def test_creates_manifest_file(self, manager, tmp_output_dir):
        item = make_item()
        manager.append(item)
        manifest = tmp_output_dir / "manifest.jsonl"
        assert manifest.exists()

    def test_appended_line_is_valid_json(self, manager, tmp_output_dir):
        item = make_item()
        manager.append(item)
        lines = (tmp_output_dir / "manifest.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["content_id"] == "abc123"

    def test_multiple_appends_produce_multiple_lines(self, manager, tmp_output_dir):
        manager.append(make_item("vid001"))
        manager.append(make_item("vid002"))
        manager.append(make_item("vid003"))
        lines = (tmp_output_dir / "manifest.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3

    def test_record_contains_engagement_data(self, manager, tmp_output_dir):
        item = make_item("x1", likes=999, views=12345)
        manager.append(item)
        lines = (tmp_output_dir / "manifest.jsonl").read_text().strip().split("\n")
        data = json.loads(lines[0])
        assert data["likes"] == 999
        assert data["views"] == 12345


class TestContains:
    def test_false_when_empty(self, manager):
        assert manager.contains("abc123") is False

    def test_true_after_append(self, manager):
        item = make_item("abc123")
        manager.append(item)
        assert manager.contains("abc123") is True

    def test_false_for_different_id(self, manager):
        manager.append(make_item("abc123"))
        assert manager.contains("xyz999") is False


class TestAllRecords:
    def test_empty_manifest(self, manager):
        assert manager.all_records() == []

    def test_returns_all_appended(self, manager):
        manager.append(make_item("v1"))
        manager.append(make_item("v2"))
        records = manager.all_records()
        assert len(records) == 2
        ids = {r["content_id"] for r in records}
        assert ids == {"v1", "v2"}


class TestFilterByPlatform:
    def test_filters_correctly(self, manager):
        manager.append(make_item("d1", platform="douyin"))
        manager.append(make_item("f1", platform="fixture"))
        manager.append(make_item("d2", platform="douyin"))

        douyin = manager.filter_by_platform("douyin")
        fixture = manager.filter_by_platform("fixture")
        assert len(douyin) == 2
        assert len(fixture) == 1

    def test_empty_for_missing_platform(self, manager):
        manager.append(make_item("f1", platform="fixture"))
        assert manager.filter_by_platform("xhs") == []


class TestConcurrentAppend:
    def test_concurrent_appends_no_data_loss(self, tmp_output_dir):
        """Multiple threads appending simultaneously should not lose records."""
        manager = ManifestManager(tmp_output_dir)
        n_threads = 20
        errors: list[Exception] = []

        def append_one(i: int) -> None:
            try:
                manager.append(make_item(f"vid-{i:04d}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=append_one, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent append: {errors}"
        records = manager.all_records()
        assert len(records) == n_threads
        ids = {r["content_id"] for r in records}
        assert len(ids) == n_threads  # No duplicates, no lost writes

    def test_jsonl_format_valid_after_concurrent_writes(self, tmp_output_dir):
        manager = ManifestManager(tmp_output_dir)
        n = 10
        threads = [
            threading.Thread(target=lambda i=i: manager.append(make_item(f"t-{i}")), args=())
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        manifest_path = tmp_output_dir / "manifest.jsonl"
        lines = [l for l in manifest_path.read_text().strip().split("\n") if l]
        assert len(lines) == n
        for line in lines:
            data = json.loads(line)  # Should not raise
            assert "content_id" in data
