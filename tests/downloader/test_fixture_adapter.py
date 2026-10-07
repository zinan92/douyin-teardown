"""Tests for the fixture adapter."""

import json
import pytest
from pathlib import Path

from content_downloader.adapters.fixture import FixtureAdapter
from content_downloader.models import ContentItem


@pytest.fixture
def adapter() -> FixtureAdapter:
    return FixtureAdapter()


class TestCanHandle:
    def test_video_url(self, adapter):
        assert adapter.can_handle("https://fixture.test/video/abc123") is True

    def test_user_url(self, adapter):
        assert adapter.can_handle("https://fixture.test/user/test-author") is True

    def test_image_url(self, adapter):
        assert adapter.can_handle("https://fixture.test/image/img01") is True

    def test_non_fixture_url(self, adapter):
        assert adapter.can_handle("https://douyin.com/video/12345") is False

    def test_empty_string(self, adapter):
        assert adapter.can_handle("") is False


class TestDownloadSingle:
    async def test_video_creates_directory_structure(self, adapter, tmp_output_dir):
        item = await adapter.download_single(
            "https://fixture.test/video/abc123", tmp_output_dir
        )

        content_dir = tmp_output_dir / "fixture" / "test-author" / "abc123"
        assert content_dir.is_dir()
        assert (content_dir / "media").is_dir()
        assert (content_dir / "media" / "video.mp4").exists()
        assert (content_dir / "media" / "cover.jpg").exists()
        assert (content_dir / "metadata.json").exists()
        assert (content_dir / "content_item.json").exists()

    async def test_video_returns_content_item(self, adapter, tmp_output_dir):
        item = await adapter.download_single(
            "https://fixture.test/video/abc123", tmp_output_dir
        )

        assert isinstance(item, ContentItem)
        assert item.platform == "fixture"
        assert item.content_id == "abc123"
        assert item.content_type == "video"
        assert item.author_id == "test-author"
        assert item.media_files == ["media/video.mp4"]
        assert item.cover_file == "media/cover.jpg"

    async def test_video_metadata_json_valid(self, adapter, tmp_output_dir):
        await adapter.download_single(
            "https://fixture.test/video/abc123", tmp_output_dir
        )
        metadata_path = tmp_output_dir / "fixture" / "test-author" / "abc123" / "metadata.json"
        data = json.loads(metadata_path.read_text())
        assert data["platform"] == "fixture"
        assert data["id"] == "abc123"

    async def test_video_content_item_json_valid(self, adapter, tmp_output_dir):
        await adapter.download_single(
            "https://fixture.test/video/abc123", tmp_output_dir
        )
        item_path = (
            tmp_output_dir / "fixture" / "test-author" / "abc123" / "content_item.json"
        )
        data = json.loads(item_path.read_text())
        restored = ContentItem(**data)
        assert restored.content_id == "abc123"

    async def test_image_creates_directory_structure(self, adapter, tmp_output_dir):
        item = await adapter.download_single(
            "https://fixture.test/image/img01", tmp_output_dir
        )

        content_dir = tmp_output_dir / "fixture" / "test-author" / "img01"
        assert content_dir.is_dir()
        assert (content_dir / "media" / "image.jpg").exists()
        assert item.content_type == "image"

    async def test_invalid_url_raises(self, adapter, tmp_output_dir):
        with pytest.raises(ValueError, match="Unrecognised fixture URL"):
            await adapter.download_single("https://fixture.test/", tmp_output_dir)

    async def test_unsupported_type_raises(self, adapter, tmp_output_dir):
        with pytest.raises(ValueError, match="Unsupported fixture content type"):
            await adapter.download_single(
                "https://fixture.test/reel/abc", tmp_output_dir
            )

    async def test_deterministic_content_id(self, adapter, tmp_output_dir):
        """Two calls with the same URL produce the same content_id."""
        item1 = await adapter.download_single(
            "https://fixture.test/video/stable", tmp_output_dir
        )
        item2 = await adapter.download_single(
            "https://fixture.test/video/stable", tmp_output_dir
        )
        assert item1.content_id == item2.content_id


class TestDownloadProfile:
    async def test_respects_limit(self, adapter, tmp_output_dir):
        result = await adapter.download_profile(
            "https://fixture.test/user/alice", tmp_output_dir, limit=3
        )
        assert result.success == 3
        assert result.total == 3
        assert len(result.items) == 3

    async def test_default_limit_is_five(self, adapter, tmp_output_dir):
        result = await adapter.download_profile(
            "https://fixture.test/user/bob", tmp_output_dir, limit=0
        )
        assert result.total == 5

    async def test_each_item_has_unique_content_id(self, adapter, tmp_output_dir):
        result = await adapter.download_profile(
            "https://fixture.test/user/carol", tmp_output_dir, limit=3
        )
        ids = [item.content_id for item in result.items]
        assert len(set(ids)) == 3

    async def test_creates_directories_per_item(self, adapter, tmp_output_dir):
        result = await adapter.download_profile(
            "https://fixture.test/user/dave", tmp_output_dir, limit=2
        )
        for item in result.items:
            content_dir = tmp_output_dir / "fixture" / item.author_id / item.content_id
            assert content_dir.is_dir()
            assert (content_dir / "media" / "video.mp4").exists()

    async def test_invalid_profile_url_raises(self, adapter, tmp_output_dir):
        with pytest.raises(ValueError, match="Unrecognised fixture profile URL"):
            await adapter.download_profile(
                "https://fixture.test/video/abc", tmp_output_dir
            )

    async def test_result_counts_consistent(self, adapter, tmp_output_dir):
        result = await adapter.download_profile(
            "https://fixture.test/user/eve", tmp_output_dir, limit=4
        )
        assert result.success + result.failed + result.skipped == result.total
