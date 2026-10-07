"""Tests for XAdapter — full download flow with mocked XFetcher."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from content_downloader.adapters.x.adapter import XAdapter, XDownloadError
from content_downloader.adapters.x.fetcher import XFetcher
from content_downloader.models import ContentItem, DownloadResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fetcher(is_available: bool = True, info: dict | None = None) -> XFetcher:
    """Create a mock XFetcher."""
    fetcher = AsyncMock(spec=XFetcher)
    fetcher.is_available = AsyncMock(return_value=is_available)
    if info is not None:
        fetcher.fetch_post = AsyncMock(return_value=info)
    else:
        fetcher.fetch_post = AsyncMock(side_effect=FileNotFoundError("no info.json"))
    return fetcher


def _sample_video_info(content_id: str = "1234567890") -> dict:
    return {
        "id": content_id,
        "title": "Test video tweet",
        "description": "Test video tweet",
        "uploader": "Test User",
        "uploader_id": "testuser",
        "timestamp": 1711699200,
        "webpage_url": f"https://x.com/testuser/status/{content_id}",
        "ext": "mp4",
        "formats": [{"vcodec": "h264"}],
        "thumbnails": [],
        "thumbnail": "https://pbs.twimg.com/thumb.jpg",
        "like_count": 10,
        "repost_count": 2,
        "comment_count": 1,
        "view_count": 500,
    }


def _sample_image_info(content_id: str = "9876543210") -> dict:
    return {
        "id": content_id,
        "title": "Test image tweet",
        "description": "Test image tweet",
        "uploader": "Photo User",
        "uploader_id": "photouser",
        "timestamp": 1711785600,
        "webpage_url": f"https://x.com/photouser/status/{content_id}",
        "ext": "jpg",
        "formats": [],
        "thumbnails": [{"url": "https://pbs.twimg.com/img.jpg"}],
        "thumbnail": "https://pbs.twimg.com/img.jpg",
        "like_count": 50,
        "repost_count": 5,
        "comment_count": 3,
        "view_count": 1000,
    }


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


def test_can_handle_x_com_status_url() -> None:
    adapter = XAdapter()
    assert adapter.can_handle("https://x.com/user/status/1234567890") is True


def test_can_handle_twitter_com_status_url() -> None:
    adapter = XAdapter()
    assert adapter.can_handle("https://twitter.com/user/status/1234567890") is True


def test_can_handle_www_prefix() -> None:
    adapter = XAdapter()
    assert adapter.can_handle("https://www.x.com/user/status/1234567890") is True
    assert adapter.can_handle("https://www.twitter.com/user/status/1234567890") is True


def test_can_handle_rejects_profile_url() -> None:
    adapter = XAdapter()
    assert adapter.can_handle("https://x.com/testuser") is False


def test_can_handle_rejects_non_x_url() -> None:
    adapter = XAdapter()
    assert adapter.can_handle("https://example.com/status/123") is False


def test_can_handle_rejects_x_without_status() -> None:
    adapter = XAdapter()
    assert adapter.can_handle("https://x.com/user/") is False


# ---------------------------------------------------------------------------
# download_single — yt-dlp not available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_single_raises_when_ytdlp_unavailable(tmp_path: Path) -> None:
    fetcher = _make_fetcher(is_available=False)
    adapter = XAdapter(fetcher=fetcher)

    with pytest.raises(XDownloadError) as exc_info:
        await adapter.download_single("https://x.com/user/status/123", tmp_path)

    err = exc_info.value.download_error
    assert err.error_type == "service_unavailable"
    assert "yt-dlp" in err.message
    assert "pip install yt-dlp" in err.message


# ---------------------------------------------------------------------------
# download_single — video tweet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_single_video_tweet(tmp_path: Path) -> None:
    info = _sample_video_info()
    fetcher = _make_fetcher(is_available=True, info=info)

    # Simulate yt-dlp creating a video file in the staging area
    def fake_fetch_post(url, output_dir):
        media_dir = output_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / "1234567890.mp4").write_bytes(b"fake video data")
        (media_dir / "1234567890.info.json").write_text(json.dumps(info))
        return info

    fetcher.fetch_post = AsyncMock(side_effect=fake_fetch_post)
    adapter = XAdapter(fetcher=fetcher)

    item = await adapter.download_single(
        "https://x.com/testuser/status/1234567890", tmp_path
    )

    assert isinstance(item, ContentItem)
    assert item.platform == "x"
    assert item.content_id == "1234567890"
    assert item.content_type == "video"
    assert item.author_id == "testuser"
    assert item.likes == 10

    # Check output directory structure
    content_dir = tmp_path / "x" / "testuser" / "1234567890"
    assert (content_dir / "metadata.json").exists()
    assert (content_dir / "content_item.json").exists()


# ---------------------------------------------------------------------------
# download_single — image tweet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_single_image_tweet(tmp_path: Path) -> None:
    info = _sample_image_info()
    fetcher = _make_fetcher(is_available=True, info=info)

    def fake_fetch_post(url, output_dir):
        media_dir = output_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / "9876543210.jpg").write_bytes(b"fake image data")
        (media_dir / "9876543210.info.json").write_text(json.dumps(info))
        return info

    fetcher.fetch_post = AsyncMock(side_effect=fake_fetch_post)
    adapter = XAdapter(fetcher=fetcher)

    item = await adapter.download_single(
        "https://x.com/photouser/status/9876543210", tmp_path
    )

    assert item.content_type == "image"
    assert item.content_id == "9876543210"
    assert item.cover_file is not None


# ---------------------------------------------------------------------------
# download_single — text-only tweet (FileNotFoundError from fetcher)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_single_text_only_tweet(tmp_path: Path) -> None:
    fetcher = _make_fetcher(is_available=True, info=None)
    # FileNotFoundError simulates no media / no info.json from yt-dlp
    fetcher.fetch_post = AsyncMock(side_effect=FileNotFoundError("no info.json"))
    adapter = XAdapter(fetcher=fetcher)

    item = await adapter.download_single(
        "https://x.com/user/status/111222333", tmp_path
    )

    assert item.platform == "x"
    assert item.content_type == "text"
    assert item.media_files == []
    assert item.content_id == "111222333"


# ---------------------------------------------------------------------------
# download_single — yt-dlp runtime error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_single_raises_on_ytdlp_failure(tmp_path: Path) -> None:
    fetcher = _make_fetcher(is_available=True)
    fetcher.fetch_post = AsyncMock(
        side_effect=RuntimeError("yt-dlp exited with code 1.\nstderr: ERROR: Private tweet")
    )
    adapter = XAdapter(fetcher=fetcher)

    with pytest.raises(XDownloadError) as exc_info:
        await adapter.download_single("https://x.com/user/status/999", tmp_path)

    err = exc_info.value.download_error
    assert err.error_type == "network"
    assert err.retryable is True


# ---------------------------------------------------------------------------
# download_single — metadata files written correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_single_writes_metadata_json(tmp_path: Path) -> None:
    info = _sample_video_info("555")
    fetcher = _make_fetcher(is_available=True, info=info)

    def fake_fetch_post(url, output_dir):
        media_dir = output_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / "555.mp4").write_bytes(b"v")
        (media_dir / "555.info.json").write_text(json.dumps(info))
        return info

    fetcher.fetch_post = AsyncMock(side_effect=fake_fetch_post)
    adapter = XAdapter(fetcher=fetcher)
    item = await adapter.download_single("https://x.com/testuser/status/555", tmp_path)

    content_dir = tmp_path / "x" / "testuser" / "555"
    metadata = json.loads((content_dir / "metadata.json").read_text())
    assert metadata["id"] == "555"

    content_item_data = json.loads((content_dir / "content_item.json").read_text())
    assert content_item_data["content_id"] == "555"
    assert content_item_data["platform"] == "x"


# ---------------------------------------------------------------------------
# download_profile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_profile_returns_unsupported_result(tmp_path: Path) -> None:
    adapter = XAdapter()
    result = await adapter.download_profile(
        "https://x.com/testuser", tmp_path
    )

    assert isinstance(result, DownloadResult)
    assert result.total == 0
    assert result.failed == 1
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "unsupported"
    assert "not supported" in result.errors[0].message


@pytest.mark.asyncio
async def test_download_profile_with_limit_still_unsupported(tmp_path: Path) -> None:
    adapter = XAdapter()
    result = await adapter.download_profile(
        "https://x.com/testuser", tmp_path, limit=10
    )
    assert result.errors[0].error_type == "unsupported"
