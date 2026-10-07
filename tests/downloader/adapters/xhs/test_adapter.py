"""Unit tests for XHSAdapter — download_single and download_profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from content_downloader.adapters.xhs.adapter import XHSAdapter, XHSDownloadError
from content_downloader.models import DownloadError, DownloadResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_FAKE_IMAGE_BYTES = b"\xff\xd8\xff\xe0fake_image_data"
_FAKE_VIDEO_BYTES = b"\x00\x00\x00\x18ftyp_fake_video"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers — patch XHSAPIClient
# ---------------------------------------------------------------------------


def _make_api_client_mock(
    note_data: dict | None = None,
    is_available: bool = True,
    download_bytes: bytes = _FAKE_IMAGE_BYTES,
) -> MagicMock:
    """Build a mock XHSAPIClient context manager."""
    mock_instance = AsyncMock()
    mock_instance.is_available = AsyncMock(return_value=is_available)
    mock_instance.get_note_detail = AsyncMock(return_value=note_data or {})

    # Make it work as async context manager
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)

    return mock_instance


def _make_httpx_client_mock(download_bytes: bytes = _FAKE_IMAGE_BYTES) -> MagicMock:
    """Build a mock httpx.AsyncClient for media downloads."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.content = download_bytes
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    return mock_http


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


def test_can_handle_explore_url():
    adapter = XHSAdapter()
    assert adapter.can_handle("https://www.xiaohongshu.com/explore/abc123") is True


def test_can_handle_discovery_url():
    adapter = XHSAdapter()
    assert adapter.can_handle("https://www.xiaohongshu.com/discovery/item/abc123") is True


def test_can_handle_xhslink_url():
    adapter = XHSAdapter()
    assert adapter.can_handle("https://xhslink.com/aBcDef") is True


def test_can_handle_profile_url():
    adapter = XHSAdapter()
    assert adapter.can_handle("https://www.xiaohongshu.com/user/profile/user123") is True


def test_can_handle_rejects_unknown_url():
    adapter = XHSAdapter()
    assert adapter.can_handle("https://www.douyin.com/video/123") is False
    assert adapter.can_handle("https://twitter.com/user/status/123") is False
    assert adapter.can_handle("https://www.youtube.com/watch?v=abc") is False


# ---------------------------------------------------------------------------
# download_single — gallery note
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_single_gallery(tmp_path: Path):
    """download_single creates img files + metadata + content_item for gallery note."""
    fixture = _load_fixture("note_gallery.json")
    mock_client = _make_api_client_mock(note_data=fixture, is_available=True)
    mock_http = _make_httpx_client_mock(_FAKE_IMAGE_BYTES)

    with patch("content_downloader.adapters.xhs.adapter.XHSAPIClient", return_value=mock_client), \
         patch("content_downloader.adapters.xhs.adapter.httpx.AsyncClient", return_value=mock_http):
        adapter = XHSAdapter()
        item = await adapter.download_single(
            "https://www.xiaohongshu.com/explore/abc123gallery",
            tmp_path,
        )

    assert item.platform == "xhs"
    assert item.content_id == "abc123gallery"
    assert item.content_type == "gallery"
    assert item.author_id == "user456"
    assert item.likes == 1024

    content_dir = tmp_path / "xhs" / "user456" / "abc123gallery"
    assert (content_dir / "metadata.json").exists()
    assert (content_dir / "content_item.json").exists()

    saved = json.loads((content_dir / "content_item.json").read_text())
    assert saved["platform"] == "xhs"
    assert saved["content_id"] == "abc123gallery"


@pytest.mark.asyncio
async def test_download_single_gallery_downloads_images(tmp_path: Path):
    """download_single downloads all images from 下载地址 list."""
    fixture = _load_fixture("note_gallery.json")
    mock_client = _make_api_client_mock(note_data=fixture, is_available=True)
    mock_http = _make_httpx_client_mock(_FAKE_IMAGE_BYTES)

    with patch("content_downloader.adapters.xhs.adapter.XHSAPIClient", return_value=mock_client), \
         patch("content_downloader.adapters.xhs.adapter.httpx.AsyncClient", return_value=mock_http):
        adapter = XHSAdapter()
        item = await adapter.download_single(
            "https://www.xiaohongshu.com/explore/abc123gallery",
            tmp_path,
        )

    # fixture has 3 images in 下载地址
    assert len(item.media_files) == 3


# ---------------------------------------------------------------------------
# download_single — video note
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_single_video(tmp_path: Path):
    """download_single creates video.mp4 + metadata + content_item for video note."""
    fixture = _load_fixture("note_video.json")
    mock_client = _make_api_client_mock(note_data=fixture, is_available=True)
    mock_http = _make_httpx_client_mock(_FAKE_VIDEO_BYTES)

    with patch("content_downloader.adapters.xhs.adapter.XHSAPIClient", return_value=mock_client), \
         patch("content_downloader.adapters.xhs.adapter.httpx.AsyncClient", return_value=mock_http):
        adapter = XHSAdapter()
        item = await adapter.download_single(
            "https://www.xiaohongshu.com/explore/xyz789video",
            tmp_path,
        )

    assert item.platform == "xhs"
    assert item.content_id == "xyz789video"
    assert item.content_type == "video"
    assert item.likes == 5000

    content_dir = tmp_path / "xhs" / "fitnessguru99" / "xyz789video"
    assert (content_dir / "metadata.json").exists()
    assert (content_dir / "content_item.json").exists()
    assert "media/video.mp4" in item.media_files


@pytest.mark.asyncio
async def test_download_single_video_file_written(tmp_path: Path):
    """download_single writes actual bytes for video.mp4."""
    fixture = _load_fixture("note_video.json")
    mock_client = _make_api_client_mock(note_data=fixture, is_available=True)
    mock_http = _make_httpx_client_mock(_FAKE_VIDEO_BYTES)

    with patch("content_downloader.adapters.xhs.adapter.XHSAPIClient", return_value=mock_client), \
         patch("content_downloader.adapters.xhs.adapter.httpx.AsyncClient", return_value=mock_http):
        adapter = XHSAdapter()
        await adapter.download_single(
            "https://www.xiaohongshu.com/explore/xyz789video",
            tmp_path,
        )

    video_path = tmp_path / "xhs" / "fitnessguru99" / "xyz789video" / "media" / "video.mp4"
    assert video_path.exists()
    assert video_path.read_bytes() == _FAKE_VIDEO_BYTES


# ---------------------------------------------------------------------------
# download_single — error cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_single_sidecar_unavailable_raises(tmp_path: Path):
    """download_single raises DownloadError when sidecar cannot be started."""
    mock_client = _make_api_client_mock(is_available=False)

    with patch(
        "content_downloader.adapters.xhs.adapter.XHSAPIClient",
        return_value=mock_client,
    ):
        adapter = XHSAdapter()
        # Mock sidecar auto-start also failing
        with patch.object(
            adapter._sidecar, "ensure_running", new_callable=AsyncMock, return_value=False,
        ):
            with pytest.raises(XHSDownloadError) as exc_info:
                await adapter.download_single(
                    "https://www.xiaohongshu.com/explore/abc",
                    tmp_path,
                )

    err = exc_info.value.download_error
    assert err.error_type == "service_unavailable"
    assert "XHS-Downloader" in err.message
    assert err.retryable is False


@pytest.mark.asyncio
async def test_download_single_empty_response_raises(tmp_path: Path):
    """download_single raises DownloadError(not_found) on empty API response."""
    mock_client = _make_api_client_mock(note_data={}, is_available=True)

    with patch(
        "content_downloader.adapters.xhs.adapter.XHSAPIClient",
        return_value=mock_client,
    ):
        adapter = XHSAdapter()
        with pytest.raises(XHSDownloadError) as exc_info:
            await adapter.download_single(
                "https://www.xiaohongshu.com/explore/missing",
                tmp_path,
            )

    assert exc_info.value.download_error.error_type == "not_found"


# ---------------------------------------------------------------------------
# download_profile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_profile_returns_unsupported_error(tmp_path: Path):
    """download_profile returns DownloadResult with unsupported DownloadError."""
    adapter = XHSAdapter()
    result = await adapter.download_profile(
        "https://www.xiaohongshu.com/user/profile/user123",
        tmp_path,
    )

    assert isinstance(result, DownloadResult)
    assert result.success == 0
    assert result.failed == 1
    assert len(result.errors) == 1

    err = result.errors[0]
    assert err.error_type == "unsupported"
    assert "python main.py" in err.message or "creator" in err.message
    assert err.retryable is False


@pytest.mark.asyncio
async def test_download_profile_extracts_user_id(tmp_path: Path):
    """download_profile includes user_id in the error content_id."""
    adapter = XHSAdapter()
    result = await adapter.download_profile(
        "https://www.xiaohongshu.com/user/profile/abc99xyz",
        tmp_path,
    )
    assert result.errors[0].content_id == "abc99xyz"


@pytest.mark.asyncio
async def test_download_profile_without_valid_profile_url(tmp_path: Path):
    """download_profile handles non-profile URL gracefully."""
    adapter = XHSAdapter()
    result = await adapter.download_profile(
        "https://example.com/unknown",
        tmp_path,
    )
    assert result.failed == 1
    assert result.errors[0].error_type == "unsupported"
