"""Tests for short URL resolution in DouyinAdapter and API client."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from content_downloader.adapters.douyin.adapter import DouyinAdapter, _extract_aweme_id
from content_downloader.adapters.douyin.api_client import DouyinAPIClient


@pytest.fixture(autouse=True)
def _patch_download_file():
    """Keep short-link tests offline while exercising adapter orchestration."""

    async def _fake_download(_client, _url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"test-media")

    with patch(
        "content_downloader.adapters.douyin.adapter._download_file",
        side_effect=_fake_download,
    ):
        yield


class TestExtractAwemeId:
    """Unit tests for URL parsing helper."""

    def test_extracts_from_standard_video_url(self) -> None:
        url = "https://www.douyin.com/video/7380308675841297704"
        assert _extract_aweme_id(url) == "7380308675841297704"

    def test_extracts_from_modal_url(self) -> None:
        url = "https://www.douyin.com/explore?modal_id=7380308675841297704"
        assert _extract_aweme_id(url) == "7380308675841297704"

    def test_returns_none_for_user_url(self) -> None:
        url = "https://www.douyin.com/user/MS4wLjABAAAAtest"
        assert _extract_aweme_id(url) is None

    def test_returns_none_for_short_url(self) -> None:
        url = "https://v.douyin.com/abc123"
        assert _extract_aweme_id(url) is None


class TestShortLinkResolution:
    """Integration tests for short URL flow in download_single."""

    @pytest.mark.asyncio
    async def test_short_url_resolved_before_download(self, tmp_path: Path) -> None:
        aweme_data = {
            "aweme_id": "7380308675841297704",
            "desc": "resolved video",
            "create_time": 1700000000,
            "author": {"uid": "12345678", "nickname": "测试"},
            "statistics": {"digg_count": 0, "comment_count": 0, "share_count": 0,
                           "collect_count": 0, "play_count": 0},
            "share_url": "https://www.douyin.com/video/7380308675841297704",
            "video": {
                "play_addr": {
                    "url_list": [
                        "https://v19-webapp.douyin.com/video?watermark=0"
                    ]
                },
                "cover": {"url_list": ["https://p3.douyinpic.com/cover.jpg"]},
            },
        }

        adapter = DouyinAdapter(cookies={"msToken": "fake"})
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.headers = {}
        mock_client.cookies = {}

        # Short URL resolves to canonical video URL
        resolved = "https://www.douyin.com/video/7380308675841297704"
        mock_client.resolve_short_url = AsyncMock(return_value=resolved)
        mock_client.get_video_detail = AsyncMock(return_value=aweme_data)
        mock_client.sign_url = MagicMock(
            return_value=("https://v19-webapp.douyin.com/video?watermark=0&X-Bogus=x", "UA")
        )
        mock_client.build_signed_path = MagicMock(
            return_value=("https://www.douyin.com/aweme/v1/play/?X-Bogus=x", "UA")
        )

        mock_http = AsyncMock()
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.content = b"video_bytes"
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_client._client = mock_http

        with patch.object(adapter, "_make_client", return_value=mock_client):
            item = await adapter.download_single(
                "https://v.douyin.com/abc123",
                tmp_path,
            )

        # resolve_short_url MUST have been called
        mock_client.resolve_short_url.assert_awaited_once_with("https://v.douyin.com/abc123")

        assert item.content_id == "7380308675841297704"
        assert item.platform == "douyin"

    @pytest.mark.asyncio
    async def test_short_url_resolution_failure_uses_original(
        self, tmp_path: Path
    ) -> None:
        """If resolve_short_url returns None, we fall back to the original URL."""
        adapter = DouyinAdapter()
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.headers = {}
        mock_client.resolve_short_url = AsyncMock(return_value=None)

        with patch.object(adapter, "_make_client", return_value=mock_client):
            with pytest.raises(ValueError, match="aweme_id"):
                # The short URL "https://v.douyin.com/abc123" itself doesn't contain
                # a video ID, so after resolution failure it raises ValueError
                await adapter.download_single(
                    "https://v.douyin.com/no-id",
                    tmp_path,
                )

    @pytest.mark.asyncio
    async def test_non_short_url_does_not_call_resolve(self, tmp_path: Path) -> None:
        aweme_data = {
            "aweme_id": "123456",
            "desc": "direct",
            "create_time": 1700000000,
            "author": {"uid": "99", "nickname": "X"},
            "statistics": {},
            "video": {
                "play_addr": {"url_list": ["https://cdn.example.com/vid.mp4"]},
                "cover": {"url_list": []},
            },
        }

        adapter = DouyinAdapter()
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.headers = {}
        mock_client.cookies = {}
        mock_client.resolve_short_url = AsyncMock()
        mock_client.get_video_detail = AsyncMock(return_value=aweme_data)
        mock_client.sign_url = MagicMock(
            return_value=("https://cdn.example.com/vid.mp4&X-Bogus=x", "UA")
        )
        mock_client.build_signed_path = MagicMock(
            return_value=("https://www.douyin.com/aweme/v1/play/?X-Bogus=x", "UA")
        )

        mock_http = AsyncMock()
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.content = b"data"
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_client._client = mock_http

        with patch.object(adapter, "_make_client", return_value=mock_client):
            await adapter.download_single(
                "https://www.douyin.com/video/123456",
                tmp_path,
            )

        # resolve_short_url should NOT be called for a non-short URL
        mock_client.resolve_short_url.assert_not_awaited()
