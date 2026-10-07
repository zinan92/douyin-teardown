"""Tests for DouyinAdapter — all HTTP calls are mocked."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import httpx
import pytest

from content_downloader.adapters.douyin.adapter import DouyinAdapter
from content_downloader.adapters.douyin.api_client import DouyinAPIClient
from content_downloader.models import ContentItem


@pytest.fixture(autouse=True)
def _patch_download_file():
    """Patch _download_file globally so tests don't hit the network.

    Writes dummy bytes to dest based on file extension.
    """
    async def _fake_download(_client, _url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "video" in str(dest) or dest.suffix == ".mp4":
            dest.write_bytes(b"\x00\x01\x02\x03video_data")
        else:
            dest.write_bytes(b"\xff\xd8\xff\xe0cover_data")

    with patch("content_downloader.adapters.douyin.adapter._download_file",
               side_effect=_fake_download):
        yield

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


def _make_dummy_bytes() -> bytes:
    """Return tiny dummy bytes to simulate a downloaded file."""
    return b"\x00\x01\x02\x03video_data"


def _make_dummy_image_bytes() -> bytes:
    return b"\xff\xd8\xff\xe0cover_data"


class TestCanHandle:
    def test_video_url(self) -> None:
        adapter = DouyinAdapter()
        assert adapter.can_handle("https://www.douyin.com/video/7380308675841297704")

    def test_user_url(self) -> None:
        adapter = DouyinAdapter()
        assert adapter.can_handle("https://www.douyin.com/user/MS4wLjABAAAAtest")

    def test_short_url(self) -> None:
        adapter = DouyinAdapter()
        assert adapter.can_handle("https://v.douyin.com/abc123")

    def test_non_douyin_url_returns_false(self) -> None:
        adapter = DouyinAdapter()
        assert not adapter.can_handle("https://www.youtube.com/watch?v=xyz")

    def test_xhs_url_returns_false(self) -> None:
        adapter = DouyinAdapter()
        assert not adapter.can_handle("https://www.xiaohongshu.com/explore/abc")


class TestDownloadSingle:
    @pytest.mark.asyncio
    async def test_downloads_video_creates_directory_structure(
        self, tmp_path: Path
    ) -> None:
        fixture = _load_fixture("aweme_detail.json")
        aweme = fixture["aweme_detail"]

        adapter = DouyinAdapter(cookies={"msToken": "fake"})

        # Mock the entire API client
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_video_detail = AsyncMock(return_value=aweme)
        mock_client.resolve_short_url = AsyncMock(return_value=None)
        mock_client.sign_url = MagicMock(
            return_value=(
                "https://v19-webapp.douyin.com/video/play/v0300fg10000cqt5ijbc77uftb2n5uag/?watermark=0&X-Bogus=fake",
                "FakeUserAgent/1.0",
            )
        )
        mock_client.build_signed_path = MagicMock(
            return_value=(
                "https://www.douyin.com/aweme/v1/play/?video_id=vid&X-Bogus=fake",
                "FakeUserAgent/1.0",
            )
        )

        with patch.object(adapter, "_make_client", return_value=mock_client):
            item = await adapter.download_single(
                "https://www.douyin.com/video/7380308675841297704",
                tmp_path,
            )

        assert item.platform == "douyin"
        assert item.content_id == "7380308675841297704"
        assert item.content_type == "video"
        assert item.author_id == "12345678"
        assert item.likes == 1234
        assert item.comments == 56
        assert item.views == 9999

        # Verify directory structure
        content_dir = tmp_path / "douyin" / "12345678" / "7380308675841297704"
        assert content_dir.exists()
        assert (content_dir / "media").exists()
        assert (content_dir / "metadata.json").exists()
        assert (content_dir / "content_item.json").exists()

        # Verify metadata.json contains raw aweme data
        metadata = json.loads((content_dir / "metadata.json").read_text())
        assert metadata["aweme_id"] == "7380308675841297704"

        # Verify content_item.json is valid ContentItem
        ci = ContentItem.model_validate_json((content_dir / "content_item.json").read_text())
        assert ci.content_id == "7380308675841297704"

    @pytest.mark.asyncio
    async def test_raises_value_error_for_invalid_url(self, tmp_path: Path) -> None:
        adapter = DouyinAdapter()
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.resolve_short_url = AsyncMock(return_value=None)

        with patch.object(adapter, "_make_client", return_value=mock_client):
            with pytest.raises(ValueError, match="aweme_id"):
                await adapter.download_single(
                    "https://www.douyin.com/not-a-video-url",
                    tmp_path,
                )

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_api_failure(self, tmp_path: Path) -> None:
        adapter = DouyinAdapter()
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.resolve_short_url = AsyncMock(return_value=None)
        mock_client.get_video_detail = AsyncMock(return_value=None)

        with patch.object(adapter, "_make_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                await adapter.download_single(
                    "https://www.douyin.com/video/9999999999",
                    tmp_path,
                )

    @pytest.mark.asyncio
    async def test_media_files_list_populated(self, tmp_path: Path) -> None:
        fixture = _load_fixture("aweme_detail.json")
        aweme = fixture["aweme_detail"]

        adapter = DouyinAdapter(cookies={"msToken": "fake"})
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_video_detail = AsyncMock(return_value=aweme)
        mock_client.resolve_short_url = AsyncMock(return_value=None)
        mock_client.sign_url = MagicMock(
            return_value=("https://v19-webapp.douyin.com/video?watermark=0&X-Bogus=x", "UA")
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
            item = await adapter.download_single(
                "https://www.douyin.com/video/7380308675841297704",
                tmp_path,
            )

        assert "media/video.mp4" in item.media_files
        assert item.cover_file == "media/cover.jpg"


class TestDownloadProfile:
    @pytest.mark.asyncio
    async def test_downloads_items_respects_limit(self, tmp_path: Path) -> None:
        page1 = _load_fixture("user_post_page1.json")
        page2 = _load_fixture("user_post_page2.json")

        adapter = DouyinAdapter(cookies={"msToken": "fake"})
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Returns page1 first, then page2
        mock_client.get_user_post = AsyncMock(
            side_effect=[
                DouyinAPIClient._normalize_paged_response(page1, item_keys=["aweme_list"]),
                DouyinAPIClient._normalize_paged_response(page2, item_keys=["aweme_list"]),
            ]
        )
        mock_client.sign_url = MagicMock(return_value=("https://signed.url", "UA"))
        mock_client.build_signed_path = MagicMock(return_value=("https://signed.url", "UA"))

        mock_http = AsyncMock()
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.content = b"data"
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_client._client = mock_http

        with patch.object(adapter, "_make_client", return_value=mock_client):
            result = await adapter.download_profile(
                "https://www.douyin.com/user/98765",
                tmp_path,
                limit=3,
            )

        assert result.success == 3
        assert result.total == 3
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_since_filter_skips_old_items(self, tmp_path: Path) -> None:
        page1 = _load_fixture("user_post_page1.json")

        adapter = DouyinAdapter(cookies={"msToken": "fake"})
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # page1 has items at create_time 1700001000 and 1700002000
        # since = 1700001500 → only 1700002000 item passes
        since_dt = datetime.fromtimestamp(1700001500, tz=timezone.utc)

        mock_client.get_user_post = AsyncMock(
            side_effect=[
                DouyinAPIClient._normalize_paged_response(page1, item_keys=["aweme_list"]),
                DouyinAPIClient._normalize_paged_response(
                    {"aweme_list": [], "has_more": 0, "max_cursor": 0},
                    item_keys=["aweme_list"],
                ),
            ]
        )
        mock_client.sign_url = MagicMock(return_value=("https://signed.url", "UA"))
        mock_client.build_signed_path = MagicMock(return_value=("https://signed.url", "UA"))

        mock_http = AsyncMock()
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.content = b"data"
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_client._client = mock_http

        with patch.object(adapter, "_make_client", return_value=mock_client):
            result = await adapter.download_profile(
                "https://www.douyin.com/user/98765",
                tmp_path,
                since=since_dt,
            )

        # Only 1 item passes the since filter (create_time 1700002000 > 1700001500)
        assert result.success == 1

    @pytest.mark.asyncio
    async def test_stagnation_guard_stops_pagination(self, tmp_path: Path) -> None:
        """When cursor doesn't advance, pagination should stop."""
        stagnant_page = {
            "aweme_list": [
                {
                    "aweme_id": "9001",
                    "desc": "stagnant",
                    "create_time": 1700001000,
                    "author": {"uid": "98765", "nickname": "Creator"},
                    "statistics": {},
                    "share_url": "https://www.douyin.com/video/9001",
                    "video": {
                        "play_addr": {"url_list": ["https://cdn.example.com/vid.mp4"]},
                        "cover": {"url_list": ["https://cdn.example.com/cover.jpg"]},
                    },
                }
            ],
            "has_more": 1,
            "max_cursor": 0,  # cursor stays at 0 — stagnation
        }

        adapter = DouyinAdapter(cookies={"msToken": "fake"})
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_user_post = AsyncMock(
            return_value=DouyinAPIClient._normalize_paged_response(
                stagnant_page, item_keys=["aweme_list"]
            )
        )
        mock_client.sign_url = MagicMock(return_value=("https://signed.url", "UA"))
        mock_client.build_signed_path = MagicMock(return_value=("https://signed.url", "UA"))

        mock_http = AsyncMock()
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.content = b"data"
        mock_resp.raise_for_status = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_client._client = mock_http

        with patch.object(adapter, "_make_client", return_value=mock_client):
            result = await adapter.download_profile(
                "https://www.douyin.com/user/98765",
                tmp_path,
            )

        # Should download the 1 item then stop (not loop forever)
        assert result.success >= 1
        # get_user_post should only be called once (stagnation guard fires after first page)
        assert mock_client.get_user_post.call_count == 1

    @pytest.mark.asyncio
    async def test_raises_on_invalid_profile_url(self, tmp_path: Path) -> None:
        adapter = DouyinAdapter()
        with pytest.raises(ValueError, match="sec_uid"):
            await adapter.download_profile(
                "https://www.douyin.com/invalid-path",
                tmp_path,
            )

    @pytest.mark.asyncio
    async def test_download_error_recorded_on_failure(self, tmp_path: Path) -> None:
        page1 = _load_fixture("user_post_page1.json")

        adapter = DouyinAdapter(cookies={"msToken": "fake"})
        mock_client = AsyncMock(spec=DouyinAPIClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_user_post = AsyncMock(
            return_value=DouyinAPIClient._normalize_paged_response(
                page1, item_keys=["aweme_list"]
            )
        )
        mock_client.sign_url = MagicMock(return_value=("https://signed.url", "UA"))
        mock_client.build_signed_path = MagicMock(return_value=("https://signed.url", "UA"))

        # Override the autouse _download_file patch to simulate network failure
        async def _fail_download(_client, _url, _dest):
            raise Exception("Connection refused")

        with patch.object(adapter, "_make_client", return_value=mock_client), \
             patch("content_downloader.adapters.douyin.adapter._download_file",
                   side_effect=_fail_download):
            result = await adapter.download_profile(
                "https://www.douyin.com/user/98765",
                tmp_path,
                limit=2,
            )

        assert result.failed == 2
        assert len(result.errors) == 2
        assert result.errors[0].error_type == "network"
