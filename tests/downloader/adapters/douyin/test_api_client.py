"""Tests for DouyinAPIClient — all HTTP calls are mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from content_downloader.adapters.douyin.api_client import DouyinAPIClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


def _make_mock_response(data: dict, status_code: int = 200) -> MagicMock:
    """Build a mock httpx.Response that returns JSON data."""
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.json.return_value = data
    mock.url = httpx.URL("https://www.douyin.com/test")
    return mock


class TestDouyinAPIClientInit:
    def test_instantiates_with_no_cookies(self) -> None:
        client = DouyinAPIClient()
        assert client.cookies == {}
        assert client.platform if hasattr(client, "platform") else True

    def test_instantiates_with_cookies(self) -> None:
        cookies = {"ttwid": "abc", "msToken": "mytoken"}
        client = DouyinAPIClient(cookies=cookies)
        assert client.cookies["ttwid"] == "abc"
        assert client._ms_token == "mytoken"

    def test_cookies_are_sanitized(self) -> None:
        # Keys/values should be stripped strings
        client = DouyinAPIClient(cookies={"  key  ": "  val  "})
        assert "key" in client.cookies
        assert client.cookies["key"] == "val"


class TestGetVideoDetail:
    @pytest.mark.asyncio
    async def test_returns_aweme_detail_on_success(self) -> None:
        fixture = _load_fixture("aweme_detail.json")
        mock_resp = _make_mock_response(fixture)

        client = DouyinAPIClient(cookies={"msToken": "fake_token"})
        with patch.object(client, "_request_json", new=AsyncMock(return_value=fixture)):
            result = await client.get_video_detail("7380308675841297704")

        assert result is not None
        assert result["aweme_id"] == "7380308675841297704"
        assert result["author"]["uid"] == "12345678"

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_response(self) -> None:
        client = DouyinAPIClient(cookies={"msToken": "fake_token"})
        with patch.object(client, "_request_json", new=AsyncMock(return_value={})):
            result = await client.get_video_detail("nonexistent_id")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_aweme_detail_missing(self) -> None:
        client = DouyinAPIClient(cookies={"msToken": "fake_token"})
        with patch.object(
            client, "_request_json", new=AsyncMock(return_value={"status_code": 0})
        ):
            result = await client.get_video_detail("789")

        assert result is None


class TestGetUserPost:
    @pytest.mark.asyncio
    async def test_returns_normalized_page(self) -> None:
        fixture = _load_fixture("user_post_page1.json")
        client = DouyinAPIClient(cookies={"msToken": "fake_token"})
        with patch.object(client, "_request_json", new=AsyncMock(return_value=fixture)):
            result = await client.get_user_post("sec_uid_test")

        assert "aweme_list" in result
        assert len(result["aweme_list"]) == 2
        assert result["has_more"] is True
        assert result["max_cursor"] == 1700002000

    @pytest.mark.asyncio
    async def test_empty_list_when_no_items(self) -> None:
        client = DouyinAPIClient(cookies={"msToken": "fake_token"})
        with patch.object(client, "_request_json", new=AsyncMock(return_value={})):
            result = await client.get_user_post("sec_uid_empty")

        assert result["aweme_list"] == []
        assert result["has_more"] is False


class TestNormalizePaged:
    def test_extracts_aweme_list(self) -> None:
        raw = {"aweme_list": [{"aweme_id": "1"}], "has_more": 1, "max_cursor": 100}
        result = DouyinAPIClient._normalize_paged_response(raw)
        assert len(result["items"]) == 1
        assert result["has_more"] is True
        assert result["max_cursor"] == 100

    def test_bool_has_more(self) -> None:
        raw = {"aweme_list": [], "has_more": 0, "max_cursor": 0}
        result = DouyinAPIClient._normalize_paged_response(raw)
        assert result["has_more"] is False

    def test_handles_empty_raw(self) -> None:
        result = DouyinAPIClient._normalize_paged_response({})
        assert result["items"] == []
        assert result["has_more"] is False
        assert result["max_cursor"] == 0


class TestResolveShortUrl:
    @pytest.mark.asyncio
    async def test_returns_resolved_url(self) -> None:
        client = DouyinAPIClient()
        mock_response = MagicMock()
        mock_response.url = httpx.URL("https://www.douyin.com/video/123456789")

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False
        client._client = mock_http_client

        result = await client.resolve_short_url("https://v.douyin.com/abc")
        assert result == "https://www.douyin.com/video/123456789"

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self) -> None:
        client = DouyinAPIClient()
        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(side_effect=Exception("network error"))
        mock_http_client.is_closed = False
        client._client = mock_http_client

        result = await client.resolve_short_url("https://v.douyin.com/err")
        assert result is None


class TestSignUrl:
    def test_sign_url_adds_xbogus(self) -> None:
        client = DouyinAPIClient()
        url = "https://www.douyin.com/aweme/v1/web/aweme/detail/?aweme_id=123"
        signed, ua = client.sign_url(url)
        assert "X-Bogus=" in signed
        assert ua == client.headers["User-Agent"]

    def test_build_signed_path(self) -> None:
        client = DouyinAPIClient()
        signed, ua = client.build_signed_path("/aweme/v1/play/", {"video_id": "abc"})
        assert "https://www.douyin.com/aweme/v1/play/" in signed
        assert ua  # non-empty user agent


def test_landed_on_video_rejects_the_404_redirect() -> None:
    """A private/removed video redirects to the feed and autoplays someone else's video."""
    from content_downloader.adapters.douyin.api_client import _landed_on_video

    vid = "7684647880715537727"
    assert _landed_on_video(f"https://www.douyin.com/video/{vid}", vid)
    assert _landed_on_video(f"https://www.douyin.com/jingxuan?modal_id={vid}", vid)
    assert not _landed_on_video(
        "https://www.douyin.com/jingxuan?previous_page=web_video_404_link&modal_id=7687200557127986153", vid
    )
    assert not _landed_on_video("https://www.douyin.com/jingxuan?modal_id=7687200557127986153", vid)
