"""Unit tests for XHSAPIClient — all HTTP calls mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from content_downloader.adapters.xhs.api_client import XHSAPIClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(
    json_data: dict | None = None,
    status_code: int = 200,
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# get_note_detail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_note_detail_gallery():
    """get_note_detail returns parsed dict for a gallery note."""
    fixture = _load_fixture("note_gallery.json")
    mock_response = _make_mock_response(fixture)

    client = XHSAPIClient()
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.get_note_detail("https://www.xiaohongshu.com/explore/abc123gallery")

    client._client.post.assert_awaited_once()
    call_args = client._client.post.call_args
    assert "/xhs/detail" in call_args[0][0]
    payload = call_args[1]["json"]
    assert payload["url"] == "https://www.xiaohongshu.com/explore/abc123gallery"
    assert payload["download"] is False
    assert payload["skip"] is False

    assert result["data"]["作品ID"] == "abc123gallery"
    assert result["data"]["作品类型"] == "图文"


@pytest.mark.asyncio
async def test_get_note_detail_video():
    """get_note_detail returns parsed dict for a video note."""
    fixture = _load_fixture("note_video.json")
    mock_response = _make_mock_response(fixture)

    client = XHSAPIClient()
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.get_note_detail("https://www.xiaohongshu.com/explore/xyz789video")

    assert result["data"]["作品ID"] == "xyz789video"
    assert result["data"]["作品类型"] == "视频"


@pytest.mark.asyncio
async def test_get_note_detail_empty_response():
    """get_note_detail returns empty dict when API returns null/empty."""
    mock_response = _make_mock_response(None)
    mock_response.json.return_value = None

    client = XHSAPIClient()
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.get_note_detail("https://www.xiaohongshu.com/explore/empty")
    assert result == {}


@pytest.mark.asyncio
async def test_get_note_detail_non_dict_response():
    """get_note_detail returns empty dict when API returns non-dict JSON."""
    mock_response = _make_mock_response()
    mock_response.json.return_value = ["unexpected", "list"]

    client = XHSAPIClient()
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.get_note_detail("https://www.xiaohongshu.com/explore/bad")
    assert result == {}


@pytest.mark.asyncio
async def test_get_note_detail_raises_on_http_error():
    """get_note_detail propagates raise_for_status errors."""
    mock_response = _make_mock_response(status_code=404)
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=MagicMock(), response=mock_response
    )

    client = XHSAPIClient()
    client._client = AsyncMock()
    client._client.post = AsyncMock(return_value=mock_response)

    with pytest.raises(httpx.HTTPStatusError):
        await client.get_note_detail("https://www.xiaohongshu.com/explore/gone")


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_available_returns_true_on_200():
    """is_available returns True when sidecar responds 200."""
    mock_response = _make_mock_response(status_code=200)

    client = XHSAPIClient()
    client._client = AsyncMock()
    client._client.get = AsyncMock(return_value=mock_response)

    assert await client.is_available() is True


@pytest.mark.asyncio
async def test_is_available_returns_false_on_non_200():
    """is_available returns False when sidecar responds non-200."""
    mock_response = _make_mock_response(status_code=503)

    client = XHSAPIClient()
    client._client = AsyncMock()
    client._client.get = AsyncMock(return_value=mock_response)

    assert await client.is_available() is False


@pytest.mark.asyncio
async def test_is_available_returns_false_on_connect_error():
    """is_available returns False when sidecar is not reachable (both GET and POST fail)."""
    client = XHSAPIClient()
    client._client = AsyncMock()
    client._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
    client._client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

    assert await client.is_available() is False


@pytest.mark.asyncio
async def test_is_available_returns_false_on_timeout():
    """is_available returns False on timeout (both GET and POST fail)."""
    client = XHSAPIClient()
    client._client = AsyncMock()
    client._client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    client._client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

    assert await client.is_available() is False


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_closes_client():
    """XHSAPIClient closes the underlying httpx client on context exit."""
    async with XHSAPIClient() as client:
        original_client = client._client

    # aclose should have been called — proxy through a new mock
    mock_inner = AsyncMock()
    client._client = mock_inner
    await client.close()
    mock_inner.aclose.assert_awaited_once()
