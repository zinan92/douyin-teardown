"""Tests for XFetcher — mocks asyncio subprocess (two-phase: metadata + download).

NOTE: This code uses asyncio.create_subprocess_exec (NOT shell) intentionally
for security. The test mocks verify this. No shell injection risk.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from content_downloader.adapters.x.fetcher import XFetcher, _find_info_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_proc(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> AsyncMock:
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.wait = AsyncMock(return_value=returncode)
    return proc


# ---------------------------------------------------------------------------
# _find_info_json
# ---------------------------------------------------------------------------


def test_find_info_json_returns_file(tmp_path: Path) -> None:
    info_file = tmp_path / "1234567890.info.json"
    info_file.write_text("{}")
    assert _find_info_json(tmp_path) == info_file


def test_find_info_json_returns_none_when_missing(tmp_path: Path) -> None:
    assert _find_info_json(tmp_path) is None


def test_find_info_json_ignores_non_info_json_files(tmp_path: Path) -> None:
    (tmp_path / "video.mp4").write_bytes(b"fake")
    assert _find_info_json(tmp_path) is None


# ---------------------------------------------------------------------------
# XFetcher.is_available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_available_returns_true_when_yt_dlp_exits_zero() -> None:
    mock_proc = _make_mock_proc(returncode=0, stdout=b"2024.01.01")
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        assert await XFetcher().is_available() is True


@pytest.mark.asyncio
async def test_is_available_returns_false_when_not_found() -> None:
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
        assert await XFetcher().is_available() is False


@pytest.mark.asyncio
async def test_is_available_returns_false_on_nonzero() -> None:
    mock_proc = _make_mock_proc(returncode=1)
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        assert await XFetcher().is_available() is False


# ---------------------------------------------------------------------------
# XFetcher.fetch_post — tweet with media
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_post_video_tweet(tmp_path: Path) -> None:
    """fetch_post returns metadata and downloads media for video tweets."""
    sample_info = {
        "id": "123",
        "title": "Test tweet",
        "uploader": "user",
        "url": "https://video.twimg.com/test.mp4",
        "formats": [{"url": "https://video.twimg.com/test.mp4"}],
    }

    metadata_proc = _make_mock_proc(returncode=0, stdout=json.dumps(sample_info).encode())
    download_proc = _make_mock_proc(returncode=0)

    call_count = 0
    def fake_subprocess(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if "--dump-json" in args:
            return metadata_proc
        return download_proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
        result = await XFetcher().fetch_post("https://x.com/user/status/123", tmp_path)

    assert result["id"] == "123"
    assert call_count == 2  # metadata + download


# ---------------------------------------------------------------------------
# XFetcher.fetch_post — text-only tweet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_post_text_only_tweet(tmp_path: Path) -> None:
    """fetch_post handles text-only tweets gracefully."""
    mock_proc = _make_mock_proc(
        returncode=1,
        stderr=b"ERROR: [twitter] 123: No video could be found in this tweet",
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await XFetcher().fetch_post("https://x.com/user/status/123", tmp_path)

    assert result["id"] == "123"
    assert result.get("_text_only") is True


@pytest.mark.asyncio
async def test_fetch_post_text_only_skips_download(tmp_path: Path) -> None:
    """Text-only tweets skip the download phase."""
    mock_proc = _make_mock_proc(
        returncode=1,
        stderr=b"ERROR: [twitter] 456: No video could be found in this tweet",
    )

    call_count = 0
    def fake_subprocess(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
        await XFetcher().fetch_post("https://x.com/user/status/456", tmp_path)

    assert call_count == 1  # only metadata, no download


# ---------------------------------------------------------------------------
# XFetcher.fetch_post — real error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_post_raises_on_real_error(tmp_path: Path) -> None:
    mock_proc = _make_mock_proc(
        returncode=1,
        stderr=b"ERROR: [twitter] 123: Unable to download JSON metadata: HTTP Error 403",
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="metadata fetch failed"):
            await XFetcher().fetch_post("https://x.com/user/status/123", tmp_path)


# ---------------------------------------------------------------------------
# Security: uses subprocess_exec not shell
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uses_subprocess_exec_not_shell(tmp_path: Path) -> None:
    """Verify yt-dlp is called via create_subprocess_exec with positional args."""
    sample_info = {"id": "abc", "title": ""}
    mock_proc = _make_mock_proc(returncode=0, stdout=json.dumps(sample_info).encode())

    captured_args = []
    def fake_subprocess(*args, **kwargs):
        captured_args.append(args)
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
        await XFetcher().fetch_post("https://x.com/user/status/abc", tmp_path)

    assert captured_args[0][0] == "yt-dlp"
    assert "--dump-json" in captured_args[0]
