"""Tests for WeChatOAAdapter — download flow with mocked HTTP and parser."""

from __future__ import annotations

import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from content_downloader.adapters.wechat_oa.adapter import (
    WeChatOAAdapter,
    _extract_article_id,
)
from content_downloader.adapters.wechat_oa.parser import WeChatOAParser
from content_downloader.models import ContentItem

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# _extract_article_id
# ---------------------------------------------------------------------------


class TestExtractArticleId:
    def test_short_form_url(self) -> None:
        url = "https://mp.weixin.qq.com/s/AbC123xyz"
        assert _extract_article_id(url) == "AbC123xyz"

    def test_long_form_url_with_mid_and_idx(self) -> None:
        url = "https://mp.weixin.qq.com/s?__biz=MzAxNzY&mid=2247483647&idx=1&sn=abc"
        result = _extract_article_id(url)
        assert result == "2247483647_1"

    def test_long_form_url_mid_only(self) -> None:
        url = "https://mp.weixin.qq.com/s?__biz=MzAxNzY&mid=2247483647"
        result = _extract_article_id(url)
        assert result == "2247483647"

    def test_fallback_for_unknown_form(self) -> None:
        url = "https://mp.weixin.qq.com/other/path"
        result = _extract_article_id(url)
        assert result  # non-empty
        assert "/" not in result  # sanitised


# ---------------------------------------------------------------------------
# WeChatOAAdapter.can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_matches_short_article_url(self) -> None:
        adapter = WeChatOAAdapter()
        assert adapter.can_handle("https://mp.weixin.qq.com/s/AbC123xyz") is True

    def test_matches_http_url(self) -> None:
        adapter = WeChatOAAdapter()
        assert adapter.can_handle("http://mp.weixin.qq.com/s/AbC123xyz") is True

    def test_rejects_other_domains(self) -> None:
        adapter = WeChatOAAdapter()
        assert adapter.can_handle("https://weixin.qq.com/s/AbC") is False
        assert adapter.can_handle("https://xiaohongshu.com/explore/abc") is False

    def test_rejects_non_s_paths(self) -> None:
        adapter = WeChatOAAdapter()
        # mp.weixin.qq.com but not /s/
        assert adapter.can_handle("https://mp.weixin.qq.com/mp/homepage") is False


# ---------------------------------------------------------------------------
# WeChatOAAdapter.download_single
# ---------------------------------------------------------------------------

SAMPLE_HTML = (FIXTURES_DIR / "sample_article.html").read_text(encoding="utf-8")


def _make_parsed_article(source_url: str = "https://mp.weixin.qq.com/s/test123") -> dict:
    """Return a fully-populated parsed article dict (as if from WeChatOAParser)."""
    return {
        "title": "2024年AI发展回顾与展望",
        "author": "科技前沿观察",
        "publish_time": "2023-12-19",
        "content_html": "<p>人工智能技术取得了突破性进展。</p>",
        "images": [
            "https://mmbiz.qpic.cn/mmbiz_jpg/abc123/0?wx_fmt=jpeg",
            "https://mmbiz.qpic.cn/mmbiz_png/def456/0?wx_fmt=png",
        ],
        "audio_urls": ["MzAwMDAwMDAwMQ=="],
        "source_url": source_url,
    }


def _make_mock_parser(parsed: dict) -> WeChatOAParser:
    """Return a mock WeChatOAParser that returns `parsed` from fetch_article."""
    mock_parser = MagicMock(spec=WeChatOAParser)
    mock_parser.fetch_article = AsyncMock(return_value=parsed)
    return mock_parser


@pytest.fixture
def fake_image_response() -> MagicMock:
    resp = MagicMock()
    resp.content = b"fake-image-bytes"
    resp.raise_for_status = MagicMock()
    return resp


class TestDownloadSingle:
    @pytest.mark.asyncio
    async def test_creates_directory_structure(self, tmp_path: pathlib.Path) -> None:
        url = "https://mp.weixin.qq.com/s/test123"
        parsed = _make_parsed_article(url)
        adapter = WeChatOAAdapter(parser=_make_mock_parser(parsed))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.content = b"fake"
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            item = await adapter.download_single(url, tmp_path)

        content_dir = tmp_path / "wechat_oa" / "科技前沿观察" / "test123"
        assert content_dir.exists()
        assert (content_dir / "media").is_dir()

    @pytest.mark.asyncio
    async def test_saves_article_html(self, tmp_path: pathlib.Path) -> None:
        url = "https://mp.weixin.qq.com/s/test123"
        parsed = _make_parsed_article(url)
        adapter = WeChatOAAdapter(parser=_make_mock_parser(parsed))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.content = b"fake"
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await adapter.download_single(url, tmp_path)

        article_html = (
            tmp_path / "wechat_oa" / "科技前沿观察" / "test123" / "media" / "article.html"
        )
        assert article_html.exists()
        assert "人工智能技术" in article_html.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_downloads_images(self, tmp_path: pathlib.Path) -> None:
        url = "https://mp.weixin.qq.com/s/test123"
        parsed = _make_parsed_article(url)
        adapter = WeChatOAAdapter(parser=_make_mock_parser(parsed))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.content = b"fake-image-bytes"
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            item = await adapter.download_single(url, tmp_path)

        media_dir = tmp_path / "wechat_oa" / "科技前沿观察" / "test123" / "media"
        assert (media_dir / "img_01.jpg").exists()
        assert (media_dir / "img_02.jpg").exists()
        assert "media/img_01.jpg" in item.media_files
        assert "media/img_02.jpg" in item.media_files

    @pytest.mark.asyncio
    async def test_saves_audio_ids_txt(self, tmp_path: pathlib.Path) -> None:
        url = "https://mp.weixin.qq.com/s/test123"
        parsed = _make_parsed_article(url)
        adapter = WeChatOAAdapter(parser=_make_mock_parser(parsed))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.content = b"fake"
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            item = await adapter.download_single(url, tmp_path)

        audio_ids = (
            tmp_path / "wechat_oa" / "科技前沿观察" / "test123" / "media" / "audio_ids.txt"
        )
        assert audio_ids.exists()
        assert "MzAwMDAwMDAwMQ==" in audio_ids.read_text(encoding="utf-8")
        assert "media/audio_ids.txt" in item.media_files

    @pytest.mark.asyncio
    async def test_writes_metadata_json(self, tmp_path: pathlib.Path) -> None:
        url = "https://mp.weixin.qq.com/s/test123"
        parsed = _make_parsed_article(url)
        adapter = WeChatOAAdapter(parser=_make_mock_parser(parsed))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.content = b"fake"
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await adapter.download_single(url, tmp_path)

        metadata_path = (
            tmp_path / "wechat_oa" / "科技前沿观察" / "test123" / "metadata.json"
        )
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["title"] == "2024年AI发展回顾与展望"
        assert metadata["author"] == "科技前沿观察"

    @pytest.mark.asyncio
    async def test_writes_content_item_json(self, tmp_path: pathlib.Path) -> None:
        url = "https://mp.weixin.qq.com/s/test123"
        parsed = _make_parsed_article(url)
        adapter = WeChatOAAdapter(parser=_make_mock_parser(parsed))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.content = b"fake"
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await adapter.download_single(url, tmp_path)

        ci_path = (
            tmp_path / "wechat_oa" / "科技前沿观察" / "test123" / "content_item.json"
        )
        assert ci_path.exists()
        ci = json.loads(ci_path.read_text(encoding="utf-8"))
        assert ci["platform"] == "wechat_oa"
        assert ci["content_type"] == "article"

    @pytest.mark.asyncio
    async def test_returns_content_item(self, tmp_path: pathlib.Path) -> None:
        url = "https://mp.weixin.qq.com/s/test123"
        parsed = _make_parsed_article(url)
        adapter = WeChatOAAdapter(parser=_make_mock_parser(parsed))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.content = b"fake"
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            item = await adapter.download_single(url, tmp_path)

        assert isinstance(item, ContentItem)
        assert item.platform == "wechat_oa"
        assert item.content_id == "test123"
        assert item.content_type == "article"
        assert item.title == "2024年AI发展回顾与展望"
        assert item.author_name == "科技前沿观察"
        assert item.publish_time == "2023-12-19"
        assert item.source_url == url
        assert item.cover_file == "media/img_01.jpg"
        assert item.likes == 0
        assert item.comments == 0

    @pytest.mark.asyncio
    async def test_no_audio_when_empty(self, tmp_path: pathlib.Path) -> None:
        url = "https://mp.weixin.qq.com/s/noaudio"
        parsed = _make_parsed_article(url)
        parsed = {**parsed, "audio_urls": [], "source_url": url}
        adapter = WeChatOAAdapter(parser=_make_mock_parser(parsed))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.content = b"fake"
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            item = await adapter.download_single(url, tmp_path)

        # No audio_ids.txt when no audio
        audio_ids = (
            tmp_path / "wechat_oa" / "科技前沿观察" / "noaudio" / "media" / "audio_ids.txt"
        )
        assert not audio_ids.exists()
        assert "media/audio_ids.txt" not in item.media_files

    @pytest.mark.asyncio
    async def test_image_download_failure_is_non_fatal(self, tmp_path: pathlib.Path) -> None:
        """Failed image downloads should be skipped, not crash the whole download."""
        url = "https://mp.weixin.qq.com/s/test_img_fail"
        import httpx as real_httpx
        parsed = _make_parsed_article(url)
        parsed = {**parsed, "source_url": url}
        adapter = WeChatOAAdapter(parser=_make_mock_parser(parsed))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(
                side_effect=real_httpx.ConnectError("Network unreachable")
            )
            mock_client_cls.return_value = mock_client

            item = await adapter.download_single(url, tmp_path)

        # article.html should still be saved
        article_html = (
            tmp_path / "wechat_oa" / "科技前沿观察" / "test_img_fail" / "media" / "article.html"
        )
        assert article_html.exists()
        # No images since all downloads failed
        assert "media/img_01.jpg" not in item.media_files
        # cover_file should be None
        assert item.cover_file is None


# ---------------------------------------------------------------------------
# WeChatOAAdapter.download_profile
# ---------------------------------------------------------------------------


class TestDownloadProfile:
    @pytest.mark.asyncio
    async def test_returns_unsupported_result(self, tmp_path: pathlib.Path) -> None:
        adapter = WeChatOAAdapter()
        result = await adapter.download_profile(
            "https://mp.weixin.qq.com/some_oa_homepage",
            tmp_path,
        )
        assert result.total == 0
        assert result.success == 0
        assert result.failed == 1
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "unsupported"

    @pytest.mark.asyncio
    async def test_unsupported_message_contains_workaround(self, tmp_path: pathlib.Path) -> None:
        adapter = WeChatOAAdapter()
        result = await adapter.download_profile(
            "https://mp.weixin.qq.com/some_oa_homepage",
            tmp_path,
        )
        assert "content-downloader download" in result.errors[0].message


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------


class TestRouterIntegration:
    def test_router_routes_wechat_oa_url(self) -> None:
        from content_downloader.router import get_adapter, classify_url

        url = "https://mp.weixin.qq.com/s/AbCdEfGh123"
        platform, url_type = classify_url(url)
        assert platform == "wechat_oa"
        assert url_type == "single"

        adapter = get_adapter(url)
        assert adapter.platform == "wechat_oa"
        assert adapter.can_handle(url) is True

    def test_wechat_oa_in_supported_platforms(self) -> None:
        from content_downloader.router import list_supported_platforms
        assert "wechat_oa" in list_supported_platforms()
