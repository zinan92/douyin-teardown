"""Unit tests for WeChatOAParser HTML extraction.

All tests use the sample_article.html fixture — no real HTTP requests.
"""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from content_downloader.adapters.wechat_oa.parser import (
    WeChatOAParser,
    _extract_title,
    _extract_author,
    _extract_publish_time,
    _extract_body,
    _extract_images,
    _extract_audio,
)

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_html() -> str:
    return (FIXTURES_DIR / "sample_article.html").read_text(encoding="utf-8")


@pytest.fixture
def parser() -> WeChatOAParser:
    return WeChatOAParser()


# ---------------------------------------------------------------------------
# _extract_title
# ---------------------------------------------------------------------------


class TestExtractTitle:
    def test_extracts_h1_rich_media_title(self, sample_html: str) -> None:
        title = _extract_title(sample_html)
        assert title == "2024年AI发展回顾与展望"

    def test_falls_back_to_og_title(self) -> None:
        html = '<meta property="og:title" content="Fallback Title" />'
        assert _extract_title(html) == "Fallback Title"

    def test_returns_empty_string_when_missing(self) -> None:
        assert _extract_title("<html></html>") == ""

    def test_strips_inner_html_tags(self) -> None:
        html = '<h1 class="rich_media_title"><span>Hello</span></h1>'
        assert _extract_title(html) == "Hello"

    def test_og_title_html_entities_decoded(self) -> None:
        html = '<meta property="og:title" content="A &amp; B" />'
        assert _extract_title(html) == "A & B"


# ---------------------------------------------------------------------------
# _extract_author
# ---------------------------------------------------------------------------


class TestExtractAuthor:
    def test_extracts_js_name(self, sample_html: str) -> None:
        author = _extract_author(sample_html)
        assert author == "科技前沿观察"

    def test_falls_back_to_meta_text(self) -> None:
        html = '<span class="rich_media_meta_text">My OA Name</span>'
        assert _extract_author(html) == "My OA Name"

    def test_returns_empty_string_when_missing(self) -> None:
        assert _extract_author("<html></html>") == ""

    def test_strips_whitespace(self) -> None:
        html = '<a id="js_name" href="#">  TechNews  </a>'
        assert _extract_author(html) == "TechNews"


# ---------------------------------------------------------------------------
# _extract_publish_time
# ---------------------------------------------------------------------------


class TestExtractPublishTime:
    def test_extracts_publish_time_em(self, sample_html: str) -> None:
        t = _extract_publish_time(sample_html)
        assert t == "2023-12-19"

    def test_falls_back_to_ct_var(self) -> None:
        html = "<script>var ct = \"1703001600\";</script>"
        assert _extract_publish_time(html) == "1703001600"

    def test_ct_var_with_single_quotes(self) -> None:
        html = "<script>var ct = '1703001600';</script>"
        assert _extract_publish_time(html) == "1703001600"

    def test_returns_empty_string_when_missing(self) -> None:
        assert _extract_publish_time("<html></html>") == ""


# ---------------------------------------------------------------------------
# _extract_body
# ---------------------------------------------------------------------------


class TestExtractBody:
    def test_extracts_js_content(self, sample_html: str) -> None:
        body = _extract_body(sample_html)
        assert "人工智能技术" in body
        assert "大语言模型" in body

    def test_returns_empty_string_when_missing(self) -> None:
        assert _extract_body("<html></html>") == ""

    def test_includes_img_tags(self, sample_html: str) -> None:
        body = _extract_body(sample_html)
        assert "data-src" in body


# ---------------------------------------------------------------------------
# _extract_images
# ---------------------------------------------------------------------------


class TestExtractImages:
    def test_extracts_data_src_urls(self, sample_html: str) -> None:
        images = _extract_images(sample_html)
        assert len(images) == 2
        assert "https://mmbiz.qpic.cn/mmbiz_jpg/abc123/0?wx_fmt=jpeg" in images
        assert "https://mmbiz.qpic.cn/mmbiz_png/def456/0?wx_fmt=png" in images

    def test_returns_empty_list_when_no_images(self) -> None:
        assert _extract_images("<html><p>Text only</p></html>") == []

    def test_deduplicates_urls(self) -> None:
        html = (
            '<img data-src="https://example.com/img.jpg" />'
            '<img data-src="https://example.com/img.jpg" />'
        )
        images = _extract_images(html)
        assert len(images) == 1

    def test_skips_non_https_data_src(self) -> None:
        html = '<img data-src="blob:local" />'
        images = _extract_images(html)
        assert images == []


# ---------------------------------------------------------------------------
# _extract_audio
# ---------------------------------------------------------------------------


class TestExtractAudio:
    def test_extracts_mpvoice_file_ids(self, sample_html: str) -> None:
        audio = _extract_audio(sample_html)
        assert audio == ["MzAwMDAwMDAwMQ=="]

    def test_returns_empty_list_when_no_audio(self) -> None:
        assert _extract_audio("<html><p>No audio</p></html>") == []

    def test_extracts_multiple_audio_tags(self) -> None:
        html = (
            '<mpvoice voice_encode_fileid="AAA" />'
            '<mpvoice voice_encode_fileid="BBB" />'
        )
        audio = _extract_audio(html)
        assert audio == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# WeChatOAParser.parse_html integration
# ---------------------------------------------------------------------------


class TestWeChatOAParserParseHtml:
    def test_parse_html_returns_complete_dict(
        self, parser: WeChatOAParser, sample_html: str
    ) -> None:
        url = "https://mp.weixin.qq.com/s/test123"
        result = parser.parse_html(sample_html, url)

        assert result["title"] == "2024年AI发展回顾与展望"
        assert result["author"] == "科技前沿观察"
        assert result["publish_time"] == "2023-12-19"
        assert "人工智能技术" in result["content_html"]
        assert len(result["images"]) == 2
        assert result["audio_urls"] == ["MzAwMDAwMDAwMQ=="]
        assert result["source_url"] == url

    def test_parse_html_empty_page(self, parser: WeChatOAParser) -> None:
        result = parser.parse_html("<html></html>", "https://mp.weixin.qq.com/s/x")
        assert result["title"] == ""
        assert result["author"] == ""
        assert result["images"] == []
        assert result["audio_urls"] == []


# ---------------------------------------------------------------------------
# WeChatOAParser.fetch_article (mocked HTTP)
# ---------------------------------------------------------------------------


class TestWeChatOAParserFetchArticle:
    @pytest.mark.asyncio
    async def test_fetch_article_calls_httpx_get(
        self, parser: WeChatOAParser, sample_html: str
    ) -> None:
        mock_response = MagicMock()
        mock_response.text = sample_html
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            url = "https://mp.weixin.qq.com/s/abc123"
            result = await parser.fetch_article(url)

        assert result["title"] == "2024年AI发展回顾与展望"
        assert result["source_url"] == url
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == url

    @pytest.mark.asyncio
    async def test_fetch_article_raises_on_http_error(
        self, parser: WeChatOAParser
    ) -> None:
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Not Found",
                    request=MagicMock(),
                    response=MagicMock(status_code=404),
                )
            )
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await parser.fetch_article("https://mp.weixin.qq.com/s/notfound")
