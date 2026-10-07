"""Tests for adapter registry and content-type routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from content_extractor.adapters.base import Extractor
from content_extractor.router import UnsupportedContentTypeError, get_extractor


class TestGetExtractor:
    """Verify get_extractor dispatches to correct adapter class."""

    def test_get_video_extractor(self) -> None:
        from content_extractor.adapters.video import VideoExtractor

        ext = get_extractor("video")
        assert isinstance(ext, VideoExtractor)

    def test_get_image_extractor(self) -> None:
        from content_extractor.adapters.image import ImageExtractor

        ext = get_extractor("image")
        assert isinstance(ext, ImageExtractor)

    def test_get_article_extractor(self) -> None:
        from content_extractor.adapters.article import ArticleExtractor

        ext = get_extractor("article")
        assert isinstance(ext, ArticleExtractor)

    def test_get_gallery_extractor(self) -> None:
        from content_extractor.adapters.gallery import GalleryExtractor

        ext = get_extractor("gallery")
        assert isinstance(ext, GalleryExtractor)


class TestUnsupportedContentType:
    """Verify unknown content types produce clear errors."""

    def test_unsupported_content_type_raises(self) -> None:
        with pytest.raises(UnsupportedContentTypeError) as exc_info:
            get_extractor("podcast")
        assert "Supported:" in str(exc_info.value)

    def test_unsupported_content_type_lists_available(self) -> None:
        with pytest.raises(UnsupportedContentTypeError) as exc_info:
            get_extractor("podcast")
        msg = str(exc_info.value)
        assert "article" in msg
        assert "gallery" in msg
        assert "image" in msg
        assert "video" in msg


class TestAdapterProtocol:
    """Verify all adapters satisfy the Extractor Protocol."""

    @pytest.mark.parametrize("content_type", ["video", "image", "article", "gallery"])
    def test_adapters_satisfy_protocol(self, content_type: str) -> None:
        ext = get_extractor(content_type)
        assert isinstance(ext, Extractor)
