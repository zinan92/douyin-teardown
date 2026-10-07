"""Tests for vision module and ImageExtractor adapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from content_extractor.config import ExtractorConfig


# ---------------------------------------------------------------------------
# Task 1: Vision module tests — preprocess_image + describe_image
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    """Tests for preprocess_image function."""

    def test_small_jpeg_no_resize(self, tmp_image_dir: Path) -> None:
        """A small JPEG (100x100) should not be resized."""
        from content_extractor.vision import preprocess_image

        b64, media_type = preprocess_image(tmp_image_dir / "media" / "test.jpg")
        assert isinstance(b64, str)
        assert media_type == "image/jpeg"
        assert len(b64) > 0

    def test_large_image_resized(self, tmp_path: Path) -> None:
        """An image larger than 1568px should be resized proportionally."""
        from content_extractor.vision import preprocess_image

        large_img = Image.new("RGB", (3000, 4000), color=(0, 128, 255))
        img_path = tmp_path / "large.png"
        large_img.save(img_path, "PNG")

        b64, media_type = preprocess_image(img_path)
        assert media_type == "image/png"
        assert len(b64) > 0

    def test_unsupported_format_converted_to_jpeg(self, tmp_path: Path) -> None:
        """BMP (unsupported by Claude) should be converted to JPEG."""
        from content_extractor.vision import preprocess_image

        bmp_img = Image.new("RGB", (200, 200), color=(0, 255, 0))
        img_path = tmp_path / "test.bmp"
        bmp_img.save(img_path, "BMP")

        b64, media_type = preprocess_image(img_path)
        assert media_type == "image/jpeg"

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """preprocess_image should raise FileNotFoundError for missing file."""
        from content_extractor.vision import preprocess_image

        with pytest.raises(FileNotFoundError):
            preprocess_image(tmp_path / "nope.jpg")

    def test_rgba_png_converted_to_rgb_for_jpeg(self, tmp_path: Path) -> None:
        """RGBA image converted to JPEG should first drop alpha channel."""
        from content_extractor.vision import preprocess_image

        rgba_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img_path = tmp_path / "alpha.bmp"
        rgba_img.save(img_path, "BMP")

        b64, media_type = preprocess_image(img_path)
        assert media_type == "image/jpeg"
        assert len(b64) > 0


class TestDescribeImage:
    """Tests for describe_image function."""

    def _make_mock_response(self, text_content: str) -> MagicMock:
        """Create a mock anthropic response object."""
        mock_resp = MagicMock()
        content_block = SimpleNamespace(type="text", text=text_content)
        mock_resp.content = [content_block]
        return mock_resp

    def test_describe_image_calls_api_correctly(
        self, mock_vision_response_json: str
    ) -> None:
        """describe_image should call client.messages.create with image block."""
        from content_extractor.vision import describe_image

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response(
            mock_vision_response_json
        )

        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ):
            result = describe_image("base64data", "image/jpeg", config)

        # Verify API was called
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == config.claude_model
        assert call_kwargs.kwargs["max_tokens"] == config.claude_max_tokens

        # Verify result parsed correctly
        assert result.ocr_text == "小红书爆款笔记分享"
        assert "Chinese text overlay" in result.visual_description
        assert result.confidence == pytest.approx(0.92)

    def test_describe_image_malformed_json(self) -> None:
        """Malformed JSON response should use raw text as description."""
        from content_extractor.vision import describe_image

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response(
            "This is not valid JSON but a description of the image"
        )

        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ):
            result = describe_image("base64data", "image/jpeg", config)

        assert result.visual_description == (
            "This is not valid JSON but a description of the image"
        )
        assert result.ocr_text == ""
        assert result.confidence == 0.0

    def test_describe_image_api_error(self) -> None:
        """API errors should raise ImageExtractionError."""
        from content_extractor.vision import ImageExtractionError, describe_image

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API connection failed")

        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ):
            with pytest.raises(ImageExtractionError, match="API connection failed"):
                describe_image("base64data", "image/jpeg", config)

    def test_describe_image_prompt_requests_chinese(
        self, mock_vision_response_json: str
    ) -> None:
        """The prompt sent to Claude should request Chinese text transcription."""
        from content_extractor.vision import describe_image

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response(
            mock_vision_response_json
        )

        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ):
            describe_image("base64data", "image/jpeg", config)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        # Find the text prompt in the message content
        prompt_text = ""
        for msg in messages:
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    prompt_text += block["text"]

        assert "Chinese" in prompt_text or "chinese" in prompt_text.lower()
        assert "JSON" in prompt_text or "json" in prompt_text.lower()


# ---------------------------------------------------------------------------
# Task 2: ImageExtractor adapter tests
# ---------------------------------------------------------------------------


def _make_image_content_item(
    content_dir: Path,
    *,
    content_type: str = "image",
    media_files: tuple[str, ...] = ("media/test.jpg",),
) -> None:
    """Create a content_item.json for image testing."""
    item = {
        "platform": "xiaohongshu",
        "content_id": "img_001",
        "content_type": content_type,
        "title": "小红书图片笔记",
        "description": "测试图片内容提取",
        "author_id": "xhs_user_1",
        "author_name": "TestUser",
        "publish_time": "2026-03-30T10:00:00+00:00",
        "source_url": "https://www.xiaohongshu.com/explore/img_001",
        "media_files": list(media_files),
        "cover_file": None,
        "metadata_file": "metadata.json",
        "likes": 100,
        "comments": 10,
        "shares": 5,
        "collects": 50,
        "views": 1000,
        "downloaded_at": "2026-03-30T12:00:00Z",
    }
    (content_dir / "content_item.json").write_text(json.dumps(item))


def _create_test_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    """Create a small solid-color test image."""
    img = Image.new("RGB", size, color=(255, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG")


class TestImageExtractor:
    """Tests for ImageExtractor.extract() pipeline."""

    def _mock_describe(self, mock_vision_response_json: str) -> MagicMock:
        """Create a mock client returning canned Chinese OCR response."""
        mock_client = MagicMock()
        content_block = SimpleNamespace(type="text", text=mock_vision_response_json)
        mock_resp = MagicMock()
        mock_resp.content = [content_block]
        mock_client.messages.create.return_value = mock_resp
        return mock_client

    def test_extract_returns_extraction_result(
        self, tmp_path: Path, mock_vision_response_json: str
    ) -> None:
        """ImageExtractor.extract() should return ExtractionResult with image data."""
        from content_extractor.adapters.image import ImageExtractor

        _create_test_image(tmp_path / "media" / "test.jpg")
        _make_image_content_item(tmp_path)

        mock_client = self._mock_describe(mock_vision_response_json)
        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ):
            result = ImageExtractor().extract(tmp_path, config)

        assert result.content_type == "image"
        assert result.content_id == "img_001"
        assert len(result.media_descriptions) == 1
        desc = result.media_descriptions[0]
        assert desc.ocr_text == "小红书爆款笔记分享"
        assert desc.confidence == pytest.approx(0.92)

    def test_extract_raw_text_contains_ocr(
        self, tmp_path: Path, mock_vision_response_json: str
    ) -> None:
        """raw_text should contain the OCR text from the image."""
        from content_extractor.adapters.image import ImageExtractor

        _create_test_image(tmp_path / "media" / "test.jpg")
        _make_image_content_item(tmp_path)

        mock_client = self._mock_describe(mock_vision_response_json)
        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ):
            result = ImageExtractor().extract(tmp_path, config)

        assert "小红书爆款笔记分享" in result.raw_text

    def test_extract_quality_language_zh(
        self, tmp_path: Path, mock_vision_response_json: str
    ) -> None:
        """Quality should detect Chinese language from OCR text."""
        from content_extractor.adapters.image import ImageExtractor

        _create_test_image(tmp_path / "media" / "test.jpg")
        _make_image_content_item(tmp_path)

        mock_client = self._mock_describe(mock_vision_response_json)
        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ):
            result = ImageExtractor().extract(tmp_path, config)

        assert result.quality.language == "zh"
        assert result.quality.word_count > 0
        assert result.quality.processing_time_seconds > 0.0

    def test_extract_no_images_raises(self, tmp_path: Path) -> None:
        """extract() with no image files should raise ImageExtractionError."""
        from content_extractor.adapters.image import ImageExtractor
        from content_extractor.vision import ImageExtractionError

        _make_image_content_item(tmp_path, media_files=())
        config = ExtractorConfig()

        with pytest.raises(ImageExtractionError, match="No image files"):
            ImageExtractor().extract(tmp_path, config)

    def test_extract_error_isolation(
        self, tmp_path: Path, mock_vision_response_json: str
    ) -> None:
        """One bad image should not fail the whole extraction."""
        from content_extractor.adapters.image import ImageExtractor

        # Create one good image and one bad (empty file)
        _create_test_image(tmp_path / "media" / "good.jpg")
        (tmp_path / "media").mkdir(parents=True, exist_ok=True)
        (tmp_path / "media" / "bad.jpg").write_bytes(b"not an image")
        _make_image_content_item(
            tmp_path, media_files=("media/good.jpg", "media/bad.jpg")
        )

        mock_client = self._mock_describe(mock_vision_response_json)
        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ):
            result = ImageExtractor().extract(tmp_path, config)

        # Should have 2 descriptions: one good, one with error fallback
        assert len(result.media_descriptions) == 2
        good = [d for d in result.media_descriptions if d.confidence > 0]
        bad = [d for d in result.media_descriptions if d.confidence == 0.0]
        assert len(good) == 1
        assert len(bad) == 1

    def test_extract_protocol_satisfaction(self) -> None:
        """ImageExtractor should satisfy the Extractor Protocol."""
        from content_extractor.adapters.base import Extractor
        from content_extractor.adapters.image import ImageExtractor

        assert isinstance(ImageExtractor(), Extractor)

    def test_extract_glob_fallback(
        self, tmp_path: Path, mock_vision_response_json: str
    ) -> None:
        """If media_files is empty, extract() should glob media/ for images."""
        from content_extractor.adapters.image import ImageExtractor

        _create_test_image(tmp_path / "media" / "photo.jpg")
        _make_image_content_item(tmp_path, media_files=())

        mock_client = self._mock_describe(mock_vision_response_json)
        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ):
            result = ImageExtractor().extract(tmp_path, config)

        assert len(result.media_descriptions) == 1
