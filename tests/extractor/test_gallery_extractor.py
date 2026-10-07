"""Tests for GalleryExtractor adapter — batched vision + narrative synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from content_extractor.config import ExtractorConfig


def _make_gallery_content_item(
    content_dir: Path,
    *,
    media_files: tuple[str, ...] = ("media/img1.jpg", "media/img2.jpg", "media/img3.jpg"),
) -> None:
    """Create a content_item.json for gallery testing."""
    item = {
        "platform": "xiaohongshu",
        "content_id": "gallery_001",
        "content_type": "gallery",
        "title": "Gallery Post",
        "description": "A test gallery post",
        "author_id": "xhs_user_1",
        "author_name": "TestUser",
        "publish_time": "2026-03-30T10:00:00+00:00",
        "source_url": "https://www.xiaohongshu.com/explore/gallery_001",
        "media_files": list(media_files),
        "cover_file": None,
        "metadata_file": "metadata.json",
        "likes": 200,
        "comments": 20,
        "shares": 10,
        "collects": 100,
        "views": 2000,
        "downloaded_at": "2026-03-30T12:00:00Z",
    }
    (content_dir / "content_item.json").write_text(json.dumps(item))


def _create_test_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    """Create a small solid-color test image."""
    img = Image.new("RGB", size, color=(255, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG")


def _make_mock_vision_response(
    ocr_text: str = "Some text",
    visual_description: str = "A test image",
    confidence: float = 0.9,
) -> str:
    """Return a JSON string mimicking Claude vision response."""
    return json.dumps({
        "ocr_text": ocr_text,
        "visual_description": visual_description,
        "confidence": confidence,
    })


def _make_mock_client(vision_json: str, narrative_text: str = "A narrative.") -> MagicMock:
    """Create a mock Anthropic client that handles both vision and narrative calls."""
    mock_client = MagicMock()

    vision_block = SimpleNamespace(type="text", text=vision_json)
    vision_resp = MagicMock()
    vision_resp.content = [vision_block]

    narrative_block = SimpleNamespace(type="text", text=narrative_text)
    narrative_resp = MagicMock()
    narrative_resp.content = [narrative_block]

    # First N calls are vision, last call is narrative synthesis
    # We'll use side_effect in individual tests for fine control
    mock_client.messages.create.return_value = vision_resp

    return mock_client


class TestGalleryThreeImages:
    """Test 1: Gallery with 3 images produces 3 MediaDescription entries and a narrative."""

    def test_three_images_produces_three_descriptions_and_narrative(
        self, tmp_path: Path
    ) -> None:
        from content_extractor.adapters.gallery import GalleryExtractor

        # Create 3 images
        for name in ("img1.jpg", "img2.jpg", "img3.jpg"):
            _create_test_image(tmp_path / "media" / name)
        _make_gallery_content_item(tmp_path)

        vision_json = _make_mock_vision_response(
            ocr_text="Chinese text",
            visual_description="A photo of food",
            confidence=0.85,
        )
        narrative_text = "This gallery shows a food journey across three images."

        vision_block = SimpleNamespace(type="text", text=vision_json)
        vision_resp = MagicMock()
        vision_resp.content = [vision_block]

        narrative_block = SimpleNamespace(type="text", text=narrative_text)
        narrative_resp = MagicMock()
        narrative_resp.content = [narrative_block]

        mock_client = MagicMock()
        # 3 vision calls + 1 narrative call
        mock_client.messages.create.side_effect = [
            vision_resp, vision_resp, vision_resp, narrative_resp
        ]

        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.time.sleep",
        ):
            result = GalleryExtractor().extract(tmp_path, config)

        assert result.content_type == "gallery"
        assert len(result.media_descriptions) == 3
        assert result.raw_text == narrative_text
        for desc in result.media_descriptions:
            assert desc.confidence == pytest.approx(0.85)


class TestGalleryBatching:
    """Test 2: Gallery with 7 images makes 2 batches (5 + 2)."""

    def test_seven_images_batched(self, tmp_path: Path) -> None:
        from content_extractor.adapters.gallery import GalleryExtractor

        filenames = [f"img{i}.jpg" for i in range(7)]
        for name in filenames:
            _create_test_image(tmp_path / "media" / name)
        _make_gallery_content_item(
            tmp_path,
            media_files=tuple(f"media/{n}" for n in filenames),
        )

        vision_json = _make_mock_vision_response()
        narrative_text = "A seven-image gallery narrative."

        vision_block = SimpleNamespace(type="text", text=vision_json)
        vision_resp = MagicMock()
        vision_resp.content = [vision_block]

        narrative_block = SimpleNamespace(type="text", text=narrative_text)
        narrative_resp = MagicMock()
        narrative_resp.content = [narrative_block]

        mock_client = MagicMock()
        # 7 vision calls + 1 narrative call
        mock_client.messages.create.side_effect = [vision_resp] * 7 + [narrative_resp]

        config = ExtractorConfig()

        sleep_mock = MagicMock()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.time.sleep",
            sleep_mock,
        ):
            result = GalleryExtractor().extract(tmp_path, config)

        assert len(result.media_descriptions) == 7
        assert result.raw_text == narrative_text
        # Should sleep between batches (after batch 1 of 5, before batch 2 of 2)
        sleep_mock.assert_called()


class TestGalleryOneImageFails:
    """Test 3: One image fails -> fallback MediaDescription, gallery still succeeds."""

    def test_one_failure_does_not_break_gallery(self, tmp_path: Path) -> None:
        from content_extractor.adapters.gallery import GalleryExtractor

        for name in ("img1.jpg", "img2.jpg", "img3.jpg"):
            _create_test_image(tmp_path / "media" / name)
        _make_gallery_content_item(tmp_path)

        vision_json = _make_mock_vision_response(confidence=0.9)

        vision_block = SimpleNamespace(type="text", text=vision_json)
        vision_resp = MagicMock()
        vision_resp.content = [vision_block]

        narrative_block = SimpleNamespace(type="text", text="Narrative with partial data.")
        narrative_resp = MagicMock()
        narrative_resp.content = [narrative_block]

        mock_client = MagicMock()
        # First image succeeds, second fails, third succeeds, then narrative
        mock_client.messages.create.side_effect = [
            vision_resp,
            Exception("API error for image 2"),
            vision_resp,
            narrative_resp,
        ]

        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.time.sleep",
        ):
            result = GalleryExtractor().extract(tmp_path, config)

        assert len(result.media_descriptions) == 3
        # Second image should have confidence=0.0 (fallback)
        assert result.media_descriptions[1].confidence == 0.0
        assert result.media_descriptions[1].description == ""
        # Other images should have real confidence
        assert result.media_descriptions[0].confidence == pytest.approx(0.9)
        assert result.media_descriptions[2].confidence == pytest.approx(0.9)


class TestGalleryAllImagesFail:
    """Test 4: All images fail -> empty narrative, all MediaDescriptions at confidence=0.0."""

    def test_all_failures_produces_empty_narrative(self, tmp_path: Path) -> None:
        from content_extractor.adapters.gallery import GalleryExtractor

        for name in ("img1.jpg", "img2.jpg"):
            _create_test_image(tmp_path / "media" / name)
        _make_gallery_content_item(
            tmp_path,
            media_files=("media/img1.jpg", "media/img2.jpg"),
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("All calls fail")

        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.time.sleep",
        ):
            result = GalleryExtractor().extract(tmp_path, config)

        assert result.raw_text == ""
        assert len(result.media_descriptions) == 2
        for desc in result.media_descriptions:
            assert desc.confidence == 0.0
            assert desc.description == ""


class TestGalleryNarrativeSynthesis:
    """Test 5: Narrative synthesis combines per-image descriptions."""

    def test_narrative_combines_descriptions(self, tmp_path: Path) -> None:
        from content_extractor.adapters.gallery import GalleryExtractor

        for name in ("img1.jpg", "img2.jpg"):
            _create_test_image(tmp_path / "media" / name)
        _make_gallery_content_item(
            tmp_path,
            media_files=("media/img1.jpg", "media/img2.jpg"),
        )

        vision_json = _make_mock_vision_response(
            ocr_text="Food review",
            visual_description="Delicious sushi plate",
        )

        vision_block = SimpleNamespace(type="text", text=vision_json)
        vision_resp = MagicMock()
        vision_resp.content = [vision_block]

        expected_narrative = (
            "This gallery presents a food review featuring delicious sushi plates "
            "across two images. The reviewer shares their dining experience."
        )
        narrative_block = SimpleNamespace(type="text", text=expected_narrative)
        narrative_resp = MagicMock()
        narrative_resp.content = [narrative_block]

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [vision_resp, vision_resp, narrative_resp]

        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.time.sleep",
        ):
            result = GalleryExtractor().extract(tmp_path, config)

        assert result.raw_text == expected_narrative
        # Verify the narrative call received image descriptions
        # The last call should be the narrative synthesis
        last_call = mock_client.messages.create.call_args_list[-1]
        messages = last_call.kwargs.get("messages", [])
        # Find user message content
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        user_content += block["text"]
                    elif isinstance(block, str):
                        user_content += block
        assert "Delicious sushi plate" in user_content or "sushi" in user_content.lower()


class TestGalleryEmptyRaises:
    """Test 6: Empty gallery (no images found) raises ImageExtractionError."""

    def test_empty_gallery_raises(self, tmp_path: Path) -> None:
        from content_extractor.adapters.gallery import GalleryExtractor
        from content_extractor.vision import ImageExtractionError

        _make_gallery_content_item(tmp_path, media_files=())
        config = ExtractorConfig()

        with pytest.raises(ImageExtractionError, match="No image files"):
            GalleryExtractor().extract(tmp_path, config)


class TestGalleryQualityMetadata:
    """Test 7: Quality metadata has correct avg confidence, language, word count, processing time."""

    def test_quality_metadata_correct(self, tmp_path: Path) -> None:
        from content_extractor.adapters.gallery import GalleryExtractor

        for name in ("img1.jpg", "img2.jpg"):
            _create_test_image(tmp_path / "media" / name)
        _make_gallery_content_item(
            tmp_path,
            media_files=("media/img1.jpg", "media/img2.jpg"),
        )

        vision_json_1 = _make_mock_vision_response(confidence=0.8)
        vision_json_2 = _make_mock_vision_response(confidence=0.6)

        vision_block_1 = SimpleNamespace(type="text", text=vision_json_1)
        vision_resp_1 = MagicMock()
        vision_resp_1.content = [vision_block_1]

        vision_block_2 = SimpleNamespace(type="text", text=vision_json_2)
        vision_resp_2 = MagicMock()
        vision_resp_2.content = [vision_block_2]

        narrative = "This is a narrative with some Chinese: delicious food review."
        narrative_block = SimpleNamespace(type="text", text=narrative)
        narrative_resp = MagicMock()
        narrative_resp.content = [narrative_block]

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            vision_resp_1, vision_resp_2, narrative_resp
        ]

        config = ExtractorConfig()

        with patch(
            "content_extractor.vision.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.create_claude_client",
            return_value=mock_client,
        ), patch(
            "content_extractor.adapters.gallery.time.sleep",
        ):
            result = GalleryExtractor().extract(tmp_path, config)

        # Avg confidence: (0.8 + 0.6) / 2 = 0.7
        assert result.quality.confidence == pytest.approx(0.7)
        assert result.quality.word_count > 0
        assert result.quality.processing_time_seconds > 0.0
        # Language detected from narrative
        assert result.quality.language in ("en", "zh", "unknown")
