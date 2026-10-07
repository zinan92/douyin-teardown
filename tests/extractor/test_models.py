"""Tests for all Pydantic data models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from content_extractor.models import (
    AnalysisResult,
    ContentItem,
    ExtractionResult,
    MediaDescription,
    QualityMetadata,
    SentimentResult,
    Transcript,
    TranscriptSegment,
)


class TestContentItem:
    """Tests for ContentItem model."""

    def test_content_item_valid_creation(
        self, sample_content_item_dict: dict[str, Any]
    ) -> None:
        item = ContentItem.model_validate(sample_content_item_dict)

        assert item.platform == "douyin"
        assert item.content_id == "7616648694510652672"
        assert item.content_type == "video"
        assert item.title == "Agent架构选型指南"
        assert item.author_name == "慢学AI"
        assert item.likes == 10943
        assert item.views == 0

    def test_content_item_frozen(
        self, sample_content_item_dict: dict[str, Any]
    ) -> None:
        item = ContentItem.model_validate(sample_content_item_dict)

        with pytest.raises(ValidationError):
            item.title = "new title"

    def test_content_item_extra_fields_ignored(
        self, sample_content_item_dict: dict[str, Any]
    ) -> None:
        data = {**sample_content_item_dict, "new_field": "should be ignored"}
        item = ContentItem.model_validate(data)

        assert not hasattr(item, "new_field")
        assert item.platform == "douyin"

    def test_content_item_media_files_immutable(
        self, sample_content_item_dict: dict[str, Any]
    ) -> None:
        item = ContentItem.model_validate(sample_content_item_dict)

        assert isinstance(item.media_files, tuple)
        assert item.media_files == ("media/video.mp4",)

    def test_content_item_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ContentItem.model_validate({"platform": "douyin"})

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "content_id" in missing_fields
        assert "title" in missing_fields


class TestTranscript:
    """Tests for Transcript and TranscriptSegment models."""

    def test_transcript_segment_creation(self) -> None:
        segment = TranscriptSegment(
            text="Hello world",
            start=0.0,
            end=5.0,
            confidence=0.95,
        )

        assert segment.text == "Hello world"
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.confidence == 0.95

        with pytest.raises(ValidationError):
            segment.text = "changed"

    def test_transcript_creation(self) -> None:
        segment = TranscriptSegment(
            text="Hello", start=0.0, end=2.5, confidence=0.9
        )
        transcript = Transcript(
            content_id="test-123",
            content_type="video",
            language="zh",
            segments=(segment,),
            full_text="Hello",
        )

        assert transcript.content_id == "test-123"
        assert transcript.language == "zh"
        assert transcript.schema_version == "1.0"
        assert len(transcript.segments) == 1


class TestAnalysisResult:
    """Tests for AnalysisResult with fixed schema (D-03)."""

    def test_analysis_result_fixed_schema(self) -> None:
        analysis = AnalysisResult(
            content_id="test-123",
            content_type="video",
            topics=("AI", "architecture"),
            viewpoints=("multi-agent is better",),
            sentiment=SentimentResult(overall="positive", confidence=0.85),
            takeaways=("Use multi-agent for complex tasks",),
        )

        assert isinstance(analysis.topics, tuple)
        assert isinstance(analysis.viewpoints, tuple)
        assert isinstance(analysis.takeaways, tuple)
        assert analysis.sentiment is not None
        assert analysis.sentiment.overall == "positive"
        assert analysis.schema_version == "1.0"


class TestQualityMetadata:
    """Tests for QualityMetadata defaults (QUAL-02)."""

    def test_quality_metadata_defaults(self) -> None:
        quality = QualityMetadata()

        assert quality.confidence == 0.0
        assert quality.language == "unknown"
        assert quality.word_count == 0
        assert quality.processing_time_seconds == 0.0


class TestExtractionResult:
    """Tests for ExtractionResult."""

    def test_platform_metadata_preserved(self) -> None:
        """Platform metadata (author, engagement) is preserved (QUAL-04)."""
        result = ExtractionResult(
            content_id="test-123",
            content_type="video",
            raw_text="Some extracted text",
            platform_metadata={
                "author_name": "慢学AI",
                "likes": 10943,
                "comments": 217,
                "shares": 1970,
                "collects": 9786,
                "views": 0,
            },
        )

        assert result.platform_metadata["author_name"] == "慢学AI"
        assert result.platform_metadata["likes"] == 10943
        assert result.platform_metadata["comments"] == 217
        assert result.platform_metadata["shares"] == 1970

    def test_extraction_result_complete(self) -> None:
        """ExtractionResult with all fields populated."""
        segment = TranscriptSegment(
            text="Hello", start=0.0, end=2.5, confidence=0.9
        )
        transcript = Transcript(
            content_id="test-123",
            content_type="video",
            language="zh",
            segments=(segment,),
            full_text="Hello",
        )
        media_desc = MediaDescription(
            file_path="media/cover.jpg",
            description="A cover image",
            ocr_text="Some text",
            confidence=0.8,
        )
        quality = QualityMetadata(
            confidence=0.92,
            language="zh",
            word_count=150,
            processing_time_seconds=3.5,
        )
        result = ExtractionResult(
            content_id="test-123",
            content_type="video",
            raw_text="Hello",
            transcript=transcript,
            media_descriptions=(media_desc,),
            quality=quality,
            platform_metadata={"author_name": "test", "likes": 100},
        )

        assert result.transcript is not None
        assert result.transcript.language == "zh"
        assert len(result.media_descriptions) == 1
        assert result.quality.confidence == 0.92
        assert result.schema_version == "1.0"

        with pytest.raises(ValidationError):
            result.raw_text = "changed"


class TestTranscriptSegmentSuspicious:
    """Tests for TranscriptSegment.is_suspicious field."""

    def test_default_not_suspicious(self) -> None:
        """is_suspicious defaults to False."""
        seg = TranscriptSegment(
            text="text", start=0.0, end=1.0, confidence=0.8
        )
        assert seg.is_suspicious is False

    def test_can_set_suspicious(self) -> None:
        """is_suspicious can be set to True."""
        seg = TranscriptSegment(
            text="text", start=0.0, end=1.0, confidence=0.8,
            is_suspicious=True,
        )
        assert seg.is_suspicious is True

    def test_model_copy_with_suspicious(self) -> None:
        """model_copy can update is_suspicious on frozen model."""
        seg = TranscriptSegment(
            text="text", start=0.0, end=1.0, confidence=0.8
        )
        flagged = seg.model_copy(update={"is_suspicious": True})
        assert flagged.is_suspicious is True
        assert seg.is_suspicious is False


class TestQualityMetadataHallucination:
    """Tests for QualityMetadata.hallucination_warnings field."""

    def test_default_empty_warnings(self) -> None:
        """hallucination_warnings defaults to empty tuple."""
        meta = QualityMetadata()
        assert meta.hallucination_warnings == ()

    def test_can_set_warnings(self) -> None:
        """hallucination_warnings can be populated."""
        meta = QualityMetadata(
            hallucination_warnings=("warning 1", "warning 2")
        )
        assert len(meta.hallucination_warnings) == 2
        assert meta.hallucination_warnings[0] == "warning 1"
