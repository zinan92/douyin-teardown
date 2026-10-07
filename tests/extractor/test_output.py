"""Tests for atomic output writer with idempotency guard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import orjson
import pytest

from content_extractor.models import (
    AnalysisResult,
    ContentItem,
    ExtractionResult,
    SentimentResult,
    Transcript,
    TranscriptSegment,
)
from content_extractor.output import (
    clear_marker,
    is_extracted,
    mark_complete,
    write_extraction_output,
    write_json_atomic,
    write_text_atomic,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def content_dir(tmp_path: Path) -> Path:
    """Return a temporary content directory."""
    d = tmp_path / "douyin" / "author1" / "content123"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def sample_extraction_result() -> ExtractionResult:
    """Minimal ExtractionResult for testing output."""
    return ExtractionResult(
        content_id="content123",
        content_type="video",
        raw_text="Hello world, this is a test transcript.",
        transcript=Transcript(
            content_id="content123",
            content_type="video",
            language="en",
            segments=(
                TranscriptSegment(
                    text="Hello world,", start=0.0, end=1.5, confidence=0.95
                ),
                TranscriptSegment(
                    text="this is a test transcript.",
                    start=1.5,
                    end=3.0,
                    confidence=0.92,
                ),
            ),
            full_text="Hello world, this is a test transcript.",
        ),
    )


@pytest.fixture()
def sample_content_item() -> ContentItem:
    """Minimal ContentItem for testing output."""
    return ContentItem(
        platform="douyin",
        content_id="content123",
        content_type="video",
        title="Test Video Title",
        description="A test video",
        author_id="author1",
        author_name="Test Author",
        publish_time="2026-03-30T00:00:00Z",
        source_url="https://example.com/video/123",
        media_files=("media/video.mp4",),
        likes=100,
        comments=20,
        shares=5,
        downloaded_at="2026-03-30T01:00:00Z",
    )


# ---------------------------------------------------------------------------
# write_json_atomic
# ---------------------------------------------------------------------------


class TestWriteJsonAtomic:
    def test_write_json_atomic(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        data = orjson.dumps({"key": "value"}, option=orjson.OPT_INDENT_2)
        write_json_atomic(target, data)

        assert target.exists()
        parsed = orjson.loads(target.read_bytes())
        assert parsed == {"key": "value"}

    def test_write_json_atomic_cleanup_on_failure(self, tmp_path: Path) -> None:
        """QUAL-03: No .tmp file left behind on failure."""
        target = tmp_path / "test.json"
        tmp_file = target.with_suffix(".json.tmp")

        with patch.object(Path, "write_bytes", side_effect=IOError("disk full")):
            with pytest.raises(IOError, match="disk full"):
                write_json_atomic(target, b'{"key": "value"}')

        assert not tmp_file.exists(), "Temp file should be cleaned up on failure"
        assert not target.exists(), "Target file should not exist on failure"


# ---------------------------------------------------------------------------
# write_text_atomic
# ---------------------------------------------------------------------------


class TestWriteTextAtomic:
    def test_write_text_atomic(self, tmp_path: Path) -> None:
        target = tmp_path / "test.md"
        write_text_atomic(target, "# Hello\n\nWorld")

        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert content == "# Hello\n\nWorld"

    def test_write_text_atomic_cleanup_on_failure(self, tmp_path: Path) -> None:
        """QUAL-03: No .tmp file left behind on failure."""
        target = tmp_path / "test.md"
        tmp_file = target.with_suffix(".md.tmp")

        with patch.object(Path, "write_text", side_effect=IOError("disk full")):
            with pytest.raises(IOError, match="disk full"):
                write_text_atomic(target, "some text")

        assert not tmp_file.exists(), "Temp file should be cleaned up on failure"
        assert not target.exists(), "Target file should not exist on failure"


# ---------------------------------------------------------------------------
# Idempotency marker
# ---------------------------------------------------------------------------


class TestIdempotencyMarker:
    def test_is_extracted_false(self, content_dir: Path) -> None:
        assert is_extracted(content_dir) is False

    def test_is_extracted_true(self, content_dir: Path) -> None:
        (content_dir / ".extraction_complete").touch()
        assert is_extracted(content_dir) is True

    def test_mark_complete(self, content_dir: Path) -> None:
        mark_complete(content_dir)
        assert (content_dir / ".extraction_complete").exists()

    def test_clear_marker(self, content_dir: Path) -> None:
        mark_complete(content_dir)
        clear_marker(content_dir)
        assert not (content_dir / ".extraction_complete").exists()

    def test_clear_marker_missing_ok(self, content_dir: Path) -> None:
        """No error when clearing a non-existent marker."""
        clear_marker(content_dir)  # should not raise


# ---------------------------------------------------------------------------
# write_extraction_output
# ---------------------------------------------------------------------------


class TestWriteExtractionOutput:
    def test_idempotency_skip(
        self,
        content_dir: Path,
        sample_extraction_result: ExtractionResult,
        sample_content_item: ContentItem,
    ) -> None:
        """FOUND-05: Second call returns False (already extracted)."""
        result1 = write_extraction_output(
            content_dir, sample_extraction_result, sample_content_item
        )
        result2 = write_extraction_output(
            content_dir, sample_extraction_result, sample_content_item
        )
        assert result1 is True
        assert result2 is False

    def test_idempotency_force(
        self,
        content_dir: Path,
        sample_extraction_result: ExtractionResult,
        sample_content_item: ContentItem,
    ) -> None:
        """FOUND-05: force=True re-processes even if already extracted."""
        write_extraction_output(
            content_dir, sample_extraction_result, sample_content_item
        )
        result = write_extraction_output(
            content_dir, sample_extraction_result, sample_content_item, force=True
        )
        assert result is True

    def test_write_extraction_output_files(
        self,
        content_dir: Path,
        sample_extraction_result: ExtractionResult,
        sample_content_item: ContentItem,
    ) -> None:
        """All three output files plus marker are created."""
        write_extraction_output(
            content_dir, sample_extraction_result, sample_content_item
        )

        assert (content_dir / "transcript.json").exists()
        assert (content_dir / "analysis.json").exists()
        assert (content_dir / "structured_text.md").exists()
        assert (content_dir / ".extraction_complete").exists()

    def test_transcript_json_valid(
        self,
        content_dir: Path,
        sample_extraction_result: ExtractionResult,
        sample_content_item: ContentItem,
    ) -> None:
        """transcript.json contains valid JSON with expected fields."""
        write_extraction_output(
            content_dir, sample_extraction_result, sample_content_item
        )
        data = orjson.loads((content_dir / "transcript.json").read_bytes())
        assert data["content_id"] == "content123"
        assert data["content_type"] == "video"
        assert data["full_text"] == "Hello world, this is a test transcript."

    def test_analysis_json_valid(
        self,
        content_dir: Path,
        sample_extraction_result: ExtractionResult,
        sample_content_item: ContentItem,
    ) -> None:
        """analysis.json contains valid JSON placeholder."""
        write_extraction_output(
            content_dir, sample_extraction_result, sample_content_item
        )
        data = orjson.loads((content_dir / "analysis.json").read_bytes())
        assert data["content_id"] == "content123"
        assert data["content_type"] == "video"

    def test_structured_text_format(
        self,
        content_dir: Path,
        sample_extraction_result: ExtractionResult,
        sample_content_item: ContentItem,
    ) -> None:
        """D-04: structured_text.md follows report format."""
        write_extraction_output(
            content_dir, sample_extraction_result, sample_content_item
        )
        md = (content_dir / "structured_text.md").read_text(encoding="utf-8")

        assert md.startswith("# Test Video Title")
        assert "## Summary" in md
        assert "## Key Takeaways" in md
        assert "## Full Transcript/Content" in md
        assert "## Analysis" in md
        assert "Test Author" in md
        assert "douyin" in md
        assert "Hello world, this is a test transcript." in md


# ---------------------------------------------------------------------------
# Analysis integration tests (Phase 8)
# ---------------------------------------------------------------------------


@pytest.fixture()
def filled_analysis() -> AnalysisResult:
    """An AnalysisResult with real data for testing output rendering."""
    return AnalysisResult(
        content_id="content123",
        content_type="video",
        topics=("AI regulation", "tech policy", "EU compliance"),
        viewpoints=("Regulation is necessary", "Over-regulation stifles innovation"),
        sentiment=SentimentResult(overall="mixed", confidence=0.82),
        takeaways=("Monitor EU AI Act timelines", "Prepare compliance roadmap"),
    )


@pytest.fixture()
def empty_analysis() -> AnalysisResult:
    """An AnalysisResult with no data (fallback/placeholder)."""
    return AnalysisResult(
        content_id="content123",
        content_type="video",
    )


class TestWriteExtractionOutputWithAnalysis:
    """write_extraction_output with real AnalysisResult data."""

    def test_analysis_json_with_real_data(
        self,
        content_dir: Path,
        sample_extraction_result: ExtractionResult,
        sample_content_item: ContentItem,
        filled_analysis: AnalysisResult,
    ) -> None:
        """Test 1: analysis.json contains real topics, viewpoints, sentiment, takeaways."""
        write_extraction_output(
            content_dir,
            sample_extraction_result,
            sample_content_item,
            analysis=filled_analysis,
        )
        data = orjson.loads((content_dir / "analysis.json").read_bytes())
        assert data["content_id"] == "content123"
        assert data["topics"] == ["AI regulation", "tech policy", "EU compliance"]
        assert len(data["viewpoints"]) == 2
        assert data["sentiment"]["overall"] == "mixed"
        assert data["sentiment"]["confidence"] == pytest.approx(0.82)
        assert len(data["takeaways"]) == 2

    def test_structured_text_with_real_analysis(
        self,
        content_dir: Path,
        sample_extraction_result: ExtractionResult,
        sample_content_item: ContentItem,
        filled_analysis: AnalysisResult,
    ) -> None:
        """Test 2: structured_text.md renders real Summary, Key Takeaways, Analysis."""
        write_extraction_output(
            content_dir,
            sample_extraction_result,
            sample_content_item,
            analysis=filled_analysis,
        )
        md = (content_dir / "structured_text.md").read_text(encoding="utf-8")

        # Summary section has real topics
        assert "AI regulation" in md
        assert "tech policy" in md

        # Key Takeaways are rendered as bullets
        assert "- Monitor EU AI Act timelines" in md
        assert "- Prepare compliance roadmap" in md

        # Analysis section has topics, viewpoints, sentiment
        assert "**Topics:**" in md
        assert "**Viewpoints:**" in md
        assert "- Regulation is necessary" in md
        assert "**Sentiment:** Overall: mixed (confidence: 0.82)" in md

        # No placeholder text
        assert "Populated by analysis phase" not in md

    def test_structured_text_with_empty_analysis(
        self,
        content_dir: Path,
        sample_extraction_result: ExtractionResult,
        sample_content_item: ContentItem,
        empty_analysis: AnalysisResult,
    ) -> None:
        """Test 3: empty AnalysisResult renders valid markdown without crash."""
        write_extraction_output(
            content_dir,
            sample_extraction_result,
            sample_content_item,
            analysis=empty_analysis,
        )
        md = (content_dir / "structured_text.md").read_text(encoding="utf-8")

        assert "## Summary" in md
        assert "No analysis available." in md
        assert "No takeaways identified." in md
        assert "## Full Transcript/Content" in md

    def test_backward_compatible_no_analysis(
        self,
        content_dir: Path,
        sample_extraction_result: ExtractionResult,
        sample_content_item: ContentItem,
    ) -> None:
        """Test 4: when analysis=None (default), writes placeholder."""
        write_extraction_output(
            content_dir,
            sample_extraction_result,
            sample_content_item,
        )
        data = orjson.loads((content_dir / "analysis.json").read_bytes())
        assert data["content_id"] == "content123"
        assert data["topics"] == []  # empty placeholder
        assert data["viewpoints"] == []
        assert data["sentiment"] is None
        assert data["takeaways"] == []

        md = (content_dir / "structured_text.md").read_text(encoding="utf-8")
        assert "No analysis available." in md
