"""Tests for extract_content and extract_batch orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from content_extractor.analysis import AnalysisError
from content_extractor.config import ExtractorConfig
from content_extractor.extract import BatchError, BatchResult, extract_batch, extract_content
from content_extractor.models import AnalysisResult, ExtractionResult, QualityMetadata, SentimentResult


def _make_content_item_json(content_dir: Path, content_type: str = "video") -> None:
    """Helper: write a minimal content_item.json into content_dir."""
    content_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "platform": "test",
        "content_id": content_dir.name,
        "content_type": content_type,
        "title": "Test",
        "description": "Test content",
        "author_id": "author1",
        "author_name": "Author",
        "publish_time": "2026-01-01T00:00:00Z",
        "source_url": "https://example.com",
        "downloaded_at": "2026-01-01T00:00:00Z",
    }
    (content_dir / "content_item.json").write_text(json.dumps(data))


def _make_extraction_result(content_id: str = "test") -> ExtractionResult:
    """Helper: build a minimal ExtractionResult."""
    return ExtractionResult(
        content_id=content_id,
        content_type="video",
        raw_text="Hello world",
        quality=QualityMetadata(confidence=0.9, language="en", word_count=2),
    )


class TestExtractContentSkip:
    """extract_content skips already-extracted items unless force=True."""

    def test_skips_extracted(self, tmp_path: Path) -> None:
        content_dir = tmp_path / "item1"
        _make_content_item_json(content_dir)
        # Create the extraction marker (matches output.MARKER_FILE)
        (content_dir / ".extraction_complete").touch()

        result = extract_content(content_dir)
        assert result is None

    def test_force_overrides_skip(self, tmp_path: Path) -> None:
        content_dir = tmp_path / "item1"
        _make_content_item_json(content_dir)
        (content_dir / ".extraction_complete").touch()

        config = ExtractorConfig(force_reprocess=True)
        expected = _make_extraction_result("item1")

        mock_adapter = MagicMock()
        mock_adapter.extract.return_value = expected

        # With force, it should NOT skip -- it will route to the adapter
        with (
            patch("content_extractor.extract.get_extractor", return_value=mock_adapter),
            patch("content_extractor.extract.write_extraction_output"),
        ):
            result = extract_content(content_dir, config)

        assert result is not None
        assert result.content_id == "item1"
        mock_adapter.extract.assert_called_once()


class TestExtractContentPipeline:
    """extract_content wires loader -> router -> adapter -> output."""

    def test_full_pipeline_with_mock_adapter(self, tmp_path: Path) -> None:
        content_dir = tmp_path / "item1"
        _make_content_item_json(content_dir)
        expected = _make_extraction_result("item1")

        mock_adapter = MagicMock()
        mock_adapter.extract.return_value = expected

        with (
            patch("content_extractor.extract.get_extractor", return_value=mock_adapter),
            patch("content_extractor.extract.write_extraction_output", return_value=True),
        ):
            result = extract_content(content_dir)

        assert result is not None
        assert result.content_id == "item1"
        assert result.raw_text == "Hello world"
        mock_adapter.extract.assert_called_once()

    def test_adapter_error_propagates(self, tmp_path: Path) -> None:
        """Adapter errors propagate to caller when not in batch mode."""
        content_dir = tmp_path / "item1"
        _make_content_item_json(content_dir)

        mock_adapter = MagicMock()
        mock_adapter.extract.side_effect = RuntimeError("adapter failed")

        with patch("content_extractor.extract.get_extractor", return_value=mock_adapter):
            with pytest.raises(RuntimeError, match="adapter failed"):
                extract_content(content_dir)


class TestExtractBatchErrorIsolation:
    """extract_batch isolates per-item errors (D-07, QUAL-01)."""

    def test_batch_error_isolation(self, tmp_path: Path) -> None:
        """All items are processed even when some fail."""
        for i in range(3):
            _make_content_item_json(tmp_path / f"item{i}")

        mock_adapter = MagicMock()
        mock_adapter.extract.side_effect = RuntimeError("adapter error")

        with patch("content_extractor.extract.get_extractor", return_value=mock_adapter):
            result = extract_batch(tmp_path)

        assert result.total == 3
        assert result.failure_count == 3
        assert result.success_count == 0
        assert len(result.failed) == 3
        # None aborted the batch
        for err in result.failed:
            assert isinstance(err, BatchError)
            assert err.error  # error message is not empty

    def test_batch_continues_after_failure(self, tmp_path: Path) -> None:
        """A failing item does not prevent processing subsequent items."""
        # item0 has invalid JSON
        bad_dir = tmp_path / "item0"
        bad_dir.mkdir(parents=True)
        (bad_dir / "content_item.json").write_text("NOT JSON")

        # item1 and item2 have valid JSON but adapter raises
        _make_content_item_json(tmp_path / "item1")
        _make_content_item_json(tmp_path / "item2")

        mock_adapter = MagicMock()
        mock_adapter.extract.side_effect = RuntimeError("adapter error")

        with patch("content_extractor.extract.get_extractor", return_value=mock_adapter):
            result = extract_batch(tmp_path)

        assert result.total == 3
        assert result.failure_count == 3  # all fail (bad JSON + adapter errors)
        assert len(result.failed) == 3

    def test_batch_empty_dir(self, tmp_path: Path) -> None:
        """Empty parent dir -> BatchResult with total=0."""
        result = extract_batch(tmp_path)

        assert result.total == 0
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.succeeded == ()
        assert result.failed == ()

    def test_batch_result_counts_match(self, tmp_path: Path) -> None:
        """total == success_count + failure_count."""
        for i in range(2):
            _make_content_item_json(tmp_path / f"item{i}")

        mock_adapter = MagicMock()
        mock_adapter.extract.side_effect = RuntimeError("adapter error")

        with patch("content_extractor.extract.get_extractor", return_value=mock_adapter):
            result = extract_batch(tmp_path)

        assert result.total == result.success_count + result.failure_count

    def test_batch_with_mixed_results(self, tmp_path: Path) -> None:
        """Batch with mocked success and real failures."""
        # item0 will succeed via mock, item1 will fail (stub)
        _make_content_item_json(tmp_path / "item0", content_type="gallery")
        _make_content_item_json(tmp_path / "item1", content_type="gallery")

        expected = _make_extraction_result("item0")
        mock_adapter = MagicMock()
        mock_adapter.extract.return_value = expected
        call_count = 0

        original_get_extractor = None

        def patched_get_extractor(content_type: str):  # noqa: ANN202
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_adapter
            raise RuntimeError("simulated adapter failure")

        with (
            patch("content_extractor.extract.get_extractor", side_effect=patched_get_extractor),
            patch("content_extractor.extract.write_extraction_output", return_value=True),
        ):
            result = extract_batch(tmp_path)

        assert result.total == 2
        assert result.success_count == 1
        assert result.failure_count == 1
        assert len(result.succeeded) == 1
        assert len(result.failed) == 1


def _make_analysis_result(content_id: str = "test") -> AnalysisResult:
    """Helper: build a populated AnalysisResult."""
    return AnalysisResult(
        content_id=content_id,
        content_type="video",
        topics=("AI", "regulation"),
        viewpoints=("AI is good",),
        sentiment=SentimentResult(overall="positive", confidence=0.9),
        takeaways=("Invest in AI",),
    )


class TestExtractContentAnalysisIntegration:
    """extract_content calls analyze_content() and passes result to output writer."""

    def test_calls_analyze_and_passes_to_output(self, tmp_path: Path) -> None:
        """Test 1: extract_content() calls analyze_content() after adapter."""
        content_dir = tmp_path / "item1"
        _make_content_item_json(content_dir)
        extraction = _make_extraction_result("item1")
        analysis = _make_analysis_result("item1")

        mock_adapter = MagicMock()
        mock_adapter.extract.return_value = extraction
        mock_write = MagicMock(return_value=True)
        mock_analyze = MagicMock(return_value=analysis)

        with (
            patch("content_extractor.extract.get_extractor", return_value=mock_adapter),
            patch("content_extractor.extract.write_extraction_output", mock_write),
            patch("content_extractor.extract.analyze_content", mock_analyze),
        ):
            result = extract_content(content_dir)

        assert result is not None
        mock_analyze.assert_called_once()
        # Verify analysis was passed to write_extraction_output
        call_kwargs = mock_write.call_args
        assert call_kwargs.kwargs.get("analysis") == analysis or (
            len(call_kwargs.args) > 3 and call_kwargs.args[3] == analysis
        )

    def test_analysis_error_produces_fallback(self, tmp_path: Path) -> None:
        """Test 2: AnalysisError is caught; extraction still succeeds with placeholder."""
        content_dir = tmp_path / "item1"
        _make_content_item_json(content_dir)
        extraction = _make_extraction_result("item1")

        mock_adapter = MagicMock()
        mock_adapter.extract.return_value = extraction
        mock_write = MagicMock(return_value=True)
        mock_analyze = MagicMock(side_effect=AnalysisError("LLM down"))

        with (
            patch("content_extractor.extract.get_extractor", return_value=mock_adapter),
            patch("content_extractor.extract.write_extraction_output", mock_write),
            patch("content_extractor.extract.analyze_content", mock_analyze),
        ):
            result = extract_content(content_dir)

        # Extraction should still succeed
        assert result is not None
        assert result.content_id == "item1"
        # write_extraction_output should still be called with a fallback analysis
        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args
        analysis_arg = call_kwargs.kwargs.get("analysis")
        assert analysis_arg is not None
        assert isinstance(analysis_arg, AnalysisResult)
        assert analysis_arg.content_id == "item1"
        assert analysis_arg.topics == ()  # fallback has empty fields

    def test_batch_isolates_analysis_errors(self, tmp_path: Path) -> None:
        """Test 3: analysis failure in one item does not affect others in batch."""
        _make_content_item_json(tmp_path / "item0")
        _make_content_item_json(tmp_path / "item1")

        extraction = _make_extraction_result("item0")
        analysis = _make_analysis_result("item0")

        mock_adapter = MagicMock()
        mock_adapter.extract.return_value = extraction
        mock_write = MagicMock(return_value=True)

        call_count = 0

        def analyze_side_effect(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return analysis
            raise AnalysisError("LLM down for item1")

        with (
            patch("content_extractor.extract.get_extractor", return_value=mock_adapter),
            patch("content_extractor.extract.write_extraction_output", mock_write),
            patch("content_extractor.extract.analyze_content", side_effect=analyze_side_effect),
        ):
            result = extract_batch(tmp_path)

        # Both items should succeed (analysis error caught gracefully)
        assert result.total == 2
        assert result.success_count == 2
        assert result.failure_count == 0


class TestBatchError:
    """BatchError captures content_dir path and error message."""

    def test_batch_error_fields(self) -> None:
        err = BatchError(content_dir="/path/to/item", error="something failed")
        assert err.content_dir == "/path/to/item"
        assert err.error == "something failed"

    def test_batch_error_is_frozen(self) -> None:
        err = BatchError(content_dir="/path", error="msg")
        with pytest.raises(AttributeError):
            err.content_dir = "/other"  # type: ignore[misc]
