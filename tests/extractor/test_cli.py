"""Tests for the Typer CLI module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from content_extractor.cli import app
from content_extractor.config import ExtractorConfig
from content_extractor.extract import BatchError, BatchResult
from content_extractor.models import ExtractionResult, QualityMetadata

runner = CliRunner()


@pytest.fixture()
def content_dir(tmp_path: Path) -> Path:
    """Create a fake content item directory with content_item.json."""
    item_dir = tmp_path / "platform" / "author" / "item1"
    item_dir.mkdir(parents=True)
    (item_dir / "content_item.json").write_text("{}")
    return item_dir


@pytest.fixture()
def batch_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple content items."""
    for name in ("item1", "item2", "item3"):
        item_dir = tmp_path / name
        item_dir.mkdir()
        (item_dir / "content_item.json").write_text("{}")
    return tmp_path


@pytest.fixture()
def sample_result() -> ExtractionResult:
    """A successful ExtractionResult for mocking."""
    return ExtractionResult(
        content_id="test-123",
        content_type="video",
        raw_text="Hello world",
        quality=QualityMetadata(
            word_count=2,
            processing_time_seconds=1.5,
            confidence=0.95,
            language="en",
        ),
    )


# ---------------------------------------------------------------------------
# extract command
# ---------------------------------------------------------------------------


class TestExtractCommand:
    """Tests for the `extract` CLI command."""

    def test_extract_valid_path_calls_extract_content(
        self, content_dir: Path, sample_result: ExtractionResult
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=sample_result) as mock:
            result = runner.invoke(app, ["extract", str(content_dir)])

        assert result.exit_code == 0
        mock.assert_called_once()
        call_args = mock.call_args
        assert call_args[0][0] == content_dir

    def test_extract_prints_result_summary(
        self, content_dir: Path, sample_result: ExtractionResult
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=sample_result):
            result = runner.invoke(app, ["extract", str(content_dir)])

        assert "test-123" in result.output
        assert "video" in result.output
        assert "2" in result.output  # word_count

    def test_extract_prints_already_extracted_when_none(
        self, content_dir: Path
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=None):
            result = runner.invoke(app, ["extract", str(content_dir)])

        assert result.exit_code == 0
        assert "Already extracted" in result.output

    def test_extract_force_flag(
        self, content_dir: Path, sample_result: ExtractionResult
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=sample_result) as mock:
            result = runner.invoke(app, ["extract", str(content_dir), "--force"])

        assert result.exit_code == 0
        config: ExtractorConfig = mock.call_args[0][1]
        assert config.force_reprocess is True

    def test_extract_whisper_model_flag(
        self, content_dir: Path, sample_result: ExtractionResult
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=sample_result) as mock:
            result = runner.invoke(app, ["extract", str(content_dir), "--whisper-model", "large-v3"])

        assert result.exit_code == 0
        config: ExtractorConfig = mock.call_args[0][1]
        assert config.whisper_model == "large-v3"

    def test_extract_exits_code_1_on_error(self, content_dir: Path) -> None:
        with patch("content_extractor.cli.extract_content", side_effect=RuntimeError("boom")):
            result = runner.invoke(app, ["extract", str(content_dir)])

        assert result.exit_code == 1
        assert "boom" in result.output

    def test_extract_invalid_path(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nope"
        result = runner.invoke(app, ["extract", str(nonexistent)])

        assert result.exit_code == 1
        assert "does not exist" in result.output or "not a directory" in result.output


# ---------------------------------------------------------------------------
# extract-batch command
# ---------------------------------------------------------------------------


class TestExtractBatchCommand:
    """Tests for the `extract-batch` CLI command."""

    def test_batch_calls_extract_content_for_each_item(
        self, batch_dir: Path, sample_result: ExtractionResult
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=sample_result) as mock:
            result = runner.invoke(app, ["extract-batch", str(batch_dir)])

        assert result.exit_code == 0
        assert mock.call_count == 3

    def test_batch_shows_progress_output(
        self, batch_dir: Path, sample_result: ExtractionResult
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=sample_result):
            result = runner.invoke(app, ["extract-batch", str(batch_dir)])

        # Should contain summary line with counts
        assert "3" in result.output  # total items
        assert "succeeded" in result.output.lower() or "Processed" in result.output

    def test_batch_error_summary_table_on_failures(
        self, batch_dir: Path, sample_result: ExtractionResult
    ) -> None:
        call_count = 0

        def mock_extract(path: Path, config: ExtractorConfig) -> ExtractionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("extraction failed")
            return sample_result

        with patch("content_extractor.cli.extract_content", side_effect=mock_extract):
            result = runner.invoke(app, ["extract-batch", str(batch_dir)])

        assert result.exit_code == 1
        assert "extraction failed" in result.output
        assert "1 failed" in result.output

    def test_batch_no_error_table_when_zero_failures(
        self, batch_dir: Path, sample_result: ExtractionResult
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=sample_result):
            result = runner.invoke(app, ["extract-batch", str(batch_dir)])

        assert result.exit_code == 0
        # Should NOT contain "Error" table header
        assert "Error" not in result.output or "0 failed" in result.output

    def test_batch_exits_code_1_when_failures(
        self, batch_dir: Path
    ) -> None:
        with patch("content_extractor.cli.extract_content", side_effect=RuntimeError("fail")):
            result = runner.invoke(app, ["extract-batch", str(batch_dir)])

        assert result.exit_code == 1

    def test_batch_exits_code_0_when_all_succeed(
        self, batch_dir: Path, sample_result: ExtractionResult
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=sample_result):
            result = runner.invoke(app, ["extract-batch", str(batch_dir)])

        assert result.exit_code == 0

    def test_batch_invalid_path(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nope"
        result = runner.invoke(app, ["extract-batch", str(nonexistent)])

        assert result.exit_code == 1

    def test_batch_force_flag(
        self, batch_dir: Path, sample_result: ExtractionResult
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=sample_result) as mock:
            result = runner.invoke(app, ["extract-batch", str(batch_dir), "--force"])

        assert result.exit_code == 0
        config: ExtractorConfig = mock.call_args[0][1]
        assert config.force_reprocess is True

    def test_batch_whisper_model_flag(
        self, batch_dir: Path, sample_result: ExtractionResult
    ) -> None:
        with patch("content_extractor.cli.extract_content", return_value=sample_result) as mock:
            result = runner.invoke(app, ["extract-batch", str(batch_dir), "--whisper-model", "small"])

        assert result.exit_code == 0
        config: ExtractorConfig = mock.call_args[0][1]
        assert config.whisper_model == "small"
