"""Integration-level tests for VideoExtractor adapter.

All FFmpeg and Whisper calls are mocked -- tests verify the adapter
orchestration logic: find video -> probe -> extract -> transcribe -> result.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from content_extractor.adapters.base import Extractor
from content_extractor.adapters.video import VideoExtractor
from content_extractor.config import ExtractorConfig
from content_extractor.models import (
    ExtractionResult,
    QualityMetadata,
    Transcript,
    TranscriptSegment,
)
from content_extractor.video.ffmpeg import AudioProbeResult
from content_extractor.video.transcribe import TranscriptionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONTENT_ITEM = {
    "platform": "douyin",
    "content_id": "test-video-001",
    "content_type": "video",
    "title": "Test Video",
    "description": "A test video description",
    "author_id": "author-123",
    "author_name": "Test Author",
    "publish_time": "2026-03-30T10:00:00Z",
    "source_url": "https://example.com/video/001",
    "media_files": ["media/video.mp4"],
    "downloaded_at": "2026-03-30T10:05:00Z",
    "likes": 100,
    "comments": 25,
    "shares": 10,
    "views": 5000,
    "collects": 50,
}


@pytest.fixture()
def content_dir(tmp_path: Path) -> Path:
    """Create a content directory with content_item.json and a video file."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "video.mp4").write_bytes(b"fake-video-data")

    item_path = tmp_path / "content_item.json"
    item_path.write_text(json.dumps(SAMPLE_CONTENT_ITEM))

    return tmp_path


@pytest.fixture()
def config() -> ExtractorConfig:
    """Default config."""
    return ExtractorConfig()


@pytest.fixture()
def mock_segments() -> tuple[TranscriptSegment, ...]:
    """Sample transcript segments."""
    return (
        TranscriptSegment(text="Hello", start=0.0, end=2.0, confidence=0.85),
        TranscriptSegment(text="World", start=2.0, end=4.0, confidence=0.90),
    )


@pytest.fixture()
def mock_transcription_result(
    mock_segments: tuple[TranscriptSegment, ...],
) -> TranscriptionResult:
    """Sample TranscriptionResult with healthy speech ratio."""
    return TranscriptionResult(
        segments=mock_segments,
        speech_ratio=0.85,
        duration_seconds=60.0,
    )


@pytest.fixture()
def probe_with_audio() -> AudioProbeResult:
    """Probe result indicating audio is present."""
    return AudioProbeResult(
        has_audio=True, codec="aac", sample_rate=44100, duration_seconds=60.0
    )


@pytest.fixture()
def probe_no_audio() -> AudioProbeResult:
    """Probe result indicating no audio stream."""
    return AudioProbeResult(
        has_audio=False, codec=None, sample_rate=None, duration_seconds=None
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVideoExtractor:
    """Tests for VideoExtractor.extract()."""

    def test_satisfies_extractor_protocol(self) -> None:
        """VideoExtractor satisfies the Extractor Protocol."""
        extractor = VideoExtractor()
        assert isinstance(extractor, Extractor)
        assert extractor.content_type == "video"

    def test_successful_extraction(
        self,
        content_dir: Path,
        config: ExtractorConfig,
        probe_with_audio: AudioProbeResult,
        mock_transcription_result: TranscriptionResult,
    ) -> None:
        """Full pipeline produces ExtractionResult with transcript."""
        extractor = VideoExtractor()

        def mock_extract_audio(video_path: Path, output_path: Path) -> Path:
            output_path.write_bytes(b"fake-wav-data")
            return output_path

        with (
            patch(
                "content_extractor.adapters.video.probe_audio_stream",
                return_value=probe_with_audio,
            ),
            patch(
                "content_extractor.adapters.video.extract_audio",
                side_effect=mock_extract_audio,
            ),
            patch(
                "content_extractor.adapters.video.transcribe_audio",
                return_value=mock_transcription_result,
            ),
            patch(
                "content_extractor.adapters.video.normalize_audio",
                side_effect=lambda inp, out: out,
            ),
        ):
            result = extractor.extract(content_dir, config)

        assert isinstance(result, ExtractionResult)
        assert result.content_id == "test-video-001"
        assert result.content_type == "video"
        assert result.transcript is not None
        assert isinstance(result.transcript, Transcript)
        assert result.transcript.language == "zh"
        assert len(result.transcript.segments) == 2
        assert result.transcript.full_text == "Hello World"
        assert result.raw_text == "Hello World"
        assert result.quality.confidence == pytest.approx(0.875, abs=0.001)
        assert result.quality.language == "zh"
        assert result.quality.word_count > 0
        assert result.quality.processing_time_seconds > 0.0

    def test_no_audio_stream_returns_empty_result(
        self,
        content_dir: Path,
        config: ExtractorConfig,
        probe_no_audio: AudioProbeResult,
    ) -> None:
        """Video with no audio returns result with transcript=None."""
        extractor = VideoExtractor()

        with patch(
            "content_extractor.adapters.video.probe_audio_stream",
            return_value=probe_no_audio,
        ):
            result = extractor.extract(content_dir, config)

        assert result.content_id == "test-video-001"
        assert result.transcript is None
        assert result.raw_text == ""
        assert result.quality.confidence == 0.0
        assert result.quality.language == "zh"

    def test_no_video_file_raises(
        self, tmp_path: Path, config: ExtractorConfig
    ) -> None:
        """Raises FileNotFoundError when no video file in media/."""
        media_dir = tmp_path / "media"
        media_dir.mkdir()
        # No video file, just a text file
        (media_dir / "readme.txt").write_text("not a video")

        item_data = {**SAMPLE_CONTENT_ITEM, "media_files": []}
        (tmp_path / "content_item.json").write_text(json.dumps(item_data))

        extractor = VideoExtractor()
        with pytest.raises(FileNotFoundError, match="No video file"):
            extractor.extract(tmp_path, config)

    def test_temp_wav_cleaned_up_on_success(
        self,
        content_dir: Path,
        config: ExtractorConfig,
        probe_with_audio: AudioProbeResult,
        mock_transcription_result: TranscriptionResult,
    ) -> None:
        """Temporary WAV files are removed after successful extraction."""
        extractor = VideoExtractor()

        def mock_extract_audio(video_path: Path, output_path: Path) -> Path:
            output_path.write_bytes(b"fake-wav")
            return output_path

        with (
            patch(
                "content_extractor.adapters.video.probe_audio_stream",
                return_value=probe_with_audio,
            ),
            patch(
                "content_extractor.adapters.video.extract_audio",
                side_effect=mock_extract_audio,
            ),
            patch(
                "content_extractor.adapters.video.transcribe_audio",
                return_value=mock_transcription_result,
            ),
            patch(
                "content_extractor.adapters.video.normalize_audio",
                side_effect=lambda inp, out: out,
            ),
        ):
            extractor.extract(content_dir, config)

        # Temp WAV files should be cleaned up
        tmp_wav = content_dir / "media" / ".tmp_audio.wav"
        tmp_norm = content_dir / "media" / ".tmp_normalized.wav"
        assert not tmp_wav.exists()
        assert not tmp_norm.exists()

    def test_temp_wav_cleaned_up_on_error(
        self,
        content_dir: Path,
        config: ExtractorConfig,
        probe_with_audio: AudioProbeResult,
    ) -> None:
        """Temporary WAV files are removed even when transcription fails."""
        extractor = VideoExtractor()

        def mock_extract_audio(video_path: Path, output_path: Path) -> Path:
            output_path.write_bytes(b"fake-wav")
            return output_path

        with (
            patch(
                "content_extractor.adapters.video.probe_audio_stream",
                return_value=probe_with_audio,
            ),
            patch(
                "content_extractor.adapters.video.extract_audio",
                side_effect=mock_extract_audio,
            ),
            patch(
                "content_extractor.adapters.video.normalize_audio",
                side_effect=lambda inp, out: out,
            ),
            patch(
                "content_extractor.adapters.video.transcribe_audio",
                side_effect=RuntimeError("transcription failed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="transcription failed"):
                extractor.extract(content_dir, config)

        tmp_wav = content_dir / "media" / ".tmp_audio.wav"
        assert not tmp_wav.exists()

    def test_whisper_model_passed_from_config(
        self,
        content_dir: Path,
        probe_with_audio: AudioProbeResult,
        mock_transcription_result: TranscriptionResult,
    ) -> None:
        """config.whisper_model is forwarded to transcribe_audio."""
        config = ExtractorConfig(whisper_model="large-v3")
        extractor = VideoExtractor()

        def mock_extract_audio(video_path: Path, output_path: Path) -> Path:
            output_path.write_bytes(b"fake-wav")
            return output_path

        with (
            patch(
                "content_extractor.adapters.video.probe_audio_stream",
                return_value=probe_with_audio,
            ),
            patch(
                "content_extractor.adapters.video.extract_audio",
                side_effect=mock_extract_audio,
            ),
            patch(
                "content_extractor.adapters.video.transcribe_audio",
                return_value=mock_transcription_result,
            ) as mock_transcribe,
            patch(
                "content_extractor.adapters.video.normalize_audio",
                side_effect=lambda inp, out: out,
            ),
        ):
            extractor.extract(content_dir, config)

        call_kwargs = mock_transcribe.call_args[1]
        assert call_kwargs["whisper_model"] == "large-v3"

    def test_platform_metadata_includes_nonzero(
        self,
        content_dir: Path,
        config: ExtractorConfig,
        probe_with_audio: AudioProbeResult,
        mock_transcription_result: TranscriptionResult,
    ) -> None:
        """Platform metadata includes non-zero engagement fields."""
        extractor = VideoExtractor()

        def mock_extract_audio(video_path: Path, output_path: Path) -> Path:
            output_path.write_bytes(b"fake-wav")
            return output_path

        with (
            patch(
                "content_extractor.adapters.video.probe_audio_stream",
                return_value=probe_with_audio,
            ),
            patch(
                "content_extractor.adapters.video.extract_audio",
                side_effect=mock_extract_audio,
            ),
            patch(
                "content_extractor.adapters.video.transcribe_audio",
                return_value=mock_transcription_result,
            ),
            patch(
                "content_extractor.adapters.video.normalize_audio",
                side_effect=lambda inp, out: out,
            ),
        ):
            result = extractor.extract(content_dir, config)

        assert result.platform_metadata["likes"] == 100
        assert result.platform_metadata["comments"] == 25
        assert result.platform_metadata["shares"] == 10
        assert result.platform_metadata["views"] == 5000
        assert result.platform_metadata["collects"] == 50

    def test_normalization_failure_falls_back(
        self,
        content_dir: Path,
        config: ExtractorConfig,
        probe_with_audio: AudioProbeResult,
        mock_transcription_result: TranscriptionResult,
    ) -> None:
        """Normalization failure falls back to unnormalized audio."""
        from content_extractor.video.ffmpeg import FFmpegError

        extractor = VideoExtractor()

        def mock_extract_audio(video_path: Path, output_path: Path) -> Path:
            output_path.write_bytes(b"fake-wav")
            return output_path

        with (
            patch(
                "content_extractor.adapters.video.probe_audio_stream",
                return_value=probe_with_audio,
            ),
            patch(
                "content_extractor.adapters.video.extract_audio",
                side_effect=mock_extract_audio,
            ),
            patch(
                "content_extractor.adapters.video.normalize_audio",
                side_effect=FFmpegError("normalization failed"),
            ),
            patch(
                "content_extractor.adapters.video.transcribe_audio",
                return_value=mock_transcription_result,
            ),
        ):
            result = extractor.extract(content_dir, config)

        # Should still succeed with transcript
        assert result.transcript is not None
        assert len(result.transcript.segments) == 2

    def test_low_speech_ratio_skips_transcription(
        self,
        content_dir: Path,
        config: ExtractorConfig,
        probe_with_audio: AudioProbeResult,
    ) -> None:
        """Videos with <10% speech return result with confidence=0.0."""
        extractor = VideoExtractor()

        low_speech_result = TranscriptionResult(
            segments=(),
            speech_ratio=0.05,
            duration_seconds=60.0,
        )

        def mock_extract_audio(video_path: Path, output_path: Path) -> Path:
            output_path.write_bytes(b"fake-wav")
            return output_path

        with (
            patch(
                "content_extractor.adapters.video.probe_audio_stream",
                return_value=probe_with_audio,
            ),
            patch(
                "content_extractor.adapters.video.extract_audio",
                side_effect=mock_extract_audio,
            ),
            patch(
                "content_extractor.adapters.video.normalize_audio",
                side_effect=lambda inp, out: out,
            ),
            patch(
                "content_extractor.adapters.video.transcribe_audio",
                return_value=low_speech_result,
            ),
        ):
            result = extractor.extract(content_dir, config)

        assert result.quality.confidence == 0.0
        assert result.platform_metadata.get("low_speech_ratio") is not None

    def test_suspicious_segments_flagged(
        self,
        content_dir: Path,
        config: ExtractorConfig,
        probe_with_audio: AudioProbeResult,
    ) -> None:
        """Low confidence segments get is_suspicious=True."""
        extractor = VideoExtractor()

        low_conf_result = TranscriptionResult(
            segments=(
                TranscriptSegment(
                    text="ok text", start=0.0, end=2.0, confidence=0.2
                ),
                TranscriptSegment(
                    text="good text", start=2.0, end=4.0, confidence=0.9
                ),
            ),
            speech_ratio=0.8,
            duration_seconds=60.0,
        )

        def mock_extract_audio(video_path: Path, output_path: Path) -> Path:
            output_path.write_bytes(b"fake-wav")
            return output_path

        with (
            patch(
                "content_extractor.adapters.video.probe_audio_stream",
                return_value=probe_with_audio,
            ),
            patch(
                "content_extractor.adapters.video.extract_audio",
                side_effect=mock_extract_audio,
            ),
            patch(
                "content_extractor.adapters.video.normalize_audio",
                side_effect=lambda inp, out: out,
            ),
            patch(
                "content_extractor.adapters.video.transcribe_audio",
                return_value=low_conf_result,
            ),
        ):
            result = extractor.extract(content_dir, config)

        assert result.transcript is not None
        assert result.transcript.segments[0].is_suspicious is True
        assert result.transcript.segments[1].is_suspicious is False
        assert len(result.quality.hallucination_warnings) > 0

    def test_hallucination_warnings_populated(
        self,
        content_dir: Path,
        config: ExtractorConfig,
        probe_with_audio: AudioProbeResult,
    ) -> None:
        """Hallucination warnings appear in QualityMetadata."""
        extractor = VideoExtractor()

        low_conf_result = TranscriptionResult(
            segments=(
                TranscriptSegment(
                    text="text", start=0.0, end=1.0, confidence=0.2
                ),
                TranscriptSegment(
                    text="more", start=1.0, end=2.0, confidence=0.3
                ),
            ),
            speech_ratio=0.8,
            duration_seconds=60.0,
        )

        def mock_extract_audio(video_path: Path, output_path: Path) -> Path:
            output_path.write_bytes(b"fake-wav")
            return output_path

        with (
            patch(
                "content_extractor.adapters.video.probe_audio_stream",
                return_value=probe_with_audio,
            ),
            patch(
                "content_extractor.adapters.video.extract_audio",
                side_effect=mock_extract_audio,
            ),
            patch(
                "content_extractor.adapters.video.normalize_audio",
                side_effect=lambda inp, out: out,
            ),
            patch(
                "content_extractor.adapters.video.transcribe_audio",
                return_value=low_conf_result,
            ),
        ):
            result = extractor.extract(content_dir, config)

        assert len(result.quality.hallucination_warnings) > 0
        # Should have confidence warning
        assert any(
            "confidence" in w.lower()
            for w in result.quality.hallucination_warnings
        )

    def test_all_suspicious_halves_confidence(
        self,
        content_dir: Path,
        config: ExtractorConfig,
        probe_with_audio: AudioProbeResult,
    ) -> None:
        """When all segments suspicious, confidence is halved."""
        extractor = VideoExtractor()

        all_sus_result = TranscriptionResult(
            segments=(
                TranscriptSegment(
                    text="a", start=0.0, end=1.0, confidence=0.2
                ),
                TranscriptSegment(
                    text="b", start=1.0, end=2.0, confidence=0.3
                ),
            ),
            speech_ratio=0.8,
            duration_seconds=60.0,
        )

        def mock_extract_audio(video_path: Path, output_path: Path) -> Path:
            output_path.write_bytes(b"fake-wav")
            return output_path

        with (
            patch(
                "content_extractor.adapters.video.probe_audio_stream",
                return_value=probe_with_audio,
            ),
            patch(
                "content_extractor.adapters.video.extract_audio",
                side_effect=mock_extract_audio,
            ),
            patch(
                "content_extractor.adapters.video.normalize_audio",
                side_effect=lambda inp, out: out,
            ),
            patch(
                "content_extractor.adapters.video.transcribe_audio",
                return_value=all_sus_result,
            ),
        ):
            result = extractor.extract(content_dir, config)

        # avg confidence of (0.2 + 0.3) / 2 = 0.25, halved = 0.125
        assert result.quality.confidence == pytest.approx(0.125, abs=0.01)
