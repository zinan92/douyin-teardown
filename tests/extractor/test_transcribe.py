"""Unit tests for the faster-whisper transcription module.

All tests mock WhisperModel to avoid loading real models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from content_extractor.models import TranscriptSegment
from content_extractor.video.transcribe import (
    TranscriptionError,
    TranscriptionResult,
    _model_cache,
    transcribe_audio,
)


@dataclass
class MockSegment:
    """Mimics faster-whisper's segment output."""

    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    """Clear the model cache before each test."""
    _model_cache.clear()


class TestTranscribeAudio:
    """Tests for transcribe_audio()."""

    def test_returns_transcription_result(self, tmp_path: Path) -> None:
        """Normal segments produce TranscriptionResult with segments and speech_ratio."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_segments = [
            MockSegment(
                start=0.0, end=2.5, text="  Hello world  ",
                avg_logprob=-0.3, no_speech_prob=0.1,
            ),
            MockSegment(
                start=2.5, end=5.0, text="  Second segment  ",
                avg_logprob=-0.5, no_speech_prob=0.2,
            ),
        ]

        mock_info = MagicMock()
        mock_info.duration = 10.0
        mock_info.duration_after_vad = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)

        with patch(
            "content_extractor.video.transcribe._get_model",
            return_value=mock_model,
        ):
            result = transcribe_audio(audio_path)

        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 2
        assert isinstance(result.segments, tuple)
        assert isinstance(result.segments[0], TranscriptSegment)
        assert result.segments[0].text == "Hello world"
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 2.5
        assert result.segments[0].confidence == pytest.approx(math.exp(-0.3), abs=0.001)
        assert result.segments[1].text == "Second segment"
        assert result.speech_ratio == pytest.approx(0.5, abs=0.001)
        assert result.duration_seconds == 10.0

    def test_filters_high_no_speech_prob(self, tmp_path: Path) -> None:
        """Segments with no_speech_prob > 0.6 are filtered out."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_segments = [
            MockSegment(
                start=0.0, end=2.0, text="Keep me",
                avg_logprob=-0.3, no_speech_prob=0.1,
            ),
            MockSegment(
                start=2.0, end=4.0, text="Filter me",
                avg_logprob=-0.3, no_speech_prob=0.7,  # > 0.6
            ),
            MockSegment(
                start=4.0, end=6.0, text="Boundary",
                avg_logprob=-0.3, no_speech_prob=0.6,  # exactly 0.6 = keep
            ),
        ]

        mock_info = MagicMock()
        mock_info.duration = 6.0
        mock_info.duration_after_vad = 4.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)

        with patch(
            "content_extractor.video.transcribe._get_model",
            return_value=mock_model,
        ):
            result = transcribe_audio(audio_path)

        assert len(result.segments) == 2
        assert result.segments[0].text == "Keep me"
        assert result.segments[1].text == "Boundary"

    def test_confidence_clamped_to_unit_range(self, tmp_path: Path) -> None:
        """Confidence from math.exp is clamped to [0.0, 1.0]."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_segments = [
            MockSegment(
                start=0.0, end=1.0, text="High confidence",
                avg_logprob=0.5,  # exp(0.5) > 1.0
                no_speech_prob=0.0,
            ),
            MockSegment(
                start=1.0, end=2.0, text="Very low",
                avg_logprob=-10.0,  # exp(-10) ~ 0.00005
                no_speech_prob=0.0,
            ),
        ]

        mock_info = MagicMock()
        mock_info.duration = 2.0
        mock_info.duration_after_vad = 2.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)

        with patch(
            "content_extractor.video.transcribe._get_model",
            return_value=mock_model,
        ):
            result = transcribe_audio(audio_path)

        assert result.segments[0].confidence == 1.0  # clamped
        assert 0.0 <= result.segments[1].confidence <= 1.0

    def test_speech_ratio_zero_when_no_duration(self, tmp_path: Path) -> None:
        """speech_ratio is 0.0 when duration is 0 (division by zero guard)."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_info = MagicMock()
        mock_info.duration = 0.0
        mock_info.duration_after_vad = 0.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), mock_info)

        with patch(
            "content_extractor.video.transcribe._get_model",
            return_value=mock_model,
        ):
            result = transcribe_audio(audio_path)

        assert result.speech_ratio == 0.0
        assert result.segments == ()

    def test_speech_ratio_computed_from_vad(self, tmp_path: Path) -> None:
        """speech_ratio = duration_after_vad / duration."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_info = MagicMock()
        mock_info.duration = 100.0
        mock_info.duration_after_vad = 25.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), mock_info)

        with patch(
            "content_extractor.video.transcribe._get_model",
            return_value=mock_model,
        ):
            result = transcribe_audio(audio_path)

        assert result.speech_ratio == pytest.approx(0.25, abs=0.001)

    def test_uses_default_language_zh(self, tmp_path: Path) -> None:
        """Language defaults to 'zh' and initial_prompt is passed."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_info = MagicMock()
        mock_info.duration = 10.0
        mock_info.duration_after_vad = 8.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), mock_info)

        with patch(
            "content_extractor.video.transcribe._get_model",
            return_value=mock_model,
        ):
            transcribe_audio(audio_path)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "zh"
        assert "普通话" in call_kwargs["initial_prompt"]
        assert call_kwargs["condition_on_previous_text"] is False
        assert call_kwargs["vad_filter"] is True

    def test_custom_language_passed(self, tmp_path: Path) -> None:
        """Custom language parameter is forwarded to model.transcribe."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_info = MagicMock()
        mock_info.duration = 10.0
        mock_info.duration_after_vad = 8.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), mock_info)

        with patch(
            "content_extractor.video.transcribe._get_model",
            return_value=mock_model,
        ):
            transcribe_audio(audio_path, language="en")

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"

    def test_uses_config_whisper_model(self, tmp_path: Path) -> None:
        """whisper_model parameter is passed to _get_model."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_info = MagicMock()
        mock_info.duration = 10.0
        mock_info.duration_after_vad = 8.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), mock_info)

        with patch(
            "content_extractor.video.transcribe._get_model",
            return_value=mock_model,
        ) as mock_get:
            transcribe_audio(audio_path, whisper_model="large-v3")

        mock_get.assert_called_once_with("large-v3")

    def test_vad_parameters_passed(self, tmp_path: Path) -> None:
        """VAD filter with min_silence_duration_ms=500 is configured."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake-audio")

        mock_info = MagicMock()
        mock_info.duration = 10.0
        mock_info.duration_after_vad = 8.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), mock_info)

        with patch(
            "content_extractor.video.transcribe._get_model",
            return_value=mock_model,
        ):
            transcribe_audio(audio_path)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["vad_parameters"] == {"min_silence_duration_ms": 500}


class TestModelCache:
    """Tests for model caching behavior."""

    def test_cache_returns_same_instance(self) -> None:
        """Second call with same model name returns cached instance."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch("content_extractor.video.transcribe.WhisperModel", mock_cls):
            from content_extractor.video.transcribe import _get_model

            model1 = _get_model("turbo")
            model2 = _get_model("turbo")

        assert model1 is model2
        assert mock_cls.call_count == 1

    def test_different_models_create_separate_instances(self) -> None:
        """Different model names create separate cached instances."""
        mock_cls = MagicMock()

        with patch("content_extractor.video.transcribe.WhisperModel", mock_cls):
            from content_extractor.video.transcribe import _get_model

            _get_model("turbo")
            _get_model("large-v3")

        assert mock_cls.call_count == 2

    def test_load_failure_raises_transcription_error(self) -> None:
        """WhisperModel load failure raises TranscriptionError."""
        mock_cls = MagicMock()
        mock_cls.side_effect = RuntimeError("Failed to load model")

        with patch("content_extractor.video.transcribe.WhisperModel", mock_cls):
            from content_extractor.video.transcribe import _get_model

            with pytest.raises(TranscriptionError, match="Failed to load"):
                _get_model("nonexistent-model")
