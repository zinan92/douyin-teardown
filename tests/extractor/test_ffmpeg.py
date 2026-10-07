"""Unit tests for the FFmpeg audio extraction module.

All tests mock subprocess.run to avoid requiring real FFmpeg/ffprobe binaries.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from content_extractor.video.ffmpeg import (
    AudioProbeResult,
    FFmpegError,
    extract_audio,
    normalize_audio,
    probe_audio_stream,
)


# ---------------------------------------------------------------------------
# probe_audio_stream tests
# ---------------------------------------------------------------------------


class TestProbeAudioStream:
    """Tests for probe_audio_stream()."""

    def test_returns_probe_result_with_audio(self, tmp_path: Path) -> None:
        """Video with an audio stream returns has_audio=True with codec info."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")

        ffprobe_output = json.dumps(
            {
                "streams": [
                    {
                        "codec_name": "aac",
                        "sample_rate": "44100",
                        "duration": "120.5",
                    }
                ]
            }
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = probe_audio_stream(video_path)

        assert result.has_audio is True
        assert result.codec == "aac"
        assert result.sample_rate == 44100
        assert result.duration_seconds == 120.5
        # Verify ffprobe was called with correct args
        call_args = mock_run.call_args
        assert "ffprobe" in call_args[0][0]

    def test_returns_no_audio_when_no_streams(self, tmp_path: Path) -> None:
        """Video with no audio stream returns has_audio=False."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")

        ffprobe_output = json.dumps({"streams": []})

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output

        with patch("subprocess.run", return_value=mock_result):
            result = probe_audio_stream(video_path)

        assert result.has_audio is False
        assert result.codec is None
        assert result.sample_rate is None
        assert result.duration_seconds is None

    def test_raises_when_ffprobe_not_found(self, tmp_path: Path) -> None:
        """Raises FFmpegError when ffprobe binary is not installed."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")

        with patch(
            "subprocess.run", side_effect=FileNotFoundError("ffprobe")
        ):
            with pytest.raises(FFmpegError, match="ffprobe not found"):
                probe_audio_stream(video_path)

    def test_raises_when_input_file_missing(self, tmp_path: Path) -> None:
        """Raises FFmpegError when the video file does not exist."""
        video_path = tmp_path / "nonexistent.mp4"

        with pytest.raises(FFmpegError, match="does not exist"):
            probe_audio_stream(video_path)

    def test_raises_when_ffprobe_returns_nonzero(self, tmp_path: Path) -> None:
        """Raises FFmpegError when ffprobe exits with error code."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Invalid data found"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(FFmpegError, match="ffprobe failed"):
                probe_audio_stream(video_path)


# ---------------------------------------------------------------------------
# extract_audio tests
# ---------------------------------------------------------------------------


class TestExtractAudio:
    """Tests for extract_audio()."""

    def test_produces_wav_at_target_path(self, tmp_path: Path) -> None:
        """Successful extraction creates WAV at the output path."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")
        output_path = tmp_path / "audio.wav"

        # Mock probe to return audio present
        probe_result = AudioProbeResult(
            has_audio=True, codec="aac", sample_rate=44100, duration_seconds=60.0
        )

        def mock_subprocess_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0

            if "ffmpeg" in cmd[0] and "ffprobe" not in cmd[0]:
                # Simulate ffmpeg creating a tmp output file
                tmp_file = output_path.with_suffix(".wav.tmp")
                tmp_file.write_bytes(b"RIFF" + b"\x00" * 100)
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with (
            patch(
                "content_extractor.video.ffmpeg.probe_audio_stream",
                return_value=probe_result,
            ),
            patch("subprocess.run", side_effect=mock_subprocess_run),
        ):
            result = extract_audio(video_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_raises_when_no_audio_stream(self, tmp_path: Path) -> None:
        """Raises FFmpegError when video has no audio stream."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")
        output_path = tmp_path / "audio.wav"

        probe_result = AudioProbeResult(
            has_audio=False, codec=None, sample_rate=None, duration_seconds=None
        )

        with patch(
            "content_extractor.video.ffmpeg.probe_audio_stream",
            return_value=probe_result,
        ):
            with pytest.raises(FFmpegError, match="No audio stream"):
                extract_audio(video_path, output_path)

    def test_raises_when_ffmpeg_returns_nonzero(self, tmp_path: Path) -> None:
        """Raises FFmpegError when ffmpeg exits with error code."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")
        output_path = tmp_path / "audio.wav"

        probe_result = AudioProbeResult(
            has_audio=True, codec="aac", sample_rate=44100, duration_seconds=60.0
        )

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Conversion failed"

        with (
            patch(
                "content_extractor.video.ffmpeg.probe_audio_stream",
                return_value=probe_result,
            ),
            patch("subprocess.run", return_value=mock_result),
        ):
            with pytest.raises(FFmpegError, match="ffmpeg failed"):
                extract_audio(video_path, output_path)

    def test_raises_when_output_is_zero_bytes(self, tmp_path: Path) -> None:
        """Raises FFmpegError when ffmpeg produces a 0-byte output."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")
        output_path = tmp_path / "audio.wav"

        probe_result = AudioProbeResult(
            has_audio=True, codec="aac", sample_rate=44100, duration_seconds=60.0
        )

        def mock_subprocess_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            if "ffmpeg" in cmd[0] and "ffprobe" not in cmd[0]:
                # Create 0-byte tmp file
                tmp_file = output_path.with_suffix(".wav.tmp")
                tmp_file.write_bytes(b"")
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with (
            patch(
                "content_extractor.video.ffmpeg.probe_audio_stream",
                return_value=probe_result,
            ),
            patch("subprocess.run", side_effect=mock_subprocess_run),
        ):
            with pytest.raises(FFmpegError, match="0 bytes"):
                extract_audio(video_path, output_path)

    def test_uses_atomic_rename(self, tmp_path: Path) -> None:
        """Extract writes to .tmp then renames to final path (atomic)."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")
        output_path = tmp_path / "audio.wav"
        tmp_output = output_path.with_suffix(".wav.tmp")

        probe_result = AudioProbeResult(
            has_audio=True, codec="aac", sample_rate=44100, duration_seconds=60.0
        )

        def mock_subprocess_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            if "ffmpeg" in cmd[0] and "ffprobe" not in cmd[0]:
                # Verify ffmpeg is told to write to .tmp path
                assert str(tmp_output) in cmd
                tmp_output.write_bytes(b"RIFF" + b"\x00" * 100)
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with (
            patch(
                "content_extractor.video.ffmpeg.probe_audio_stream",
                return_value=probe_result,
            ),
            patch("subprocess.run", side_effect=mock_subprocess_run),
        ):
            extract_audio(video_path, output_path)

        # After success, tmp file should be renamed to output
        assert output_path.exists()
        assert not tmp_output.exists()


# ---------------------------------------------------------------------------
# normalize_audio tests
# ---------------------------------------------------------------------------


class TestNormalizeAudio:
    """Tests for normalize_audio()."""

    def test_normalizes_audio_successfully(self, tmp_path: Path) -> None:
        """Successful normalization creates WAV at the output path."""
        input_path = tmp_path / "input.wav"
        input_path.write_bytes(b"fake-audio")
        output_path = tmp_path / "normalized.wav"

        def mock_subprocess_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            # Simulate ffmpeg creating a tmp output file
            tmp_file = output_path.with_suffix(".wav.tmp")
            tmp_file.write_bytes(b"RIFF" + b"\x00" * 100)
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = normalize_audio(input_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_uses_loudnorm_filter(self, tmp_path: Path) -> None:
        """normalize_audio uses loudnorm filter with correct parameters."""
        input_path = tmp_path / "input.wav"
        input_path.write_bytes(b"fake-audio")
        output_path = tmp_path / "normalized.wav"

        def mock_subprocess_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            tmp_file = output_path.with_suffix(".wav.tmp")
            tmp_file.write_bytes(b"RIFF" + b"\x00" * 100)
            mock.stdout = ""
            mock.stderr = ""
            # Verify loudnorm filter is in command
            cmd_str = " ".join(cmd)
            assert "loudnorm" in cmd_str
            assert "I=-16" in cmd_str
            return mock

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            normalize_audio(input_path, output_path)

    def test_raises_when_ffmpeg_fails(self, tmp_path: Path) -> None:
        """Raises FFmpegError when ffmpeg exits with error code."""
        input_path = tmp_path / "input.wav"
        input_path.write_bytes(b"fake-audio")
        output_path = tmp_path / "normalized.wav"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Normalization failed"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(FFmpegError, match="ffmpeg failed"):
                normalize_audio(input_path, output_path)

    def test_raises_when_output_zero_bytes(self, tmp_path: Path) -> None:
        """Raises FFmpegError when ffmpeg produces 0-byte output."""
        input_path = tmp_path / "input.wav"
        input_path.write_bytes(b"fake-audio")
        output_path = tmp_path / "normalized.wav"

        def mock_subprocess_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            tmp_file = output_path.with_suffix(".wav.tmp")
            tmp_file.write_bytes(b"")
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            with pytest.raises(FFmpegError, match="0 bytes"):
                normalize_audio(input_path, output_path)

    def test_raises_when_input_not_found(self, tmp_path: Path) -> None:
        """Raises FFmpegError when input file does not exist."""
        input_path = tmp_path / "nonexistent.wav"
        output_path = tmp_path / "normalized.wav"

        with pytest.raises(FFmpegError, match="does not exist"):
            normalize_audio(input_path, output_path)

    def test_uses_atomic_tmp_file(self, tmp_path: Path) -> None:
        """normalize_audio writes to .tmp then renames atomically."""
        input_path = tmp_path / "input.wav"
        input_path.write_bytes(b"fake-audio")
        output_path = tmp_path / "normalized.wav"
        tmp_output = output_path.with_suffix(".wav.tmp")

        def mock_subprocess_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            assert str(tmp_output) in cmd
            tmp_output.write_bytes(b"RIFF" + b"\x00" * 100)
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            normalize_audio(input_path, output_path)

        assert output_path.exists()
        assert not tmp_output.exists()
