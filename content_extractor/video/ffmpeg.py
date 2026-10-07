"""FFmpeg audio extraction, probing, and normalization via subprocess.

Provides three functions:
- probe_audio_stream: Check if a video has an audio track via ffprobe.
- extract_audio: Extract audio to 16kHz mono WAV via ffmpeg.
- normalize_audio: Volume-normalize audio via ffmpeg loudnorm filter.

All raise FFmpegError on any failure, with descriptive messages.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """Raised for any FFmpeg or ffprobe failure."""


@dataclass(frozen=True)
class AudioProbeResult:
    """Result of probing a video file for audio stream info."""

    has_audio: bool
    codec: str | None
    sample_rate: int | None
    duration_seconds: float | None


def probe_audio_stream(video_path: Path) -> AudioProbeResult:
    """Probe a video file for audio stream information via ffprobe.

    Parameters
    ----------
    video_path:
        Path to the video file to probe.

    Returns
    -------
    AudioProbeResult
        Frozen dataclass with audio stream metadata.

    Raises
    ------
    FFmpegError
        If ffprobe is not installed, the file does not exist,
        or ffprobe returns a non-zero exit code.
    """
    if not video_path.exists():
        raise FFmpegError(f"Video file does not exist: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,duration",
        "-of", "json",
        str(video_path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise FFmpegError(
            "ffprobe not found. Install FFmpeg: brew install ffmpeg"
        ) from exc

    if completed.returncode != 0:
        raise FFmpegError(
            f"ffprobe failed (exit {completed.returncode}): "
            f"{completed.stderr}"
        )

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON: {exc}") from exc

    streams = data.get("streams", [])
    if not streams:
        return AudioProbeResult(
            has_audio=False,
            codec=None,
            sample_rate=None,
            duration_seconds=None,
        )

    stream = streams[0]
    sample_rate_raw = stream.get("sample_rate")
    duration_raw = stream.get("duration")

    return AudioProbeResult(
        has_audio=True,
        codec=stream.get("codec_name"),
        sample_rate=int(sample_rate_raw) if sample_rate_raw else None,
        duration_seconds=float(duration_raw) if duration_raw else None,
    )


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from a video file to 16kHz mono WAV.

    Uses probe_audio_stream first to verify audio exists, then
    extracts via ffmpeg to a temp file and renames atomically.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    output_path:
        Desired path for the output WAV file.

    Returns
    -------
    Path
        The output_path on success.

    Raises
    ------
    FFmpegError
        If no audio stream, ffmpeg fails, or output is 0 bytes.
    """
    probe = probe_audio_stream(video_path)
    if not probe.has_audio:
        raise FFmpegError(f"No audio stream in {video_path}")

    tmp_path = output_path.with_suffix(".wav.tmp")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        "-f", "wav",
        str(tmp_path),
    ]

    logger.info("Extracting audio: %s -> %s", video_path, output_path)

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as exc:
        raise FFmpegError(
            "ffmpeg not found. Install FFmpeg: brew install ffmpeg"
        ) from exc

    if completed.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed (exit {completed.returncode}): "
            f"{completed.stderr}"
        )

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise FFmpegError(
            f"ffmpeg produced 0 bytes output at {tmp_path}"
        )

    tmp_path.rename(output_path)
    logger.info("Audio extracted successfully: %s", output_path)

    return output_path


def normalize_audio(input_path: Path, output_path: Path) -> Path:
    """Normalize audio volume using ffmpeg loudnorm filter.

    Applies EBU R128 loudness normalization (I=-16, TP=-1.5, LRA=11)
    and resamples to 16kHz mono WAV for optimal Whisper input.

    Parameters
    ----------
    input_path:
        Path to the source audio file.
    output_path:
        Desired path for the normalized output WAV file.

    Returns
    -------
    Path
        The output_path on success.

    Raises
    ------
    FFmpegError
        If input file missing, ffmpeg fails, or output is 0 bytes.
    """
    if not input_path.exists():
        raise FFmpegError(f"Audio file does not exist: {input_path}")

    tmp_path = output_path.with_suffix(".wav.tmp")

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        "-f", "wav",
        str(tmp_path),
    ]

    logger.info("Normalizing audio: %s -> %s", input_path, output_path)

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as exc:
        raise FFmpegError(
            "ffmpeg not found. Install FFmpeg: brew install ffmpeg"
        ) from exc

    if completed.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed (exit {completed.returncode}): "
            f"{completed.stderr}"
        )

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise FFmpegError(
            f"ffmpeg produced 0 bytes output at {tmp_path}"
        )

    tmp_path.rename(output_path)
    logger.info("Audio normalized successfully: %s", output_path)

    return output_path
