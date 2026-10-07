"""Whisper transcription module with dual backend.

Uses mlx-whisper on Apple Silicon (GPU via Metal) when available, otherwise
falls back to faster-whisper + CTranslate2 (CPU int8 everywhere, CUDA on
nvidia). Selection can be forced via the CONTENT_EXTRACTOR_WHISPER_BACKEND
environment variable ("mlx" or "faster").

Anti-hallucination settings applied to both backends:
- condition_on_previous_text=False to prevent cascade errors
- no_speech_prob > 0.6 filtering
- Explicit language='zh' with Chinese initial_prompt

faster-whisper path additionally uses vad_filter with
min_silence_duration_ms=500. mlx-whisper has no built-in VAD; its
speech_ratio is approximated from segment coverage instead.

Model instances are cached to avoid ~5 second reload per call.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from content_extractor.models import TranscriptSegment

logger = logging.getLogger(__name__)


# Model name mapping for mlx-whisper (HuggingFace repos from mlx-community).
# Users can also pass a full HF repo id directly; that's passed through.
_MLX_MODEL_MAP: dict[str, str] = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def _resolve_backend() -> str:
    """Pick "mlx" or "faster" based on platform + env override.

    Apple Silicon macOS with mlx-whisper importable → "mlx".
    CONTENT_EXTRACTOR_WHISPER_BACKEND=faster|mlx forces the choice.
    Everything else → "faster".
    """
    forced = os.environ.get("CONTENT_EXTRACTOR_WHISPER_BACKEND", "").strip().lower()
    if forced in ("mlx", "faster"):
        return forced

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return "faster"

    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        return "faster"

    return "mlx"


def _resolve_mlx_model(model_name: str) -> str:
    """Map a generic model name to an mlx-community HF repo id.

    If `model_name` already looks like an HF repo id (contains "/"), return
    as-is. Otherwise look it up in _MLX_MODEL_MAP, falling back to the turbo
    default with a warning.
    """
    if "/" in model_name:
        return model_name
    if model_name in _MLX_MODEL_MAP:
        return _MLX_MODEL_MAP[model_name]
    logger.warning(
        "Unknown MLX whisper model %r; falling back to %s",
        model_name,
        _MLX_MODEL_MAP["turbo"],
    )
    return _MLX_MODEL_MAP["turbo"]


class TranscriptionError(Exception):
    """Raised when transcription fails."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of transcription including VAD metadata.

    Attributes
    ----------
    segments:
        Immutable tuple of transcript segments with timestamps.
    speech_ratio:
        Fraction of audio containing speech (from VAD), 0.0 to 1.0.
    duration_seconds:
        Total audio duration in seconds.
    """

    segments: tuple[TranscriptSegment, ...]
    speech_ratio: float
    duration_seconds: float


# Module-level model cache. Justified by ~5s model load time.
# This is the ONE mutable module-level state.
_model_cache: dict[str, WhisperModel] = {}


def _get_model(model_name: str) -> WhisperModel:
    """Return a cached WhisperModel, loading if necessary.

    Parameters
    ----------
    model_name:
        Model size or path (e.g. "turbo", "large-v3").

    Returns
    -------
    WhisperModel
        Cached or freshly loaded model instance.

    Raises
    ------
    TranscriptionError
        If the model fails to load.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    logger.info("Loading Whisper model: %s", model_name)

    try:
        model = WhisperModel(
            model_size_or_path=model_name,
            device="auto",
            compute_type="int8",
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Failed to load Whisper model '{model_name}': {exc}"
        ) from exc

    _model_cache[model_name] = model
    return model


def _transcribe_with_faster_whisper(
    audio_path: Path,
    *,
    whisper_model: str,
    language: str,
) -> TranscriptionResult:
    """Transcribe using faster-whisper (CPU int8 / CUDA)."""
    model = _get_model(whisper_model)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        initial_prompt="以下是普通话的句子。",
        condition_on_previous_text=False,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments: list[TranscriptSegment] = []

    for segment in segments_iter:
        if segment.no_speech_prob > 0.6:
            continue

        raw_confidence = math.exp(segment.avg_logprob)
        confidence = min(max(raw_confidence, 0.0), 1.0)

        segments.append(
            TranscriptSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                confidence=confidence,
            )
        )

    duration = info.duration if info.duration else 0.0
    speech_ratio = (
        info.duration_after_vad / duration
        if duration > 0
        else 0.0
    )

    return TranscriptionResult(
        segments=tuple(segments),
        speech_ratio=speech_ratio,
        duration_seconds=duration,
    )


def _transcribe_with_mlx(
    audio_path: Path,
    *,
    whisper_model: str,
    language: str,
) -> TranscriptionResult:
    """Transcribe using mlx-whisper (Apple Silicon GPU via Metal).

    mlx-whisper has no built-in VAD. Segments whose no_speech_prob > 0.6 are
    dropped to match the faster-whisper behavior. duration_seconds is taken
    from the last segment's end; speech_ratio is approximated as the fraction
    of that timeline covered by retained speech segments.
    """
    import mlx_whisper  # imported lazily; availability checked in _resolve_backend

    repo_id = _resolve_mlx_model(whisper_model)
    logger.info("Transcribing via mlx-whisper (%s) on Apple Silicon", repo_id)

    try:
        result: dict[str, Any] = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=repo_id,
            language=language,
            initial_prompt="以下是普通话的句子。",
            condition_on_previous_text=False,
            temperature=0.0,
            verbose=False,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"mlx-whisper transcribe failed for {audio_path}: {exc}"
        ) from exc

    raw_segments = result.get("segments") or []
    segments: list[TranscriptSegment] = []
    speech_duration = 0.0
    max_end = 0.0

    for segment in raw_segments:
        no_speech_prob = float(segment.get("no_speech_prob") or 0.0)
        if no_speech_prob > 0.6:
            continue

        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        if end > max_end:
            max_end = end
        speech_duration += max(0.0, end - start)

        avg_logprob = segment.get("avg_logprob")
        if avg_logprob is None:
            confidence = 0.0
        else:
            raw_confidence = math.exp(float(avg_logprob))
            confidence = min(max(raw_confidence, 0.0), 1.0)

        text = (segment.get("text") or "").strip()
        if not text:
            continue

        segments.append(
            TranscriptSegment(
                text=text,
                start=start,
                end=end,
                confidence=confidence,
            )
        )

    duration = max_end
    speech_ratio = speech_duration / duration if duration > 0 else 0.0

    return TranscriptionResult(
        segments=tuple(segments),
        speech_ratio=speech_ratio,
        duration_seconds=duration,
    )


def transcribe_audio(
    audio_path: Path,
    *,
    whisper_model: str = "turbo",
    language: str = "zh",
) -> TranscriptionResult:
    """Transcribe an audio file using the best-available Whisper backend.

    On Apple Silicon with mlx-whisper installed, uses mlx-whisper for
    GPU-accelerated inference via Metal. Otherwise uses faster-whisper.
    Override with CONTENT_EXTRACTOR_WHISPER_BACKEND=faster|mlx.

    Parameters
    ----------
    audio_path:
        Path to a WAV audio file (16kHz mono recommended).
    whisper_model:
        Model name. Accepts short names ("turbo", "large-v3", ...) or full
        HuggingFace repo ids for mlx-whisper.
    language:
        Language code for transcription (default "zh").

    Returns
    -------
    TranscriptionResult
        Frozen dataclass with segments, speech_ratio, and duration.

    Raises
    ------
    TranscriptionError
        If the model fails to load or transcription fails.
    """
    backend = _resolve_backend()
    if backend == "mlx":
        return _transcribe_with_mlx(
            audio_path,
            whisper_model=whisper_model,
            language=language,
        )
    return _transcribe_with_faster_whisper(
        audio_path,
        whisper_model=whisper_model,
        language=language,
    )
