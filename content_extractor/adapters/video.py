"""Video content extractor.

Orchestrates the video extraction pipeline:
find video -> probe audio -> extract audio -> transcribe -> build result.

Uses FFmpeg for audio extraction and faster-whisper for transcription.
Handles no-audio gracefully and always cleans up temp files.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from content_extractor.config import ExtractorConfig
from content_extractor.loader import load_content_item
from content_extractor.models import (
    ExtractionResult,
    QualityMetadata,
    Transcript,
    TranscriptSegment,
)
from content_extractor.video.ffmpeg import (
    FFmpegError,
    extract_audio,
    normalize_audio,
    probe_audio_stream,
)
from content_extractor.video.hallucination import (
    check_segment_suspicious,
    check_transcript_hallucinations,
)
from content_extractor.video.transcribe import TranscriptionResult, transcribe_audio

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".flv")


def _find_video_file(content_dir: Path) -> Path:
    """Find the first video file in content_dir/media/.

    Raises
    ------
    FileNotFoundError
        If no video file with a recognized extension is found.
    """
    media_dir = content_dir / "media"
    if not media_dir.exists():
        raise FileNotFoundError(f"No video file in {media_dir}")

    for child in sorted(media_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
            return child

    raise FileNotFoundError(f"No video file in {media_dir}")


def _compute_word_count(text: str) -> int:
    """Compute word count handling CJK and Latin text.

    Each CJK character counts as one word.
    Latin alphabetic sequences count as one word each.
    """
    if not text:
        return 0
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
    latin_words = len(re.findall(r"[a-zA-Z]+", text))
    return cjk_chars + latin_words


def _build_platform_metadata(
    item: object,
) -> dict[str, int | str]:
    """Extract non-zero engagement fields from a ContentItem."""
    fields = ("likes", "comments", "shares", "collects", "views")
    metadata: dict[str, int | str] = {}
    for field in fields:
        value = getattr(item, field, 0)
        if value:
            metadata[field] = value
    return metadata


class VideoExtractor:
    """Extract text from video content via speech-to-text."""

    content_type: str = "video"

    @staticmethod
    def _load_from_transcript_json(
        json_path: Path,
        item: object,
        start_time: float,
        content_dir: Path,
    ) -> ExtractionResult:
        """Load an ExtractionResult from an existing transcript.json file."""
        import json as _json

        raw = _json.loads(json_path.read_text())
        segments_data = raw.get("segments", [])

        segments = tuple(
            TranscriptSegment(
                text=s["text"],
                start=s["start"],
                end=s["end"],
                confidence=s.get("confidence", 0.9),
                is_suspicious=s.get("is_suspicious", False),
            )
            for s in segments_data
        )

        full_text = raw.get("full_text", " ".join(s.text for s in segments))
        language = raw.get("language", "zh")
        content_id = raw.get("content_id", getattr(item, "content_id", "unknown"))
        elapsed = time.monotonic() - start_time

        # Preserve no-audio/low-speech semantics: if no segments and no text,
        # return the same fallback shape as the no-audio path
        if not segments and not full_text.strip():
            return ExtractionResult(
                content_id=content_id,
                content_type="video",
                raw_text="",
                transcript=None,
                quality=QualityMetadata(
                    confidence=0.0,
                    language=language,
                    word_count=0,
                    processing_time_seconds=elapsed,
                ),
                platform_metadata=_build_platform_metadata(item),
            )

        avg_confidence = (
            sum(s.confidence for s in segments) / len(segments)
            if segments
            else 0.0
        )

        transcript = Transcript(
            content_id=content_id,
            content_type="video",
            language=language,
            segments=segments,
            full_text=full_text,
        )

        return ExtractionResult(
            content_id=content_id,
            content_type="video",
            raw_text=full_text,
            transcript=transcript,
            quality=QualityMetadata(
                confidence=avg_confidence,
                language=language,
                word_count=_compute_word_count(full_text),
                processing_time_seconds=elapsed,
            ),
            platform_metadata=_build_platform_metadata(item),
        )

    def extract(
        self, content_dir: Path, config: ExtractorConfig
    ) -> ExtractionResult:
        """Extract transcript from video files.

        Pipeline: find video -> probe -> extract audio -> transcribe -> result.

        If transcript.json already exists, reuses it instead of re-running
        Whisper (which can take 30+ minutes for long videos on CPU).
        Use --force to bypass this cache and re-transcribe.

        Videos with no audio stream return a result with empty transcript
        and confidence=0.0 (not an error).
        """
        item = load_content_item(content_dir)
        start_time = time.monotonic()

        # Reuse existing transcript.json if available (skip expensive Whisper)
        existing_transcript = content_dir / "transcript.json"
        if existing_transcript.exists() and not config.force_reprocess:
            try:
                result = self._load_from_transcript_json(
                    existing_transcript, item, start_time, content_dir
                )
                logger.info("Reusing existing transcript.json for %s", item.content_id)
                return result
            except Exception as exc:
                logger.warning(
                    "Failed to load cached transcript.json for %s: %s. Re-transcribing.",
                    item.content_id, exc,
                )

        video_path = _find_video_file(content_dir)

        # Probe for audio stream
        probe = probe_audio_stream(video_path)

        if not probe.has_audio:
            elapsed = time.monotonic() - start_time
            logger.info("No audio stream in %s", video_path)
            return ExtractionResult(
                content_id=item.content_id,
                content_type="video",
                raw_text="",
                transcript=None,
                quality=QualityMetadata(
                    confidence=0.0,
                    language="zh",
                    word_count=0,
                    processing_time_seconds=elapsed,
                ),
                platform_metadata=_build_platform_metadata(item),
            )

        # Extract audio to temp WAV
        wav_path = content_dir / "media" / ".tmp_audio.wav"
        normalized_path = content_dir / "media" / ".tmp_normalized.wav"

        try:
            extract_audio(video_path, wav_path)

            # Normalize audio volume (non-fatal on failure)
            audio_path = wav_path
            try:
                normalize_audio(wav_path, normalized_path)
                audio_path = normalized_path
            except FFmpegError:
                logger.warning(
                    "Audio normalization failed for %s, using unnormalized audio",
                    video_path,
                )

            transcription = transcribe_audio(
                audio_path,
                whisper_model=config.whisper_model,
            )
        finally:
            # Always clean up temp files
            for tmp in (wav_path, normalized_path):
                if tmp.exists():
                    tmp.unlink()

        segments = transcription.segments
        speech_ratio = transcription.speech_ratio
        elapsed = time.monotonic() - start_time
        platform_meta = _build_platform_metadata(item)

        # Gate on speech ratio: skip transcription for non-speech audio
        if speech_ratio < 0.10:
            logger.info(
                "Low speech ratio (%.2f) for %s, skipping transcript",
                speech_ratio,
                video_path,
            )
            platform_meta["low_speech_ratio"] = f"{speech_ratio:.2f}"
            return ExtractionResult(
                content_id=item.content_id,
                content_type="video",
                raw_text="",
                transcript=None,
                quality=QualityMetadata(
                    confidence=0.0,
                    language="zh",
                    word_count=0,
                    processing_time_seconds=elapsed,
                ),
                platform_metadata=platform_meta,
            )

        # Flag suspicious segments via hallucination heuristics
        flagged_segments = tuple(
            seg.model_copy(update={"is_suspicious": True})
            if check_segment_suspicious(seg)
            else seg
            for seg in segments
        )

        # Generate hallucination warnings
        warnings = check_transcript_hallucinations(
            flagged_segments, speech_ratio
        )

        # Build transcript
        full_text = " ".join(seg.text for seg in flagged_segments)
        avg_confidence = (
            sum(s.confidence for s in flagged_segments) / len(flagged_segments)
            if flagged_segments
            else 0.0
        )

        # Halve confidence if all segments are suspicious
        if (
            flagged_segments
            and all(s.is_suspicious for s in flagged_segments)
        ):
            avg_confidence = avg_confidence / 2.0

        word_count = _compute_word_count(full_text)

        transcript = Transcript(
            content_id=item.content_id,
            content_type="video",
            language="zh",
            segments=flagged_segments,
            full_text=full_text,
        )

        quality = QualityMetadata(
            confidence=avg_confidence,
            language="zh",
            word_count=word_count,
            processing_time_seconds=elapsed,
            hallucination_warnings=warnings,
        )

        return ExtractionResult(
            content_id=item.content_id,
            content_type="video",
            raw_text=full_text,
            transcript=transcript,
            quality=quality,
            platform_metadata=platform_meta,
        )
