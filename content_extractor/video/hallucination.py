"""Hallucination detection heuristics for Whisper transcriptions.

Whisper generates confident-sounding fabricated text on silence, music,
and background noise. This module provides heuristics to flag unreliable
segments and generate transcript-level quality warnings.

Exports:
    check_segment_suspicious: Per-segment suspicion check.
    detect_repeated_ngrams: Find repeated character n-grams in text.
    check_transcript_hallucinations: Orchestrate all checks, return warnings.
"""

from __future__ import annotations

import re
from collections import Counter

from content_extractor.models import TranscriptSegment

# CJK character regex for counting
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# Thresholds
_CONFIDENCE_THRESHOLD = 0.4
_CJK_CHARS_PER_SEC_THRESHOLD = 6.0
_AVG_CONFIDENCE_THRESHOLD = 0.5
_SPEECH_RATIO_THRESHOLD = 0.10


def check_segment_suspicious(segment: TranscriptSegment) -> bool:
    """Check if a single transcript segment looks suspicious.

    A segment is suspicious if:
    - confidence < 0.4, OR
    - CJK characters per second > 6.0 (impossibly fast speech)

    Parameters
    ----------
    segment:
        A transcript segment to evaluate.

    Returns
    -------
    bool
        True if the segment is suspicious.
    """
    if segment.confidence < _CONFIDENCE_THRESHOLD:
        return True

    duration = segment.end - segment.start
    if duration <= 0:
        return False

    cjk_count = len(_CJK_PATTERN.findall(segment.text))
    chars_per_sec = cjk_count / duration

    return chars_per_sec > _CJK_CHARS_PER_SEC_THRESHOLD


def detect_repeated_ngrams(
    text: str,
    *,
    n: int = 4,
    threshold: int = 3,
) -> tuple[str, ...]:
    """Detect repeated character n-grams in text.

    Splits text into overlapping character n-grams and returns
    those appearing at or above the threshold count.

    Parameters
    ----------
    text:
        The text to analyze.
    n:
        N-gram size (default 4 characters for CJK).
    threshold:
        Minimum occurrence count to flag (default 3).

    Returns
    -------
    tuple[str, ...]
        Repeated n-gram strings, deduplicated.
    """
    if len(text) < n:
        return ()

    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    counts = Counter(ngrams)

    repeated = tuple(
        gram for gram, count in counts.items() if count >= threshold
    )
    return repeated


def check_transcript_hallucinations(
    segments: tuple[TranscriptSegment, ...],
    speech_ratio: float,
) -> tuple[str, ...]:
    """Run all hallucination heuristics and return warning strings.

    Checks:
    1. Low speech ratio (< 10%)
    2. Low average confidence (< 0.5)
    3. Repeated n-grams across all segment text
    4. Suspicious segment count

    Parameters
    ----------
    segments:
        Transcript segments to analyze.
    speech_ratio:
        VAD-based speech ratio (0.0 to 1.0).

    Returns
    -------
    tuple[str, ...]
        Warning strings for each triggered heuristic.
    """
    warnings: list[str] = []

    # 1. Low speech ratio
    if speech_ratio < _SPEECH_RATIO_THRESHOLD:
        warnings.append(
            "Audio has less than 10% speech -- transcript may be unreliable"
        )

    if not segments:
        return tuple(warnings)

    # 2. Low average confidence
    avg_confidence = sum(s.confidence for s in segments) / len(segments)
    if avg_confidence < _AVG_CONFIDENCE_THRESHOLD:
        warnings.append(
            f"Low overall confidence ({avg_confidence:.2f}) -- review recommended"
        )

    # 3. Repeated n-grams
    full_text = "".join(s.text for s in segments)
    repeated = detect_repeated_ngrams(full_text)
    if repeated:
        phrases = ", ".join(repeated[:5])  # limit display
        warnings.append(
            f"Repetitive text detected -- possible hallucination: {phrases}"
        )

    # 4. Suspicious segment count
    suspicious_count = sum(
        1 for s in segments if check_segment_suspicious(s)
    )
    if suspicious_count > 0:
        warnings.append(
            f"{suspicious_count} of {len(segments)} segments flagged as suspicious"
        )

    return tuple(warnings)
