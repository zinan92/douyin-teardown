"""Unit tests for the hallucination detection module.

Tests each heuristic in isolation and the orchestrator function.
"""

from __future__ import annotations

import pytest

from content_extractor.models import TranscriptSegment
from content_extractor.video.hallucination import (
    check_segment_suspicious,
    check_transcript_hallucinations,
    detect_repeated_ngrams,
)


# ---------------------------------------------------------------------------
# check_segment_suspicious tests
# ---------------------------------------------------------------------------


class TestCheckSegmentSuspicious:
    """Tests for check_segment_suspicious()."""

    def test_low_confidence_is_suspicious(self) -> None:
        """Segment with confidence < 0.4 is suspicious."""
        seg = TranscriptSegment(
            text="some text", start=0.0, end=2.0, confidence=0.3
        )
        assert check_segment_suspicious(seg) is True

    def test_high_cjk_chars_per_second_is_suspicious(self) -> None:
        """Segment with > 6 CJK chars/sec is suspicious."""
        # 20 CJK chars in 2 seconds = 10 chars/sec > 6.0
        text = "这是一个测试文本用于检测" * 2  # 24 CJK chars
        seg = TranscriptSegment(
            text=text, start=0.0, end=2.0, confidence=0.8
        )
        assert check_segment_suspicious(seg) is True

    def test_normal_segment_not_suspicious(self) -> None:
        """Segment with good confidence and normal speed is not suspicious."""
        seg = TranscriptSegment(
            text="正常文本", start=0.0, end=5.0, confidence=0.8
        )
        assert check_segment_suspicious(seg) is False

    def test_boundary_confidence_not_suspicious(self) -> None:
        """Segment with confidence exactly 0.4 is not suspicious."""
        seg = TranscriptSegment(
            text="边界值", start=0.0, end=5.0, confidence=0.4
        )
        assert check_segment_suspicious(seg) is False

    def test_zero_duration_not_division_error(self) -> None:
        """Segment with zero duration does not cause division by zero."""
        seg = TranscriptSegment(
            text="短", start=0.0, end=0.0, confidence=0.8
        )
        # Should not raise; behavior depends on implementation
        check_segment_suspicious(seg)

    def test_latin_text_not_counted_as_cjk(self) -> None:
        """Latin text is not counted toward CJK chars/sec."""
        seg = TranscriptSegment(
            text="This is a long English sentence with many words",
            start=0.0,
            end=1.0,
            confidence=0.8,
        )
        assert check_segment_suspicious(seg) is False


# ---------------------------------------------------------------------------
# detect_repeated_ngrams tests
# ---------------------------------------------------------------------------


class TestDetectRepeatedNgrams:
    """Tests for detect_repeated_ngrams()."""

    def test_detects_repeated_text(self) -> None:
        """Text with repeated 4-grams is detected."""
        text = "这是一个测试" * 5  # same 6 chars repeated 5 times
        result = detect_repeated_ngrams(text)
        assert len(result) > 0

    def test_no_repeats_in_unique_text(self) -> None:
        """Unique text produces no repeated n-grams."""
        text = "一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾"
        result = detect_repeated_ngrams(text)
        assert len(result) == 0

    def test_custom_n_and_threshold(self) -> None:
        """Custom n and threshold parameters work."""
        text = "AB" * 10  # "AB" repeated 10 times
        result = detect_repeated_ngrams(text, n=2, threshold=5)
        assert len(result) > 0

    def test_empty_text_returns_empty(self) -> None:
        """Empty text returns empty tuple."""
        result = detect_repeated_ngrams("")
        assert result == ()

    def test_short_text_returns_empty(self) -> None:
        """Text shorter than n returns empty."""
        result = detect_repeated_ngrams("短")
        assert result == ()


# ---------------------------------------------------------------------------
# check_transcript_hallucinations tests
# ---------------------------------------------------------------------------


class TestCheckTranscriptHallucinations:
    """Tests for check_transcript_hallucinations()."""

    def test_low_speech_ratio_produces_warning(self) -> None:
        """speech_ratio < 0.10 produces speech warning."""
        segments = (
            TranscriptSegment(
                text="text", start=0.0, end=1.0, confidence=0.8
            ),
        )
        warnings = check_transcript_hallucinations(segments, 0.05)
        assert any("10% speech" in w for w in warnings)

    def test_low_avg_confidence_produces_warning(self) -> None:
        """Average confidence < 0.5 produces confidence warning."""
        segments = (
            TranscriptSegment(
                text="text", start=0.0, end=1.0, confidence=0.3
            ),
            TranscriptSegment(
                text="more", start=1.0, end=2.0, confidence=0.2
            ),
        )
        warnings = check_transcript_hallucinations(segments, 0.8)
        assert any("confidence" in w.lower() for w in warnings)

    def test_suspicious_segments_produce_warning(self) -> None:
        """Suspicious segments produce count warning."""
        segments = (
            TranscriptSegment(
                text="text", start=0.0, end=1.0, confidence=0.2
            ),
            TranscriptSegment(
                text="good", start=1.0, end=2.0, confidence=0.9
            ),
        )
        warnings = check_transcript_hallucinations(segments, 0.8)
        assert any("suspicious" in w.lower() for w in warnings)

    def test_repetitive_text_produces_warning(self) -> None:
        """Repeated text across segments produces repetition warning."""
        repeated = "这是一个测试" * 5
        segments = (
            TranscriptSegment(
                text=repeated, start=0.0, end=10.0, confidence=0.8
            ),
        )
        warnings = check_transcript_hallucinations(segments, 0.8)
        assert any("repetitive" in w.lower() for w in warnings)

    def test_clean_transcript_no_warnings(self) -> None:
        """Clean transcript with good metrics produces no warnings."""
        segments = (
            TranscriptSegment(
                text="第一句话", start=0.0, end=3.0, confidence=0.9
            ),
            TranscriptSegment(
                text="第二句话不同", start=3.0, end=6.0, confidence=0.85
            ),
        )
        warnings = check_transcript_hallucinations(segments, 0.8)
        assert warnings == ()

    def test_empty_segments_no_crash(self) -> None:
        """Empty segments tuple does not crash."""
        warnings = check_transcript_hallucinations((), 0.8)
        assert isinstance(warnings, tuple)

    def test_returns_tuple_of_strings(self) -> None:
        """Return type is tuple[str, ...]."""
        segments = (
            TranscriptSegment(
                text="text", start=0.0, end=1.0, confidence=0.2
            ),
        )
        warnings = check_transcript_hallucinations(segments, 0.05)
        assert isinstance(warnings, tuple)
        assert all(isinstance(w, str) for w in warnings)
