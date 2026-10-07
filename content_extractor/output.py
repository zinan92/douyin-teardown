"""Atomic output writer with idempotency guard.

Writes transcript.json, analysis.json, and structured_text.md to a content
directory using atomic temp-file-then-rename. Uses .extraction_complete marker
for idempotency -- skips already-extracted items unless force=True.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from content_extractor.models import AnalysisResult, ExtractionResult

if TYPE_CHECKING:
    from content_extractor.models import ContentItem

MARKER_FILE = ".extraction_complete"


# ---------------------------------------------------------------------------
# Atomic writers
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, data: bytes) -> None:
    """Write JSON bytes atomically via temp file + rename.

    On failure, the temp file is cleaned up (QUAL-03).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_bytes(data)
        tmp.rename(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text atomically via temp file + rename.

    On failure, the temp file is cleaned up (QUAL-03).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.rename(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Idempotency marker
# ---------------------------------------------------------------------------


def is_extracted(content_dir: Path) -> bool:
    """Check if content has already been extracted."""
    return (content_dir / MARKER_FILE).exists()


def mark_complete(content_dir: Path) -> None:
    """Mark content as extracted by creating the completion marker."""
    (content_dir / MARKER_FILE).touch()


def clear_marker(content_dir: Path) -> None:
    """Remove the extraction marker (for --force reprocessing)."""
    (content_dir / MARKER_FILE).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Structured text rendering (D-04)
# ---------------------------------------------------------------------------


def _render_structured_text(
    result: ExtractionResult,
    *,
    item_title: str,
    item_author: str,
    item_platform: str,
    item_publish_time: str,
    item_likes: int = 0,
    item_comments: int = 0,
    item_shares: int = 0,
    analysis: AnalysisResult | None = None,
    structured_transcript: str | None = None,
) -> str:
    """Render structured_text.md in D-04 report format.

    When *analysis* is provided with populated fields, renders real
    Summary, Key Takeaways, and Analysis sections. Otherwise falls back
    to placeholder text.
    """
    header = (
        f"# {item_title}\n"
        f"\n"
        f"**Author:** {item_author} | **Platform:** {item_platform}"
        f" | **Published:** {item_publish_time}\n"
        f"**Engagement:** {item_likes} likes, {item_comments} comments,"
        f" {item_shares} shares\n"
    )

    # Determine if analysis has any real content
    has_analysis = (
        analysis is not None
        and (analysis.topics or analysis.viewpoints or analysis.takeaways or analysis.sentiment)
    )

    # Summary section
    if has_analysis and analysis is not None and analysis.topics:
        summary = f"This content covers: {', '.join(analysis.topics)}."
    else:
        summary = "No analysis available."

    # Key Takeaways section
    if has_analysis and analysis is not None and analysis.takeaways:
        takeaway_lines = "\n".join(f"- {t}" for t in analysis.takeaways)
    else:
        takeaway_lines = "No takeaways identified."

    # Analysis section
    if has_analysis and analysis is not None:
        analysis_parts: list[str] = []

        if analysis.topics:
            analysis_parts.append(f"**Topics:** {', '.join(analysis.topics)}")

        if analysis.viewpoints:
            vp_lines = "\n".join(f"- {v}" for v in analysis.viewpoints)
            analysis_parts.append(f"**Viewpoints:**\n{vp_lines}")

        if analysis.sentiment is not None:
            analysis_parts.append(
                f"**Sentiment:** Overall: {analysis.sentiment.overall}"
                f" (confidence: {analysis.sentiment.confidence})"
            )

        analysis_text = "\n\n".join(analysis_parts) if analysis_parts else "No analysis available."
    else:
        analysis_text = "No analysis available."

    # Use structured transcript if available, fall back to raw text
    content_section = structured_transcript if structured_transcript else result.raw_text

    return (
        f"{header}"
        f"\n"
        f"## Summary\n"
        f"\n"
        f"{summary}\n"
        f"\n"
        f"## Key Takeaways\n"
        f"\n"
        f"{takeaway_lines}\n"
        f"\n"
        f"## Content\n"
        f"\n"
        f"{content_section}\n"
        f"\n"
        f"## Analysis\n"
        f"\n"
        f"{analysis_text}\n"
    )


# ---------------------------------------------------------------------------
# High-level output writer
# ---------------------------------------------------------------------------


def write_extraction_output(
    content_dir: Path,
    result: ExtractionResult,
    content_item: "ContentItem",
    *,
    force: bool = False,
    analysis: AnalysisResult | None = None,
    analysis_degraded: bool = False,
    structured_transcript: str | None = None,
) -> bool:
    """Write all extraction output files to content_dir.

    Returns True if files were written, False if skipped (already extracted).

    When *analysis* is provided, writes real analysis data; otherwise writes
    a placeholder. When force=True, clears the marker and re-writes all
    files (FOUND-05). When *analysis_degraded* is True, writes an
    extraction_status.json signaling that LLM analysis was skipped.
    """
    if is_extracted(content_dir) and not force:
        return False

    if force:
        clear_marker(content_dir)

    # transcript.json
    if result.transcript is not None:
        transcript_data = orjson.dumps(
            result.transcript.model_dump(), option=orjson.OPT_INDENT_2
        )
    else:
        transcript_data = orjson.dumps(
            {
                "content_id": result.content_id,
                "content_type": result.content_type,
                "full_text": result.raw_text,
            },
            option=orjson.OPT_INDENT_2,
        )
    write_json_atomic(content_dir / "transcript.json", transcript_data)

    # analysis.json -- use provided analysis or fallback to placeholder
    effective_analysis = analysis if analysis is not None else AnalysisResult(
        content_id=result.content_id,
        content_type=result.content_type,
    )
    analysis_data = orjson.dumps(
        effective_analysis.model_dump(), option=orjson.OPT_INDENT_2
    )
    write_json_atomic(content_dir / "analysis.json", analysis_data)

    # structured_text.md (D-04)
    md_text = _render_structured_text(
        result,
        item_title=content_item.title,
        item_author=content_item.author_name,
        item_platform=content_item.platform,
        item_publish_time=content_item.publish_time,
        item_likes=content_item.likes,
        item_comments=content_item.comments,
        item_shares=content_item.shares,
        analysis=effective_analysis,
        structured_transcript=structured_transcript,
    )
    write_text_atomic(content_dir / "structured_text.md", md_text)

    # extractor_output.json — rewriter-compatible handoff format
    extractor_output = {
        "content_id": result.content_id,
        "source_platform": content_item.platform,
        "title": content_item.title,
        "transcript": result.raw_text,
        "key_points": list(effective_analysis.takeaways),
        "visual_descriptions": [
            desc.description for desc in result.media_descriptions
        ],
        "metadata": {
            "duration_seconds": int(result.quality.processing_time_seconds) if result.transcript else None,
            "publish_date": content_item.publish_time[:10] if content_item.publish_time else None,
            "engagement": {
                "views": content_item.views,
                "likes": content_item.likes,
                "comments": content_item.comments,
                "shares": content_item.shares,
                "collects": content_item.collects,
            },
        },
    }
    write_json_atomic(
        content_dir / "extractor_output.json",
        orjson.dumps(extractor_output, option=orjson.OPT_INDENT_2),
    )

    # extraction_status.json — signals degraded extraction to downstream tools
    status_data = orjson.dumps(
        {
            "content_id": result.content_id,
            "transcript": "ok",
            "analysis": "degraded" if analysis_degraded else "ok",
        },
        option=orjson.OPT_INDENT_2,
    )
    write_json_atomic(content_dir / "extraction_status.json", status_data)

    mark_complete(content_dir)
    return True
