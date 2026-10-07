"""Top-level extraction orchestration.

``extract_content`` processes a single ContentItem directory through the
loader -> router -> adapter -> output writer pipeline.

``extract_batch`` scans a parent directory for ContentItem dirs and processes
each one, isolating per-item errors so a single failure does not abort the
batch (D-07, QUAL-01).  Errors are printed to stderr (D-09).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from content_extractor.analysis import AnalysisError, analyze_content, restructure_transcript
from content_extractor.config import ExtractorConfig
from content_extractor.loader import load_content_item
from content_extractor.models import AnalysisResult, ExtractionResult
from content_extractor.output import is_extracted, write_extraction_output
from content_extractor.router import get_extractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchError:
    """Captures a per-item extraction failure."""

    content_dir: str
    error: str


@dataclass(frozen=True)
class BatchResult:
    """Aggregate result from batch extraction."""

    succeeded: tuple[ExtractionResult, ...]
    failed: tuple[BatchError, ...]
    total: int
    success_count: int
    failure_count: int


def extract_content(
    content_dir: Path,
    config: ExtractorConfig | None = None,
) -> ExtractionResult | None:
    """Extract content from a single ContentItem directory.

    Pipeline: load -> route -> adapt -> write output.

    Returns the ExtractionResult on success, or None if the item was
    already extracted and ``config.force_reprocess`` is False.
    """
    config = config if config is not None else ExtractorConfig()

    # Skip already-extracted unless force
    if is_extracted(content_dir) and not config.force_reprocess:
        return None

    # Load and validate ContentItem
    item = load_content_item(content_dir)

    # Route to the correct adapter
    adapter = get_extractor(item.content_type)

    # Run extraction
    result = adapter.extract(content_dir, config)

    if not config.llm_enabled:
        write_extraction_output(
            content_dir, result, item,
            force=config.force_reprocess,
            analysis=AnalysisResult(content_id=result.content_id, content_type=result.content_type),
            analysis_degraded=True,
        )
        return result

    # Run LLM transcript restructuring
    structured_transcript = None
    try:
        structured_transcript = restructure_transcript(
            raw_text=result.raw_text,
            config=config,
        )
        if structured_transcript:
            logger.info("Transcript restructured successfully for %s", result.content_id)
    except Exception as exc:
        print(
            f"⚠️  Transcript restructuring failed for {result.content_id}: {exc}\n"
            f"   Raw transcript will be used instead.",
            file=sys.stderr,
        )

    # Run LLM analysis
    analysis_degraded = False
    try:
        analysis = analyze_content(
            raw_text=result.raw_text,
            content_id=result.content_id,
            content_type=result.content_type,
            config=config,
        )
    except AnalysisError as exc:
        print(
            f"⚠️  LLM analysis failed for {result.content_id}: {exc}\n"
            f"   Transcript extracted successfully, but summary/takeaways will be empty.",
            file=sys.stderr,
        )
        analysis_degraded = True
        analysis = AnalysisResult(
            content_id=result.content_id,
            content_type=result.content_type,
        )

    # Write output files
    write_extraction_output(
        content_dir, result, item,
        force=config.force_reprocess,
        analysis=analysis,
        analysis_degraded=analysis_degraded,
        structured_transcript=structured_transcript,
    )

    return result


def extract_batch(
    parent_dir: Path,
    config: ExtractorConfig | None = None,
) -> BatchResult:
    """Extract content from all ContentItem directories under *parent_dir*.

    Scans recursively for directories containing ``content_item.json``.
    Per-item errors are isolated -- a failing item does not abort the batch
    (D-07, QUAL-01).  Errors are printed to stderr (D-09).
    """
    config = config if config is not None else ExtractorConfig()

    content_dirs = sorted(
        p.parent for p in parent_dir.rglob("content_item.json")
    )

    succeeded: list[ExtractionResult] = []
    failed: list[BatchError] = []

    for content_dir in content_dirs:
        try:
            result = extract_content(content_dir, config)
            if result is not None:
                succeeded.append(result)
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR extracting {content_dir}: {exc}",
                file=sys.stderr,
            )
            failed.append(
                BatchError(
                    content_dir=str(content_dir),
                    error=str(exc),
                )
            )

    total = len(content_dirs)
    return BatchResult(
        succeeded=tuple(succeeded),
        failed=tuple(failed),
        total=total,
        success_count=len(succeeded),
        failure_count=len(failed),
    )
