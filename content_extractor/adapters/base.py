"""Extractor Protocol definition.

All content type adapters must satisfy this Protocol via structural subtyping.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from content_extractor.models import ExtractionResult


@runtime_checkable
class Extractor(Protocol):
    """Protocol all content type extractors must satisfy."""

    content_type: str

    def extract(
        self, content_dir: Path, config: "ExtractorConfig"
    ) -> "ExtractionResult":
        """Extract text content from media files in content_dir."""
        ...
