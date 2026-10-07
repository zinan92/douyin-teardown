"""Content extractor -- turn raw multimedia into structured text."""

from content_extractor.config import ExtractorConfig
from content_extractor.extract import (
    BatchError,
    BatchResult,
    extract_batch,
    extract_content,
)
from content_extractor.loader import (
    ContentItemInvalidError,
    ContentItemNotFoundError,
    load_content_item,
)
from content_extractor.models import ExtractionResult

__version__ = "0.1.0"

# CLI-05: Public API alias -- `extract(path)` delegates to `extract_content`
extract = extract_content

__all__ = [
    "BatchError",
    "BatchResult",
    "ContentItemInvalidError",
    "ContentItemNotFoundError",
    "ExtractionResult",
    "ExtractorConfig",
    "extract",
    "extract_batch",
    "extract_content",
    "load_content_item",
]
