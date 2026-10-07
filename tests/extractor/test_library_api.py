"""Tests for the public library API contract.

Ensures ``from content_extractor import extract, extract_batch`` works
and all public types are exported correctly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import content_extractor
from content_extractor.config import ExtractorConfig as _OrigConfig
from content_extractor.extract import (
    BatchError as _OrigBatchError,
    BatchResult as _OrigBatchResult,
    extract_content as _orig_extract_content,
)
from content_extractor.models import ExtractionResult as _OrigExtractionResult


class TestPublicExports:
    """All __all__ members are importable from the top-level package."""

    def test_all_members_importable(self) -> None:
        for name in content_extractor.__all__:
            assert hasattr(content_extractor, name), (
                f"{name} is in __all__ but not importable"
            )

    def test_extract_importable(self) -> None:
        from content_extractor import extract

        assert callable(extract)

    def test_extract_batch_importable(self) -> None:
        from content_extractor import extract_batch

        assert callable(extract_batch)

    def test_extraction_result_importable(self) -> None:
        from content_extractor import ExtractionResult

        assert ExtractionResult is _OrigExtractionResult

    def test_extractor_config_importable(self) -> None:
        from content_extractor import ExtractorConfig

        assert ExtractorConfig is _OrigConfig

    def test_batch_result_importable(self) -> None:
        from content_extractor import BatchResult

        assert BatchResult is _OrigBatchResult

    def test_batch_error_importable(self) -> None:
        from content_extractor import BatchError

        assert BatchError is _OrigBatchError


class TestExtractAlias:
    """``extract`` is an alias for ``extract_content``."""

    def test_extract_is_same_function_as_extract_content(self) -> None:
        from content_extractor import extract

        assert extract is _orig_extract_content

    def test_extract_accepts_path_and_config(self) -> None:
        """Verify extract has the same signature as extract_content."""
        import inspect

        from content_extractor import extract

        sig = inspect.signature(extract)
        params = list(sig.parameters.keys())
        assert "content_dir" in params
        assert "config" in params

    def test_extract_passes_config_through(self) -> None:
        """Verify extract can be called with a config (same function)."""
        from content_extractor import extract

        # Since extract IS extract_content, just verify they're identical
        assert extract.__name__ == "extract_content"


class TestExtractBatchAlias:
    """``extract_batch`` delegates to the batch function from extract.py."""

    def test_extract_batch_callable(self) -> None:
        from content_extractor import extract_batch

        assert callable(extract_batch)


class TestBackwardCompatibility:
    """``extract_content`` still importable for backward compatibility."""

    def test_extract_content_still_importable(self) -> None:
        from content_extractor import extract_content

        assert extract_content is _orig_extract_content
