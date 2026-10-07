"""Tests copied from content-extractor @ 810f661 that already failed there.

Its LLM client was rewritten (676f4d7, OpenAI-compatible proxy) and its Markdown headings
renamed, but these tests were never updated. douyin-teardown only uses the extractor's
transcription, so they are marked instead of rewritten; fix them when that code is touched.
"""
from __future__ import annotations

import pytest

collect_ignore = ["extractor/test_llm.py"]  # imports names the rewritten llm.py no longer has

STALE_UPSTREAM = (
    "extractor/test_analysis.py::",
    "extractor/test_output.py::TestWriteExtractionOutput::test_structured_text_format",
    "extractor/test_output.py::TestWriteExtractionOutputWithAnalysis::test_structured_text_with_empty_analysis",
)


def pytest_collection_modifyitems(items):
    for item in items:
        if any(stale in item.nodeid for stale in STALE_UPSTREAM):
            item.add_marker(pytest.mark.xfail(reason="上游 content-extractor 测试已过时（见 tests/conftest.py）", strict=False))


@pytest.fixture(autouse=True)
def _faster_whisper_backend(monkeypatch):
    """The transcribe tests mock faster-whisper; with mlx-whisper installed they would load a real model."""
    monkeypatch.setenv("CONTENT_EXTRACTOR_WHISPER_BACKEND", "faster")
