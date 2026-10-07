"""Shared text utility functions for content extractors.

Provides language detection and word counting for both CJK and Latin text.
Extracted from article adapter to be reusable across image and gallery adapters.
"""

from __future__ import annotations

import re


def _compute_word_count(text: str) -> int:
    """Compute word count handling both CJK and Latin text.

    CJK characters are counted individually (each is roughly a word).
    Latin words are counted by matching alphabetic sequences.
    """
    if not text:
        return 0
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
    latin_words = len(re.findall(r"[a-zA-Z]+", text))
    return cjk_chars + latin_words


def _detect_language(text: str) -> str:
    """Simple language detection based on CJK character ratio."""
    if not text:
        return "unknown"
    cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
    if cjk_count > len(text) * 0.1:
        return "zh"
    return "en"
