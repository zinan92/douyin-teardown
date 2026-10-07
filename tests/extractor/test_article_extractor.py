"""Tests for ArticleExtractor adapter.

Covers requirements:
    ARTC-01: HTML -> clean structured markdown
    ARTC-02: Preserve headings, lists, emphasis
    ARTC-03: Extract metadata (author, date, word count)
    E2E: Full pipeline produces output files
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from content_extractor.adapters.article import ArticleExtractor
from content_extractor.config import ExtractorConfig
from content_extractor.extract import extract_content

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_HTML = FIXTURES_DIR / "sample_article.html"
WECHAT_ARTICLE_JSON = FIXTURES_DIR / "wechat_article.json"


def _setup_content_dir(
    tmp_path: Path,
    *,
    html_content: str | None = None,
    html_fixture: Path | None = None,
) -> Path:
    """Set up a content directory with content_item.json and media/article.html."""
    content_dir = tmp_path / "wechat_oa" / "gh_abc123def456" / "wechat_oa_article_20260325"
    content_dir.mkdir(parents=True)

    # Copy content_item.json from fixture
    shutil.copy(WECHAT_ARTICLE_JSON, content_dir / "content_item.json")

    # Create media directory and HTML file
    media_dir = content_dir / "media"
    media_dir.mkdir()

    if html_fixture is not None:
        shutil.copy(html_fixture, media_dir / "article.html")
    elif html_content is not None:
        (media_dir / "article.html").write_text(html_content, encoding="utf-8")
    else:
        shutil.copy(SAMPLE_HTML, media_dir / "article.html")

    return content_dir


class TestArticleClean:
    """ARTC-01: HTML with paragraphs and boilerplate -> clean markdown text."""

    def test_html_produces_clean_markdown(self, tmp_path: Path) -> None:
        content_dir = _setup_content_dir(tmp_path, html_fixture=SAMPLE_HTML)
        extractor = ArticleExtractor()
        result = extractor.extract(content_dir, ExtractorConfig())

        # Should contain article body text
        assert "人工智能" in result.raw_text
        assert "量子计算" in result.raw_text

        # Should NOT contain boilerplate/scripts/nav
        assert "<script" not in result.raw_text
        assert "<nav" not in result.raw_text
        assert "sidebar" not in result.raw_text.lower()
        assert "广告位招租" not in result.raw_text

    def test_empty_html_returns_empty_result(self, tmp_path: Path) -> None:
        minimal_html = "<html><body></body></html>"
        content_dir = _setup_content_dir(tmp_path, html_content=minimal_html)
        extractor = ArticleExtractor()
        result = extractor.extract(content_dir, ExtractorConfig())

        assert result.raw_text == ""
        assert result.quality.confidence == 0.0


class TestArticleStructure:
    """ARTC-02: HTML structure -> markdown headings, lists, emphasis."""

    def setup_method(self, method: object) -> None:
        """Extract once and reuse the result across structure tests."""
        self._result = None

    def _get_result(self, tmp_path: Path) -> None:
        if self._result is None:
            content_dir = _setup_content_dir(tmp_path, html_fixture=SAMPLE_HTML)
            extractor = ArticleExtractor()
            self._result = extractor.extract(content_dir, ExtractorConfig())

    def test_headings_preserved(self, tmp_path: Path) -> None:
        self._get_result(tmp_path)
        # Markdown heading markers
        assert "# " in self._result.raw_text or "## " in self._result.raw_text

    def test_lists_preserved(self, tmp_path: Path) -> None:
        self._get_result(tmp_path)
        # Markdown list markers
        assert "- " in self._result.raw_text or "* " in self._result.raw_text

    def test_emphasis_preserved(self, tmp_path: Path) -> None:
        self._get_result(tmp_path)
        # Bold or italic markers
        assert "**" in self._result.raw_text or "*" in self._result.raw_text


class TestArticleMetadata:
    """ARTC-03: Extract metadata (author, date, title, word count)."""

    def test_author_extracted(self, tmp_path: Path) -> None:
        content_dir = _setup_content_dir(tmp_path, html_fixture=SAMPLE_HTML)
        extractor = ArticleExtractor()
        result = extractor.extract(content_dir, ExtractorConfig())

        assert isinstance(result.platform_metadata.get("extracted_author"), str)
        assert len(result.platform_metadata["extracted_author"]) > 0

    def test_date_extracted(self, tmp_path: Path) -> None:
        content_dir = _setup_content_dir(tmp_path, html_fixture=SAMPLE_HTML)
        extractor = ArticleExtractor()
        result = extractor.extract(content_dir, ExtractorConfig())

        assert isinstance(result.platform_metadata.get("extracted_date"), str)
        assert len(result.platform_metadata["extracted_date"]) > 0

    def test_title_extracted(self, tmp_path: Path) -> None:
        content_dir = _setup_content_dir(tmp_path, html_fixture=SAMPLE_HTML)
        extractor = ArticleExtractor()
        result = extractor.extract(content_dir, ExtractorConfig())

        assert isinstance(result.platform_metadata.get("extracted_title"), str)
        assert len(result.platform_metadata["extracted_title"]) > 0

    def test_word_count_chinese(self, tmp_path: Path) -> None:
        content_dir = _setup_content_dir(tmp_path, html_fixture=SAMPLE_HTML)
        extractor = ArticleExtractor()
        result = extractor.extract(content_dir, ExtractorConfig())

        # Fixture has 200+ CJK characters; CJK-aware counting must produce >= 100
        assert result.quality.word_count >= 100

    def test_content_type_is_article(self, tmp_path: Path) -> None:
        content_dir = _setup_content_dir(tmp_path, html_fixture=SAMPLE_HTML)
        extractor = ArticleExtractor()
        result = extractor.extract(content_dir, ExtractorConfig())

        assert result.content_type == "article"


class TestArticleEndToEnd:
    """Full pipeline via extract_content() -> output files."""

    def test_full_pipeline_produces_output_files(self, tmp_path: Path) -> None:
        content_dir = _setup_content_dir(tmp_path, html_fixture=SAMPLE_HTML)

        result = extract_content(content_dir)
        assert result is not None

        # All three output files must exist
        assert (content_dir / "transcript.json").exists()
        assert (content_dir / "analysis.json").exists()
        assert (content_dir / "structured_text.md").exists()

        # Idempotency marker must exist
        assert (content_dir / ".extraction_complete").exists()

        # Verify transcript.json contains the raw text
        transcript_data = json.loads((content_dir / "transcript.json").read_bytes())
        assert "full_text" in transcript_data
        assert len(transcript_data["full_text"]) > 0

    def test_idempotency_skips_reextraction(self, tmp_path: Path) -> None:
        content_dir = _setup_content_dir(tmp_path, html_fixture=SAMPLE_HTML)

        # First extraction
        first_result = extract_content(content_dir)
        assert first_result is not None

        # Second extraction should be skipped (returns None)
        second_result = extract_content(content_dir)
        assert second_result is None
