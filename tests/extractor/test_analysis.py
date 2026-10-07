"""Tests for LLM analysis module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from content_extractor.config import ExtractorConfig
from content_extractor.models import AnalysisResult, SentimentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_LLM_JSON = """{
  "topics": ["AI regulation", "tech policy", "EU compliance"],
  "viewpoints": ["Regulation is necessary for safety", "Over-regulation stifles innovation"],
  "sentiment": {"overall": "mixed", "confidence": 0.82},
  "takeaways": ["Monitor EU AI Act timelines", "Prepare compliance roadmap"]
}"""


def _mock_llm_response(text: str) -> MagicMock:
    """Build a mock anthropic Message with a single text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    message = MagicMock()
    message.content = [block]
    return message


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnalyzeContentHappyPath:
    """Test 1: valid LLM response returns populated AnalysisResult."""

    def test_valid_response_returns_analysis_result(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_llm_response(_VALID_LLM_JSON)

        with patch(
            "content_extractor.analysis.create_claude_client",
            return_value=mock_client,
        ):
            from content_extractor.analysis import analyze_content

            result = analyze_content(
                "Some article text about AI regulation.",
                content_id="item1",
                content_type="article",
            )

        assert isinstance(result, AnalysisResult)
        assert result.content_id == "item1"
        assert result.content_type == "article"
        assert len(result.topics) == 3
        assert "AI regulation" in result.topics
        assert len(result.viewpoints) == 2
        assert result.sentiment is not None
        assert result.sentiment.overall == "mixed"
        assert result.sentiment.confidence == pytest.approx(0.82)
        assert len(result.takeaways) == 2


class TestAnalyzeContentMalformedJSON:
    """Test 2: malformed JSON returns fallback AnalysisResult."""

    def test_malformed_json_returns_fallback(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_llm_response(
            "This is not JSON at all!"
        )

        with patch(
            "content_extractor.analysis.create_claude_client",
            return_value=mock_client,
        ):
            from content_extractor.analysis import analyze_content

            result = analyze_content(
                "Some text",
                content_id="item2",
                content_type="video",
            )

        assert isinstance(result, AnalysisResult)
        assert result.content_id == "item2"
        assert result.content_type == "video"
        assert result.topics == ()
        assert result.viewpoints == ()
        assert result.sentiment is None
        assert result.takeaways == ()


class TestAnalyzeContentEmptyInput:
    """Test 3: empty input returns fallback without LLM call."""

    def test_empty_text_skips_llm(self) -> None:
        mock_client = MagicMock()

        with patch(
            "content_extractor.analysis.create_claude_client",
            return_value=mock_client,
        ):
            from content_extractor.analysis import analyze_content

            result = analyze_content(
                "   ",
                content_id="item3",
                content_type="article",
            )

        assert isinstance(result, AnalysisResult)
        assert result.content_id == "item3"
        assert result.topics == ()
        mock_client.messages.create.assert_not_called()

    def test_whitespace_only_skips_llm(self) -> None:
        mock_client = MagicMock()

        with patch(
            "content_extractor.analysis.create_claude_client",
            return_value=mock_client,
        ):
            from content_extractor.analysis import analyze_content

            result = analyze_content(
                "\n\t  \n",
                content_id="item4",
                content_type="video",
            )

        assert isinstance(result, AnalysisResult)
        mock_client.messages.create.assert_not_called()


class TestAnalyzeContentPassthrough:
    """Test 4: content_id and content_type pass through to result."""

    def test_ids_pass_through(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_llm_response(_VALID_LLM_JSON)

        with patch(
            "content_extractor.analysis.create_claude_client",
            return_value=mock_client,
        ):
            from content_extractor.analysis import analyze_content

            result = analyze_content(
                "Text",
                content_id="unique-id-789",
                content_type="gallery",
            )

        assert result.content_id == "unique-id-789"
        assert result.content_type == "gallery"


class TestAnalyzeContentConfig:
    """Test 5: config values are used in API call."""

    def test_config_used_in_api_call(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_llm_response(_VALID_LLM_JSON)
        config = ExtractorConfig(
            claude_model="claude-test-model",
            claude_max_tokens=2048,
            claude_temperature=0.5,
        )

        with patch(
            "content_extractor.analysis.create_claude_client",
            return_value=mock_client,
        ):
            from content_extractor.analysis import analyze_content

            analyze_content(
                "Some text",
                content_id="item5",
                content_type="article",
                config=config,
            )

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-test-model"
        assert call_kwargs.kwargs["max_tokens"] == 2048
        assert call_kwargs.kwargs["temperature"] == pytest.approx(0.5)


class TestAnalyzeContentAPIError:
    """Test 6: API exception is wrapped in AnalysisError."""

    def test_api_error_raises_analysis_error(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API connection failed")

        with patch(
            "content_extractor.analysis.create_claude_client",
            return_value=mock_client,
        ):
            from content_extractor.analysis import AnalysisError, analyze_content

            with pytest.raises(AnalysisError, match="API connection failed"):
                analyze_content(
                    "Some text",
                    content_id="item6",
                    content_type="article",
                )
