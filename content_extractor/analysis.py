"""LLM-powered content analysis module.

Sends extracted text to Claude (via CLI Proxy API or direct Anthropic API)
and returns a structured AnalysisResult with topics, viewpoints, sentiment,
and takeaways. Also provides transcript restructuring for human-readable output.

Usage:
    from content_extractor.analysis import analyze_content, restructure_transcript
"""

from __future__ import annotations

import logging
import re

import orjson

from content_extractor.config import ExtractorConfig
from content_extractor.llm import LLMAPIError, llm_chat
from content_extractor.models import AnalysisResult, SentimentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class AnalysisError(Exception):
    """Raised when LLM analysis fails (API error, timeout, etc.)."""


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM response text.

    Handles three common LLM response patterns:
    1. Clean JSON: ``{"topics": [...]}``
    2. Markdown-fenced: ``````json\\n{...}\\n``````
    3. JSON embedded in prose: ``Here is the analysis:\\n{...}``

    Returns the parsed dict, or None if extraction fails.
    """
    text = text.strip()

    # Try direct parse first (cheapest path)
    try:
        result = orjson.loads(text)
        if isinstance(result, dict):
            return result
    except (orjson.JSONDecodeError, ValueError, TypeError):
        pass

    # Strip markdown fences
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            result = orjson.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except (orjson.JSONDecodeError, ValueError, TypeError):
            pass

    # Find first { ... last } as a fallback
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            result = orjson.loads(text[first_brace : last_brace + 1])
            if isinstance(result, dict):
                return result
        except (orjson.JSONDecodeError, ValueError, TypeError):
            pass

    return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT = """\
Analyze the following text and return a JSON object with exactly these fields:

{
  "topics": ["topic1", "topic2", ...],
  "viewpoints": ["viewpoint1", "viewpoint2", ...],
  "sentiment": {"overall": "positive|negative|neutral|mixed", "confidence": 0.0},
  "takeaways": ["takeaway1", "takeaway2", ...]
}

Rules:
- topics: 3-7 main themes or subjects discussed in the content
- viewpoints: core arguments, perspectives, or positions presented
- sentiment: overall emotional tone with confidence score (0.0-1.0)
  - "positive", "negative", "neutral", or "mixed"
- takeaways: actionable insights or key lessons from the content
- Handle both Chinese and English content naturally
- Return ONLY valid JSON, no markdown fences or extra text
"""

_RESTRUCTURE_PROMPT = """\
你是一个内容编辑。下面是一段口播视频的语音转录原文（由 Whisper 生成），没有标点、没有分段、没有结构。

请你把它整理成一篇**结构清晰、可读性强**的文章。要求：

1. **加标点符号**：句号、逗号、问号、感叹号，让句子完整
2. **分段落**：按话题/语义自然分段，每段 3-5 句话
3. **加小标题**：用 ## 标记每个话题段落的主题（简短有力，4-10 个字）
4. **保留原意**：不要改变说话者的观点和用词，只做格式整理
5. **去除口语冗余**：去掉反复的「OK」「那个」「就是说」等语气词，但保留口语化的表达风格
6. **标记金句**：特别有力的句子用 **加粗** 标记

输出纯 Markdown，不要加任何前言或解释。直接输出整理后的文章。
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def restructure_transcript(
    raw_text: str,
    *,
    config: ExtractorConfig | None = None,
) -> str | None:
    """Restructure raw Whisper transcript into readable, structured markdown.

    Returns the structured text, or None if LLM call fails or text is too long.
    """
    config = config if config is not None else ExtractorConfig()

    if not raw_text or not raw_text.strip():
        return None

    # Skip restructuring for very long transcripts rather than silently truncating
    max_chars = 25000  # ~6k tokens in Chinese
    if len(raw_text) > max_chars:
        logger.info(
            "Transcript too long for restructuring (%d chars > %d limit), keeping raw text",
            len(raw_text), max_chars,
        )
        return None

    try:
        structured = llm_chat(
            model=config.claude_model,
            messages=[{"role": "user", "content": raw_text}],
            system=_RESTRUCTURE_PROMPT,
            max_tokens=8192,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("Transcript restructuring failed: %s", exc)
        return None

    return structured if structured.strip() else None


def analyze_content(
    raw_text: str,
    *,
    content_id: str,
    content_type: str,
    config: ExtractorConfig | None = None,
) -> AnalysisResult:
    """Analyze extracted text via Claude and return structured AnalysisResult."""
    config = config if config is not None else ExtractorConfig()

    if not raw_text or not raw_text.strip():
        return AnalysisResult(
            content_id=content_id,
            content_type=content_type,
        )

    try:
        response_text = llm_chat(
            model=config.claude_model,
            messages=[{"role": "user", "content": raw_text}],
            system=_ANALYSIS_PROMPT,
            max_tokens=config.claude_max_tokens,
            temperature=config.claude_temperature,
        )
    except Exception as exc:
        raise AnalysisError(f"LLM API call failed: {exc}") from exc

    # Parse JSON response — LLM may wrap in markdown fences or add extra text
    parsed = _extract_json(response_text)
    if parsed is None:
        logger.warning(
            "Failed to parse analysis JSON for %s, returning fallback",
            content_id,
        )
        return AnalysisResult(
            content_id=content_id,
            content_type=content_type,
        )

    sentiment_data = parsed.get("sentiment")
    sentiment = None
    if isinstance(sentiment_data, dict):
        sentiment = SentimentResult(
            overall=sentiment_data.get("overall", "neutral"),
            confidence=float(sentiment_data.get("confidence", 0.0)),
        )

    return AnalysisResult(
        content_id=content_id,
        content_type=content_type,
        topics=tuple(parsed.get("topics", ())),
        viewpoints=tuple(parsed.get("viewpoints", ())),
        sentiment=sentiment,
        takeaways=tuple(parsed.get("takeaways", ())),
    )
