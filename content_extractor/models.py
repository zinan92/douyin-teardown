"""All Pydantic data models for content-extractor.

Input models:
    ContentItem -- mirror of content-downloader's output schema.

Output models:
    TranscriptSegment, Transcript -- speech-to-text results (D-01, D-02).
    SentimentResult, AnalysisResult -- LLM structured analysis (D-03).
    QualityMetadata -- per-item extraction quality metrics (QUAL-02).
    MediaDescription -- per-file vision/OCR result.
    ExtractionResult -- aggregate result from each adapter.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


class ContentItem(BaseModel):
    """Mirror of content-downloader's ContentItem. Reads content_item.json.

    Uses ``extra="ignore"`` so new upstream fields do not break validation.
    Uses ``tuple[str, ...]`` for media_files to guarantee immutability.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    platform: str
    content_id: str
    content_type: str  # "video" | "image" | "article" | "gallery"
    title: str
    description: str
    author_id: str
    author_name: str
    publish_time: str
    source_url: str
    media_files: tuple[str, ...] = ()
    cover_file: str | None = None
    metadata_file: str = "metadata.json"
    likes: int = 0
    comments: int = 0
    shares: int = 0
    collects: int = 0
    views: int = 0
    downloaded_at: str


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class TranscriptSegment(BaseModel):
    """A single segment of a transcript with timing info."""

    model_config = ConfigDict(frozen=True)

    text: str
    start: float  # seconds
    end: float  # seconds
    confidence: float  # 0.0-1.0
    is_suspicious: bool = False


class Transcript(BaseModel):
    """transcript.json schema (D-01, D-02)."""

    model_config = ConfigDict(frozen=True)

    content_id: str
    content_type: str
    language: str
    segments: tuple[TranscriptSegment, ...] = ()
    full_text: str
    schema_version: str = "1.0"


class SentimentResult(BaseModel):
    """Sentiment analysis output."""

    model_config = ConfigDict(frozen=True)

    overall: str  # "positive" | "negative" | "neutral" | "mixed"
    confidence: float


class AnalysisResult(BaseModel):
    """analysis.json schema -- fixed structure across all content types (D-03)."""

    model_config = ConfigDict(frozen=True)

    content_id: str
    content_type: str
    topics: tuple[str, ...] = ()
    viewpoints: tuple[str, ...] = ()
    sentiment: SentimentResult | None = None
    takeaways: tuple[str, ...] = ()
    schema_version: str = "1.0"


class QualityMetadata(BaseModel):
    """Per-item extraction quality metrics (QUAL-02)."""

    model_config = ConfigDict(frozen=True)

    confidence: float = 0.0
    language: str = "unknown"
    word_count: int = 0
    processing_time_seconds: float = 0.0
    hallucination_warnings: tuple[str, ...] = ()


class MediaDescription(BaseModel):
    """Per-media-file extraction result (images, article HTML, etc.)."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    description: str
    ocr_text: str = ""
    confidence: float = 0.0


class ExtractionResult(BaseModel):
    """Aggregate result returned by each adapter."""

    model_config = ConfigDict(frozen=True)

    content_id: str
    content_type: str
    raw_text: str
    transcript: Transcript | None = None
    media_descriptions: tuple[MediaDescription, ...] = ()
    quality: QualityMetadata = QualityMetadata()
    platform_metadata: dict[str, int | str] = {}
    schema_version: str = "1.0"
