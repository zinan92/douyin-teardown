"""Standardized data models for content-downloader."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ContentItem(BaseModel):
    """Standardized representation of a single downloaded content item."""

    model_config = {"frozen": True}

    platform: str
    """Platform name: 'douyin' | 'xhs' | 'wechat_oa' | 'x' | 'fixture'"""

    content_id: str
    """Platform-native content identifier."""

    content_type: str
    """Content type: 'video' | 'image' | 'article' | 'gallery'"""

    title: str
    description: str

    author_id: str
    author_name: str

    publish_time: str
    """ISO 8601 timestamp string."""

    source_url: str

    media_files: list[str] = Field(default_factory=list)
    """Relative paths inside the content directory, e.g. ['media/video.mp4']."""

    cover_file: str | None = None
    """Relative path to cover image, e.g. 'media/cover.jpg'."""

    metadata_file: str = "metadata.json"
    """Relative path to raw platform metadata file."""

    likes: int = 0
    comments: int = 0
    shares: int = 0
    collects: int = 0
    views: int = 0

    downloaded_at: str
    """ISO 8601 timestamp when this item was downloaded."""


class DownloadError(BaseModel):
    """Record of a failed download attempt."""

    model_config = {"frozen": True}

    content_id: str
    source_url: str
    error_type: str
    """'auth' | 'rate_limit' | 'not_found' | 'network' | 'unsupported'"""
    message: str
    retryable: bool


class DownloadResult(BaseModel):
    """Aggregate result of a download operation (single or batch)."""

    model_config = {"frozen": True}

    items: list[ContentItem] = Field(default_factory=list)
    errors: list[DownloadError] = Field(default_factory=list)
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
