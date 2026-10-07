"""Shared fixtures for all tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


@pytest.fixture()
def sample_content_item_dict() -> dict[str, Any]:
    """Return a valid douyin video dict with all 18 fields."""
    return {
        "platform": "douyin",
        "content_id": "7616648694510652672",
        "content_type": "video",
        "title": "Agent架构选型指南",
        "description": "Agent架构选型，从单Agent到Multi-Agent",
        "author_id": "59130943719",
        "author_name": "慢学AI",
        "publish_time": "2026-03-13T14:00:00+00:00",
        "source_url": "https://www.iesdouyin.com/share/video/7616648694510652672",
        "media_files": ["media/video.mp4"],
        "cover_file": "media/cover.jpg",
        "metadata_file": "metadata.json",
        "likes": 10943,
        "comments": 217,
        "shares": 1970,
        "collects": 9786,
        "views": 0,
        "downloaded_at": "2026-03-30T02:06:53.588772Z",
    }


@pytest.fixture()
def tmp_content_dir(
    tmp_path: Path, sample_content_item_dict: dict[str, Any]
) -> Path:
    """Write content_item.json to tmp_path and return tmp_path."""
    item_path = tmp_path / "content_item.json"
    item_path.write_text(json.dumps(sample_content_item_dict), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def tmp_image_dir(tmp_path: Path) -> Path:
    """Create a temp dir with a small 100x100 red JPEG test image."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(media_dir / "test.jpg", "JPEG")
    return tmp_path


@pytest.fixture()
def mock_vision_response_json() -> str:
    """Return a valid JSON string mimicking Claude vision response."""
    return json.dumps({
        "ocr_text": "小红书爆款笔记分享",
        "visual_description": "A red background image with Chinese text overlay",
        "confidence": 0.92,
    })
