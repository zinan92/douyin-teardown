"""Tests for ContentItem loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from content_extractor.loader import (
    ContentItemInvalidError,
    ContentItemNotFoundError,
    load_content_item,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLoadContentItem:
    """Tests for load_content_item function."""

    def test_load_valid_content_item(
        self, tmp_content_dir: Path, sample_content_item_dict: dict[str, Any]
    ) -> None:
        item = load_content_item(tmp_content_dir)

        assert item.platform == "douyin"
        assert item.content_id == "7616648694510652672"
        assert item.content_type == "video"
        assert item.title == "Agent架构选型指南"
        assert item.author_name == "慢学AI"
        assert item.likes == 10943
        assert item.downloaded_at == "2026-03-30T02:06:53.588772Z"
        assert isinstance(item.media_files, tuple)

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ContentItemNotFoundError) as exc_info:
            load_content_item(tmp_path)

        assert str(tmp_path) in str(exc_info.value)

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        item_path = tmp_path / "content_item.json"
        item_path.write_text("not valid json {{{", encoding="utf-8")

        with pytest.raises(ContentItemInvalidError):
            load_content_item(tmp_path)

    def test_load_extra_fields_ignored(
        self, tmp_path: Path, sample_content_item_dict: dict[str, Any]
    ) -> None:
        data = {**sample_content_item_dict, "new_field": "extra value"}
        item_path = tmp_path / "content_item.json"
        item_path.write_text(json.dumps(data), encoding="utf-8")

        item = load_content_item(tmp_path)

        assert item.platform == "douyin"
        assert not hasattr(item, "new_field")

    def test_load_missing_required_field(
        self, tmp_path: Path, sample_content_item_dict: dict[str, Any]
    ) -> None:
        data = {**sample_content_item_dict}
        del data["platform"]
        item_path = tmp_path / "content_item.json"
        item_path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ContentItemInvalidError):
            load_content_item(tmp_path)

    @pytest.mark.parametrize(
        "fixture_file",
        [
            "douyin_video.json",
            "xhs_video.json",
            "wechat_article.json",
            "x_video.json",
        ],
    )
    def test_load_all_fixtures(self, tmp_path: Path, fixture_file: str) -> None:
        fixture_path = FIXTURES_DIR / fixture_file
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        item_path = tmp_path / "content_item.json"
        item_path.write_text(json.dumps(data), encoding="utf-8")

        item = load_content_item(tmp_path)

        assert item.content_id
        assert item.platform
        assert item.content_type in ("video", "image", "article", "gallery")
