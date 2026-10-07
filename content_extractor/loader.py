"""Load and validate ContentItem from a content directory.

Reads ``content_item.json`` using orjson for speed, validates via Pydantic.
Raises clear custom exceptions on missing or invalid files.
"""

from __future__ import annotations

from pathlib import Path

import orjson
from pydantic import ValidationError

from content_extractor.models import ContentItem


class ContentItemNotFoundError(Exception):
    """Raised when content_item.json does not exist in the given directory."""


class ContentItemInvalidError(Exception):
    """Raised when content_item.json cannot be parsed or fails validation."""


def load_content_item(content_dir: Path) -> ContentItem:
    """Load and validate a ContentItem from *content_dir*.

    Parameters
    ----------
    content_dir:
        Path to a content directory containing ``content_item.json``.

    Returns
    -------
    ContentItem
        A frozen, validated ContentItem instance.

    Raises
    ------
    ContentItemNotFoundError
        If ``content_item.json`` does not exist in *content_dir*.
    ContentItemInvalidError
        If the file contains invalid JSON or fails Pydantic validation.
    """
    item_path = content_dir / "content_item.json"

    if not item_path.exists():
        raise ContentItemNotFoundError(
            f"No content_item.json in {content_dir}"
        )

    raw = item_path.read_bytes()

    try:
        data = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise ContentItemInvalidError(
            f"Invalid JSON in {item_path}: {exc}"
        ) from exc

    try:
        return ContentItem.model_validate(data)
    except ValidationError as exc:
        raise ContentItemInvalidError(
            f"Validation failed for {item_path}: {exc}"
        ) from exc
