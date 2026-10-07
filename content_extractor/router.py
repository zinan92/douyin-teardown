"""Content type routing registry.

Maps content_type strings to adapter classes. Auto-registers all built-in
adapters at module load time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from content_extractor.adapters.article import ArticleExtractor
from content_extractor.adapters.base import Extractor
from content_extractor.adapters.gallery import GalleryExtractor
from content_extractor.adapters.image import ImageExtractor
from content_extractor.adapters.video import VideoExtractor

if TYPE_CHECKING:
    pass


class UnsupportedContentTypeError(Exception):
    """Raised when no adapter is registered for a given content_type."""


_REGISTRY: dict[str, type] = {}


def register(content_type: str, extractor_cls: type) -> None:
    """Register an extractor class for a content type."""
    _REGISTRY[content_type] = extractor_cls


def get_extractor(content_type: str) -> Extractor:
    """Look up and instantiate the extractor for a content type.

    Raises UnsupportedContentTypeError if no adapter is registered.
    """
    cls = _REGISTRY.get(content_type)
    if cls is None:
        raise UnsupportedContentTypeError(
            f"No extractor registered for content_type={content_type!r}. "
            f"Supported: {sorted(_REGISTRY.keys())}"
        )
    return cls()


# Auto-register built-in adapters
register("video", VideoExtractor)
register("image", ImageExtractor)
register("article", ArticleExtractor)
register("gallery", GalleryExtractor)
