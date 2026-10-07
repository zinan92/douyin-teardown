"""Image content extractor using Claude vision API.

Sends images to Claude for combined OCR + visual description in a single
API call. Supports Chinese text overlay recognition (Xiaohongshu images).

Uses the reusable vision module for preprocessing and API calls.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from content_extractor.config import ExtractorConfig
from content_extractor.loader import load_content_item
from content_extractor.models import (
    ExtractionResult,
    MediaDescription,
    QualityMetadata,
)
from content_extractor.text_utils import _compute_word_count, _detect_language
from content_extractor.vision import (
    ImageExtractionError,
    describe_image,
    preprocess_image,
)

logger = logging.getLogger(__name__)

# Image file extensions supported by Claude vision
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})


class ImageExtractor:
    """Extract text from image content via Claude vision API."""

    content_type: str = "image"

    def extract(
        self, content_dir: Path, config: ExtractorConfig
    ) -> ExtractionResult:
        """Extract OCR text and visual descriptions from images.

        Parameters
        ----------
        content_dir:
            Path to content directory with content_item.json and media/.
        config:
            Extraction configuration (Claude model, max_tokens, etc.).

        Returns
        -------
        ExtractionResult
            Aggregated result with media_descriptions, raw_text, and quality.

        Raises
        ------
        ImageExtractionError
            If no image files are found in the content directory.
        """
        start_time = time.monotonic()

        item = load_content_item(content_dir)

        # Find image files from content_item.json media_files
        image_files = [
            f
            for f in item.media_files
            if Path(f).suffix.lower() in _IMAGE_EXTENSIONS
        ]

        # Fallback: glob media/ directory for image files
        if not image_files:
            media_dir = content_dir / "media"
            if media_dir.is_dir():
                image_files = [
                    f"media/{p.name}"
                    for p in sorted(media_dir.iterdir())
                    if p.suffix.lower() in _IMAGE_EXTENSIONS
                ]

        if not image_files:
            raise ImageExtractionError(
                f"No image files found in {content_dir}"
            )

        # Process each image
        descriptions: list[MediaDescription] = []
        for image_file in image_files:
            image_path = content_dir / image_file
            try:
                base64_data, media_type = preprocess_image(image_path)
                response = describe_image(base64_data, media_type, config)
                descriptions.append(
                    MediaDescription(
                        file_path=image_file,
                        description=response.visual_description,
                        ocr_text=response.ocr_text,
                        confidence=response.confidence,
                    )
                )
            except Exception:
                logger.exception("Failed to process image: %s", image_file)
                descriptions.append(
                    MediaDescription(
                        file_path=image_file,
                        description="",
                        ocr_text="",
                        confidence=0.0,
                    )
                )

        # Combine OCR text from all images
        ocr_texts = [d.ocr_text for d in descriptions if d.ocr_text]
        combined_text = "\n".join(ocr_texts).strip()

        # Compute quality metrics
        confidences = [d.confidence for d in descriptions]
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        language = _detect_language(combined_text)
        word_count = _compute_word_count(combined_text)
        elapsed = time.monotonic() - start_time

        return ExtractionResult(
            content_id=item.content_id,
            content_type="image",
            raw_text=combined_text,
            media_descriptions=tuple(descriptions),
            quality=QualityMetadata(
                confidence=avg_confidence,
                language=language,
                word_count=word_count,
                processing_time_seconds=elapsed,
            ),
        )
