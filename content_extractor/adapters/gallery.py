"""Gallery content extractor using batched Claude vision + narrative synthesis.

Processes multi-image gallery posts by:
1. Discovering image files from ContentItem
2. Sending each image to Claude vision in batches (max 5 per batch)
3. Synthesizing per-image descriptions into a coherent gallery narrative via LLM

Per-image failures produce fallback MediaDescription(confidence=0.0), not
gallery-level failure. Only an empty gallery (no images found) raises.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from content_extractor.config import ExtractorConfig
from content_extractor.llm import create_claude_client
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

# Max images per batch to respect rate limits
_BATCH_SIZE = 5

# Pause between batches (seconds)
_BATCH_DELAY = 1.0

# System prompt for narrative synthesis
_NARRATIVE_SYSTEM = (
    "You are a content analyst. Synthesize individual image descriptions "
    "from a gallery post into a coherent narrative."
)

# User prompt template for narrative synthesis
_NARRATIVE_USER_TEMPLATE = (
    "{numbered_descriptions}\n\n"
    "Write a coherent narrative (2-4 paragraphs) describing this gallery as "
    "a whole. Preserve any Chinese text references. Focus on the story arc "
    "and key information across all images."
)


class GalleryExtractor:
    """Extract text from gallery (multi-image) content."""

    content_type: str = "gallery"

    def extract(
        self, content_dir: Path, config: ExtractorConfig
    ) -> ExtractionResult:
        """Extract descriptions from all gallery images and synthesize narrative.

        Parameters
        ----------
        content_dir:
            Path to content directory with content_item.json and media/.
        config:
            Extraction configuration (Claude model, max_tokens, etc.).

        Returns
        -------
        ExtractionResult
            Aggregated result with narrative raw_text, per-image
            media_descriptions, and quality metadata.

        Raises
        ------
        ImageExtractionError
            If no image files are found in the content directory.
        """
        start_time = time.monotonic()

        item = load_content_item(content_dir)

        # --- Image discovery (same pattern as ImageExtractor) ---
        image_files = [
            f
            for f in item.media_files
            if Path(f).suffix.lower() in _IMAGE_EXTENSIONS
        ]

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

        # --- Batched extraction ---
        descriptions = _extract_images_batched(
            image_files, content_dir, config
        )

        # --- Narrative synthesis ---
        narrative = _synthesize_narrative(descriptions, config)

        # --- Quality metadata ---
        confidences = [d.confidence for d in descriptions]
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        language = _detect_language(narrative)
        word_count = _compute_word_count(narrative)
        elapsed = time.monotonic() - start_time

        return ExtractionResult(
            content_id=item.content_id,
            content_type="gallery",
            raw_text=narrative,
            media_descriptions=tuple(descriptions),
            quality=QualityMetadata(
                confidence=avg_confidence,
                language=language,
                word_count=word_count,
                processing_time_seconds=elapsed,
            ),
        )


def _extract_images_batched(
    image_files: list[str],
    content_dir: Path,
    config: ExtractorConfig,
) -> list[MediaDescription]:
    """Process images in batches, with per-image error isolation.

    Returns a list of MediaDescription (one per image). Failed images
    get a fallback entry with confidence=0.0.
    """
    descriptions: list[MediaDescription] = []

    for batch_idx in range(0, len(image_files), _BATCH_SIZE):
        if batch_idx > 0:
            time.sleep(_BATCH_DELAY)

        batch = image_files[batch_idx : batch_idx + _BATCH_SIZE]

        for image_file in batch:
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
                logger.exception(
                    "Failed to process gallery image: %s", image_file
                )
                descriptions.append(
                    MediaDescription(
                        file_path=image_file,
                        description="",
                        ocr_text="",
                        confidence=0.0,
                    )
                )

    return descriptions


def _synthesize_narrative(
    descriptions: list[MediaDescription],
    config: ExtractorConfig,
) -> str:
    """Synthesize per-image descriptions into a coherent gallery narrative.

    If no successful descriptions exist (all empty), returns empty string
    without making an LLM call.
    """
    # Filter to descriptions that have actual content
    successful = [
        d for d in descriptions
        if d.description or d.ocr_text
    ]

    if not successful:
        return ""

    # Build numbered description list
    lines: list[str] = []
    for idx, desc in enumerate(descriptions, start=1):
        parts: list[str] = []
        if desc.description:
            parts.append(desc.description)
        if desc.ocr_text:
            parts.append(f"Text: {desc.ocr_text}")
        entry = ". ".join(parts) if parts else "(image processing failed)"
        lines.append(f"Image {idx}: {entry}")

    numbered_descriptions = "\n".join(lines)
    user_prompt = _NARRATIVE_USER_TEMPLATE.format(
        numbered_descriptions=numbered_descriptions
    )

    try:
        client = create_claude_client(config)
        response = client.messages.create(
            model=config.claude_model,
            max_tokens=config.claude_max_tokens,
            temperature=config.claude_temperature,
            system=_NARRATIVE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )

        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text

        return ""
    except Exception:
        logger.exception("Narrative synthesis failed")
        return ""
