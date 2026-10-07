"""Image preprocessing and Claude vision API call.

Provides reusable functions for:
- Image preprocessing (resize, format conversion, base64 encoding)
- Claude vision API calls for combined OCR + visual description

Used by ImageExtractor (Phase 4) and GalleryExtractor (Phase 7).
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import orjson
from PIL import Image
from pydantic import BaseModel, ConfigDict

from content_extractor.config import ExtractorConfig
from content_extractor.llm import create_claude_client

logger = logging.getLogger(__name__)

# Max dimension for Claude vision API (recommended by Anthropic docs)
_MAX_DIMENSION = 1568

# Formats Claude vision supports natively
_SUPPORTED_FORMATS: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}

# Vision prompt requesting structured JSON with Chinese text support
_VISION_PROMPT = """\
Analyze this image and return a JSON object with exactly these fields:

{
  "ocr_text": "ALL visible text in the image, transcribed in original language \
(including Chinese characters). Note text position context (top, bottom, overlay, \
caption). Preserve line breaks.",
  "visual_description": "Detailed description of the image content, layout, colors, \
and visual elements.",
  "confidence": 0.0  // 0.0 to 1.0, how confident you are in the OCR accuracy
}

Important:
- Transcribe ALL visible text including Chinese/CJK characters in their original \
language, do NOT translate
- Note where text appears (overlay, top, bottom, watermark, etc.)
- Return ONLY valid JSON, no markdown fences or extra text
"""


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class ImageExtractionError(Exception):
    """Raised when image extraction fails."""


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class VisionResponse(BaseModel):
    """Parsed response from Claude vision API."""

    model_config = ConfigDict(frozen=True)

    ocr_text: str
    visual_description: str
    confidence: float


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def preprocess_image(image_path: Path) -> tuple[str, str]:
    """Load, resize if needed, and base64-encode an image.

    Parameters
    ----------
    image_path:
        Path to the image file on disk.

    Returns
    -------
    tuple[str, str]
        (base64_encoded_data, media_type) e.g. ("abc...", "image/jpeg")

    Raises
    ------
    FileNotFoundError
        If image_path does not exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    # Capture format before any transforms (resize drops .format)
    original_format = img.format or "UNKNOWN"

    # Resize if the longest side exceeds the max dimension
    max_side = max(img.width, img.height)
    if max_side > _MAX_DIMENSION:
        scale = _MAX_DIMENSION / max_side
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)

    # Determine output format
    if original_format in _SUPPORTED_FORMATS:
        out_format = original_format
        media_type = _SUPPORTED_FORMATS[original_format]
    else:
        out_format = "JPEG"
        media_type = "image/jpeg"

    # JPEG does not support alpha -- convert RGBA/P to RGB
    if out_format == "JPEG" and img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format=out_format)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")

    return b64, media_type


def describe_image(
    base64_data: str,
    media_type: str,
    config: ExtractorConfig,
) -> VisionResponse:
    """Send an image to Claude vision and parse the structured response.

    Parameters
    ----------
    base64_data:
        Base64-encoded image bytes.
    media_type:
        MIME type, e.g. "image/jpeg".
    config:
        ExtractorConfig with claude_model, claude_max_tokens, claude_temperature.

    Returns
    -------
    VisionResponse
        Parsed OCR text, visual description, and confidence.

    Raises
    ------
    ImageExtractionError
        If the API call fails.
    """
    client = create_claude_client(config)

    try:
        response = client.messages.create(
            model=config.claude_model,
            max_tokens=config.claude_max_tokens,
            temperature=config.claude_temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": _VISION_PROMPT,
                        },
                    ],
                }
            ],
        )
    except Exception as exc:
        raise ImageExtractionError(
            f"Claude vision API call failed: {exc}"
        ) from exc

    # Extract text from response
    raw_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw_text = block.text
            break

    # Parse JSON response
    try:
        parsed = orjson.loads(raw_text)
        return VisionResponse(
            ocr_text=parsed.get("ocr_text", ""),
            visual_description=parsed.get("visual_description", ""),
            confidence=float(parsed.get("confidence", 0.0)),
        )
    except (orjson.JSONDecodeError, ValueError, TypeError):
        logger.warning("Failed to parse vision JSON, using raw text as description")
        return VisionResponse(
            ocr_text="",
            visual_description=raw_text,
            confidence=0.0,
        )
