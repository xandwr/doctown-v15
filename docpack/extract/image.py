"""Direct image file extraction.

Handles standalone image files (PNG, JPEG, GIF, WebP, etc.) by wrapping
them as ExtractedDocument with the image as the sole content.
"""

from __future__ import annotations

import io

from .base import ExtractedDocument, ExtractedImage


# Image extensions we support
IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
})


def extract_image(data: bytes, filename: str = "image") -> ExtractedDocument:
    """
    Extract an image file as an ExtractedDocument.

    The image becomes searchable through vision model analysis.

    Args:
        data: Raw image bytes
        filename: Original filename for context

    Returns:
        ExtractedDocument with the image as content
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        width, height = img.size
        fmt = img.format.lower() if img.format else "unknown"

        # Determine format from PIL
        if fmt in ("jpeg", "jpg"):
            fmt = "jpeg"
        elif fmt == "png":
            fmt = "png"
        elif fmt == "gif":
            fmt = "gif"
        elif fmt == "webp":
            fmt = "webp"
        elif fmt in ("tiff", "tif"):
            fmt = "tiff"
        elif fmt == "bmp":
            fmt = "bmp"

    except Exception:
        # Fallback if PIL fails - use basic detection
        width, height = 0, 0
        fmt = "unknown"

        # Try to detect format from magic bytes
        if data.startswith(b"\x89PNG"):
            fmt = "png"
        elif data.startswith(b"\xff\xd8\xff"):
            fmt = "jpeg"
        elif data.startswith(b"GIF8"):
            fmt = "gif"
        elif data.startswith(b"RIFF") and b"WEBP" in data[:12]:
            fmt = "webp"
        elif data.startswith(b"BM"):
            fmt = "bmp"

    extracted_image = ExtractedImage(
        data=data,
        format=fmt,
        width=width,
        height=height,
        page_number=None,
        image_index=0,
        context=f"Standalone image: {filename}",
    )

    # The "text" is just a placeholder - the real content comes from vision analysis
    return ExtractedDocument(
        text=f"[Image: {filename}]",
        images=[extracted_image],
        page_count=None,
        metadata={"type": "image", "format": fmt},
    )


def can_extract_image(extension: str) -> bool:
    """Check if the extension is a supported image format."""
    return extension.lower() in IMAGE_EXTENSIONS
