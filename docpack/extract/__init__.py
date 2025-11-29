"""Document extraction module for PDF, DOCX, PPTX, and image files."""

from __future__ import annotations

from .audio import AUDIO_EXTENSIONS, can_extract_audio, extract_audio
from .base import ExtractedDocument, ExtractedImage
from .image import IMAGE_EXTENSIONS, can_extract_image, extract_image

# Extensions that can be extracted (documents)
DOCUMENT_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".pptx"})

# All extractable extensions (documents + images + audio)
EXTRACTABLE_EXTENSIONS: frozenset[str] = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS


def extract_document(data: bytes, extension: str, filename: str = "") -> ExtractedDocument:
    """
    Extract text and images from a document or image based on its extension.

    Args:
        data: Raw file bytes
        extension: File extension (e.g., '.pdf', '.docx', '.pptx', '.png')
        filename: Original filename (used for context in image extraction)

    Returns:
        ExtractedDocument with text content and embedded images

    Raises:
        ValueError: If the extension is not supported
    """
    ext = extension.lower()

    if ext == ".pdf":
        from .pdf import extract_pdf

        return extract_pdf(data)
    elif ext == ".docx":
        from .docx import extract_docx

        return extract_docx(data)
    elif ext == ".pptx":
        from .pptx import extract_pptx

        return extract_pptx(data)
    elif ext in IMAGE_EXTENSIONS:
        return extract_image(data, filename or "image")
    elif ext in AUDIO_EXTENSIONS:
        return extract_audio(data, filename or "audio")
    else:
        raise ValueError(f"Unsupported document format: {extension}")


def can_extract(extension: str) -> bool:
    """Check if the given extension is extractable."""
    return extension.lower() in EXTRACTABLE_EXTENSIONS


__all__ = [
    "ExtractedDocument",
    "ExtractedImage",
    "EXTRACTABLE_EXTENSIONS",
    "DOCUMENT_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "extract_document",
    "extract_image",
    "extract_audio",
    "can_extract",
    "can_extract_image",
    "can_extract_audio",
]
