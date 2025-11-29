"""Document extraction module for PDF, DOCX, and PPTX files."""

from __future__ import annotations

from .base import ExtractedDocument, ExtractedImage

# Extensions that can be extracted
EXTRACTABLE_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".pptx"})


def extract_document(data: bytes, extension: str) -> ExtractedDocument:
    """
    Extract text and images from a document based on its extension.

    Args:
        data: Raw document bytes
        extension: File extension (e.g., '.pdf', '.docx', '.pptx')

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
    else:
        raise ValueError(f"Unsupported document format: {extension}")


def can_extract(extension: str) -> bool:
    """Check if the given extension is extractable."""
    return extension.lower() in EXTRACTABLE_EXTENSIONS


__all__ = [
    "ExtractedDocument",
    "ExtractedImage",
    "EXTRACTABLE_EXTENSIONS",
    "extract_document",
    "can_extract",
]
