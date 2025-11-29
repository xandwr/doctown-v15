"""Base types for document extraction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractedImage:
    """An image extracted from a document."""

    data: bytes
    format: str  # 'png', 'jpeg', etc.
    width: int
    height: int
    page_number: int | None = None
    image_index: int = 0
    context: str | None = None  # Nearby text/heading


@dataclass
class ExtractedDocument:
    """Result of document extraction."""

    text: str
    images: list[ExtractedImage] = field(default_factory=list)
    page_count: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)
