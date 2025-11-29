"""PDF document extraction using PyMuPDF."""

from __future__ import annotations

import io

from .base import ExtractedDocument, ExtractedImage


def extract_pdf(data: bytes) -> ExtractedDocument:
    """
    Extract text and images from PDF bytes.

    Args:
        data: Raw PDF file bytes

    Returns:
        ExtractedDocument with text content and embedded images
    """
    import fitz  # pymupdf

    doc = fitz.open(stream=data, filetype="pdf")
    text_parts: list[str] = []
    images: list[ExtractedImage] = []
    current_heading: str | None = None

    for page_num, page in enumerate(doc, start=1):
        # Extract text
        page_text = page.get_text()
        if page_text.strip():
            text_parts.append(page_text)

            # Simple heading detection for context
            lines = page_text.split("\n")
            for line in lines:
                stripped = line.strip()
                # Heuristic: short uppercase lines or lines ending with colon might be headings
                if stripped and len(stripped) < 100:
                    if stripped.isupper() or stripped.endswith(":"):
                        current_heading = stripped

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if base_image:
                    img_data = base_image["image"]
                    img_ext = base_image["ext"]

                    # Get image dimensions
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # If dimensions not in metadata, try to get from pixmap
                    if width == 0 or height == 0:
                        try:
                            from PIL import Image

                            pil_img = Image.open(io.BytesIO(img_data))
                            width, height = pil_img.size
                        except Exception:
                            pass

                    images.append(
                        ExtractedImage(
                            data=img_data,
                            format=img_ext,
                            width=width,
                            height=height,
                            page_number=page_num,
                            image_index=img_index,
                            context=current_heading,
                        )
                    )
            except Exception:
                # Skip problematic images
                continue

    # Get document metadata and page count before closing
    metadata: dict[str, str] = {}
    page_count = len(doc)
    try:
        doc_metadata = doc.metadata
        if doc_metadata:
            for key in ["title", "author", "subject", "creator"]:
                if doc_metadata.get(key):
                    metadata[key] = doc_metadata[key]
    except Exception:
        pass

    doc.close()

    return ExtractedDocument(
        text="\n\n".join(text_parts),
        images=images,
        page_count=page_count,
        metadata=metadata,
    )
