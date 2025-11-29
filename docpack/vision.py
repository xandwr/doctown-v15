"""
Vision model processing for images extracted from documents.

Uses Ollama with qwen3-vl:2b (or similar vision models) to generate
natural language descriptions of images, diagrams, and charts.
These descriptions become searchable chunks alongside text content.
"""

from __future__ import annotations

import base64
import io
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from docpack.runtime import RuntimeConfig, get_global_config
from docpack.storage import (
    get_image_data,
    get_unanalyzed_images,
    insert_chunk,
    mark_image_analyzed,
    set_metadata,
)


DEFAULT_VISION_MODEL = "qwen3-vl:2b"


# =============================================================================
# Image Filtering
# =============================================================================


def should_analyze_image(
    width: int,
    height: int,
    size_bytes: int,
    format: str,
) -> tuple[bool, str]:
    """
    Determine if an image should be analyzed by the vision model.

    Filters out small icons, spacers, and low-information images.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        size_bytes: Size of image in bytes
        format: Image format ('png', 'jpeg', etc.)

    Returns:
        Tuple of (should_analyze, reason)
    """
    # Minimum size threshold - skip tiny icons and spacers
    if width < 100 or height < 100:
        return False, "too_small_dimensions"

    if size_bytes < 5_000:  # 5KB minimum (PDFs often have well-compressed images)
        return False, "too_small_bytes"

    # Maximum size threshold - vision models have limits
    if width > 4096 or height > 4096:
        return False, "too_large"

    # Extreme aspect ratios are usually decorative elements
    aspect = width / height
    if aspect < 0.1 or aspect > 10:
        return False, "extreme_aspect_ratio"

    # Check entropy if we have PIL available
    # (This would require loading the image, so we skip for now
    # and rely on size/dimension heuristics)

    return True, "pass"


def calculate_entropy(image_data: bytes) -> float:
    """
    Calculate image entropy to detect low-information images.

    Higher entropy = more information content.
    Solid colors and gradients have low entropy.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_data))
        # Convert to grayscale for entropy calculation
        gray = img.convert("L")

        # Calculate histogram
        histogram = gray.histogram()
        total_pixels = gray.width * gray.height

        # Calculate entropy from histogram
        entropy = 0.0
        for count in histogram:
            if count > 0:
                p = count / total_pixels
                entropy -= p * math.log2(p)

        return entropy
    except Exception:
        return 8.0  # Return high entropy on error to not skip


# =============================================================================
# Vision Model Integration
# =============================================================================


def describe_image(
    image_data: bytes,
    context: str | None = None,
    model: str = DEFAULT_VISION_MODEL,
) -> str:
    """
    Generate a description of an image using Ollama vision model.

    Args:
        image_data: Raw image bytes
        context: Optional context (nearby heading/text from document)
        model: Vision model to use

    Returns:
        Natural language description of the image
    """
    from ollama import chat

    b64_image = base64.b64encode(image_data).decode("utf-8")

    prompt = (
        "Describe this image in detail. Focus on any data, diagrams, charts, "
        "tables, text, or key visual information. Be specific and factual."
    )
    if context:
        prompt += f"\n\nContext from document: {context[:500]}"

    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }
        ],
    )

    if response.message.content is None:
        raise ValueError("No content returned from vision model")

    return response.message.content


# =============================================================================
# Batch Processing
# =============================================================================


def vision_all(
    conn: sqlite3.Connection,
    model: str = DEFAULT_VISION_MODEL,
    config: RuntimeConfig | None = None,
    *,
    verbose: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """
    Analyze all unprocessed images with vision model.

    Creates chunks from image descriptions that are then embedded
    and searchable alongside text chunks.

    Args:
        conn: Database connection
        model: Vision model to use
        config: Runtime configuration
        verbose: Print progress
        progress_callback: Progress callback(current, total)

    Returns:
        Number of images analyzed
    """
    cfg = config or get_global_config()

    # Get all unanalyzed images
    images = get_unanalyzed_images(conn)
    total = len(images)

    if not images:
        if verbose:
            print("No images to analyze")
        return 0

    mode = "CPU (parallel)" if cfg.force_cpu else "GPU"
    if verbose:
        print(f"Analyzing {total} images with {model} [{mode}]...")

    if progress_callback:
        progress_callback(0, total)

    # CPU mode: parallel processing
    if cfg.force_cpu:
        return _vision_parallel(conn, images, model, cfg, verbose, progress_callback)

    # GPU mode: sequential (vision models don't batch well)
    return _vision_sequential(conn, images, model, verbose, progress_callback)


def _vision_sequential(
    conn: sqlite3.Connection,
    images: list[dict],
    model: str,
    verbose: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> int:
    """Process images sequentially (GPU mode)."""
    total = len(images)
    analyzed = 0
    skipped = 0
    errors = 0

    for i, img in enumerate(images):
        image_id = img["id"]
        file_id = img["file_id"]
        file_path = img["file_path"]
        page_number = img.get("page_number")
        context = img.get("context")

        # Check if image should be analyzed
        should_analyze, reason = should_analyze_image(
            img["width"],
            img["height"],
            img["size_bytes"],
            img["format"],
        )

        if not should_analyze:
            skipped += 1
            mark_image_analyzed(conn, image_id)
            if verbose:
                print(f"  [{i+1}/{total}] SKIP: {reason}")
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        try:
            # Get image data
            image_data = get_image_data(conn, image_id)
            if not image_data:
                errors += 1
                continue

            # Optional entropy check
            entropy = calculate_entropy(image_data)
            if entropy < 4.0:
                skipped += 1
                mark_image_analyzed(conn, image_id)
                if verbose:
                    print(f"  [{i+1}/{total}] SKIP: low_entropy ({entropy:.1f})")
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            # Generate description
            description = describe_image(image_data, context=context, model=model)

            # Build context string for the chunk
            location = f"page {page_number}" if page_number else "document"
            chunk_text = f"[Image from {file_path}, {location}]\n\n{description}"

            # Get the next chunk index for this file
            cursor = conn.execute(
                "SELECT COALESCE(MAX(chunk_index), -1) + 1 FROM chunks WHERE file_id = ?",
                (file_id,),
            )
            next_index = cursor.fetchone()[0]

            # Insert as a chunk with media_type='image'
            conn.execute(
                """
                INSERT INTO chunks (file_id, chunk_index, text, token_count, media_type)
                VALUES (?, ?, ?, ?, 'image')
                """,
                (file_id, next_index, chunk_text, len(chunk_text) // 4),
            )

            # Mark image as analyzed
            mark_image_analyzed(conn, image_id)
            conn.commit()

            analyzed += 1
            if verbose:
                preview = description[:60] + "..." if len(description) > 60 else description
                print(f"  [{i+1}/{total}] {file_path}: {preview}")

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  [{i+1}/{total}] ERROR: {e}")

        if progress_callback:
            progress_callback(i + 1, total)

    # Update metadata
    set_metadata(conn, "vision_count", str(analyzed))
    set_metadata(conn, "vision_model", model)

    if verbose:
        print(f"\nDone: {analyzed}/{total} analyzed, {skipped} skipped, {errors} errors")

    return analyzed


def _vision_parallel(
    conn: sqlite3.Connection,
    images: list[dict],
    model: str,
    config: RuntimeConfig,
    verbose: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> int:
    """Process images in parallel (CPU mode)."""
    total = len(images)
    analyzed = 0
    skipped = 0
    errors = 0

    def process_single(img: dict) -> tuple[dict, str | None, str | None]:
        """Process single image, return (img, description, error)."""
        # Check if image should be analyzed
        should_analyze, reason = should_analyze_image(
            img["width"],
            img["height"],
            img["size_bytes"],
            img["format"],
        )

        if not should_analyze:
            return img, None, f"skip:{reason}"

        try:
            # Get image data (need separate connection for thread safety)
            image_data = get_image_data(conn, img["id"])
            if not image_data:
                return img, None, "no_data"

            # Entropy check
            entropy = calculate_entropy(image_data)
            if entropy < 4.0:
                return img, None, f"skip:low_entropy"

            # Generate description
            description = describe_image(
                image_data,
                context=img.get("context"),
                model=model,
            )
            return img, description, None

        except Exception as e:
            return img, None, str(e)

    # Process in parallel
    completed = 0
    results: list[tuple[dict, str | None, str | None]] = []

    with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures = {executor.submit(process_single, img): img for img in images}

        for future in as_completed(futures):
            img, description, error = future.result()
            results.append((img, description, error))
            completed += 1

            if progress_callback:
                progress_callback(completed, total)

    # Process results and update database (single-threaded for SQLite)
    for img, description, error in results:
        image_id = img["id"]
        file_id = img["file_id"]
        file_path = img["file_path"]
        page_number = img.get("page_number")

        if error and error.startswith("skip:"):
            skipped += 1
            mark_image_analyzed(conn, image_id)
            continue

        if error:
            errors += 1
            if verbose:
                print(f"  ERROR {file_path}: {error}")
            continue

        if description:
            # Build context string for the chunk
            location = f"page {page_number}" if page_number else "document"
            chunk_text = f"[Image from {file_path}, {location}]\n\n{description}"

            # Get the next chunk index for this file
            cursor = conn.execute(
                "SELECT COALESCE(MAX(chunk_index), -1) + 1 FROM chunks WHERE file_id = ?",
                (file_id,),
            )
            next_index = cursor.fetchone()[0]

            # Insert as a chunk with media_type='image'
            conn.execute(
                """
                INSERT INTO chunks (file_id, chunk_index, text, token_count, media_type)
                VALUES (?, ?, ?, ?, 'image')
                """,
                (file_id, next_index, chunk_text, len(chunk_text) // 4),
            )

            mark_image_analyzed(conn, image_id)
            analyzed += 1

    conn.commit()

    # Update metadata
    set_metadata(conn, "vision_count", str(analyzed))
    set_metadata(conn, "vision_model", model)

    if verbose:
        print(f"\nDone: {analyzed}/{total} analyzed, {skipped} skipped, {errors} errors")

    return analyzed
