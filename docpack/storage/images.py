"""Image storage operations."""

from __future__ import annotations

import sqlite3
from typing import Any

from .schema import IntegrityError, StorageError


def insert_image(
    conn: sqlite3.Connection,
    file_id: int,
    image_index: int,
    format: str,
    width: int,
    height: int,
    image_data: bytes,
    *,
    page_number: int | None = None,
    context: str | None = None,
) -> int:
    """
    Insert an image record linked to a file.

    Args:
        conn: Database connection
        file_id: ID of the parent file
        image_index: Position of image within the page/file
        format: Image format ('png', 'jpeg', etc.)
        width: Image width in pixels
        height: Image height in pixels
        image_data: Raw image bytes
        page_number: Page/slide number (1-indexed, None for DOCX)
        context: Nearby text/heading for context

    Returns:
        The image_id of the inserted record

    Raises:
        IntegrityError: If image with this file_id, page_number, and image_index exists
    """
    try:
        cursor = conn.execute(
            """
            INSERT INTO images (
                file_id, page_number, image_index, format,
                width, height, size_bytes, image_data, context
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                page_number,
                image_index,
                format,
                width,
                height,
                len(image_data),
                image_data,
                context,
            ),
        )
        conn.commit()
        if cursor.lastrowid is None:
            raise StorageError("Failed to insert image: lastrowid is None")
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise IntegrityError(str(e)) from e


def get_images(
    conn: sqlite3.Connection,
    file_id: int | None = None,
    *,
    unanalyzed_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Retrieve images, optionally filtered by file or analysis status.

    Args:
        conn: Database connection
        file_id: If provided, filter to images from this file only
        unanalyzed_only: If True, only return images that haven't been analyzed

    Returns:
        List of image dictionaries (without image_data for efficiency)
    """
    where_clauses = []
    params: list[Any] = []

    if file_id is not None:
        where_clauses.append("file_id = ?")
        params.append(file_id)

    if unanalyzed_only:
        where_clauses.append("analyzed_at IS NULL")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    cursor = conn.execute(
        f"""
        SELECT id, file_id, page_number, image_index, format,
               width, height, size_bytes, context, analyzed_at
        FROM images
        {where_sql}
        ORDER BY file_id, page_number, image_index
        """,
        params,
    )

    return [dict(row) for row in cursor.fetchall()]


def get_image_data(conn: sqlite3.Connection, image_id: int) -> bytes | None:
    """
    Retrieve the raw image data for a specific image.

    Args:
        conn: Database connection
        image_id: ID of the image

    Returns:
        Raw image bytes, or None if not found
    """
    cursor = conn.execute(
        "SELECT image_data FROM images WHERE id = ?",
        (image_id,),
    )
    row = cursor.fetchone()
    return row["image_data"] if row else None


def mark_image_analyzed(conn: sqlite3.Connection, image_id: int) -> None:
    """
    Mark an image as analyzed by setting analyzed_at timestamp.

    Args:
        conn: Database connection
        image_id: ID of the image to mark
    """
    conn.execute(
        "UPDATE images SET analyzed_at = datetime('now') WHERE id = ?",
        (image_id,),
    )
    conn.commit()


def get_unanalyzed_images(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    Get all images that haven't been analyzed yet.

    Returns images with their file paths for context.
    """
    cursor = conn.execute(
        """
        SELECT i.id, i.file_id, i.page_number, i.image_index, i.format,
               i.width, i.height, i.size_bytes, i.context, f.path as file_path
        FROM images i
        JOIN files f ON i.file_id = f.id
        WHERE i.analyzed_at IS NULL
        ORDER BY f.id, i.page_number, i.image_index
        """
    )
    return [dict(row) for row in cursor.fetchall()]


def count_images(conn: sqlite3.Connection, *, unanalyzed_only: bool = False) -> int:
    """Count total images, optionally only unanalyzed ones."""
    if unanalyzed_only:
        cursor = conn.execute("SELECT COUNT(*) FROM images WHERE analyzed_at IS NULL")
    else:
        cursor = conn.execute("SELECT COUNT(*) FROM images")
    return cursor.fetchone()[0]
