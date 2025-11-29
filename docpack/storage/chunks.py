"""Chunk storage operations."""

from __future__ import annotations

import sqlite3
from typing import Any

from .schema import IntegrityError, StorageError


def insert_chunk(
    conn: sqlite3.Connection,
    file_id: int,
    order_index: int,
    text: str,
    token_count: int,
    *,
    start_char: int | None = None,
    end_char: int | None = None,
) -> int:
    """
    Insert a chunk record linked to a file.

    Args:
        conn: Database connection
        file_id: ID of the parent file
        order_index: Position of chunk within the file (0-indexed)
        text: The chunk text content
        token_count: Number of tokens in the chunk
        start_char: Starting character position in original file
        end_char: Ending character position in original file

    Returns:
        The chunk_id of the inserted record

    Raises:
        IntegrityError: If chunk with this file_id and order_index exists
    """
    try:
        cursor = conn.execute(
            """
            INSERT INTO chunks (file_id, chunk_index, text, token_count, start_char, end_char)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_id, order_index, text, token_count, start_char, end_char),
        )
        conn.commit()
        if cursor.lastrowid is None:
            raise StorageError("Failed to insert chunk: lastrowid is None")
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise IntegrityError(str(e)) from e


def get_chunks(
    conn: sqlite3.Connection,
    file_id: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve chunks, optionally filtered by file.

    Args:
        conn: Database connection
        file_id: If provided, filter to chunks from this file only

    Returns:
        List of chunk dictionaries with keys:
        id, file_id, chunk_index, text, token_count, start_char, end_char, summary
    """
    if file_id is not None:
        cursor = conn.execute(
            """
            SELECT id, file_id, chunk_index, text, token_count, start_char, end_char, summary
            FROM chunks
            WHERE file_id = ?
            ORDER BY chunk_index
            """,
            (file_id,),
        )
    else:
        cursor = conn.execute(
            """
            SELECT id, file_id, chunk_index, text, token_count, start_char, end_char, summary
            FROM chunks
            ORDER BY file_id, chunk_index
            """
        )

    return [dict(row) for row in cursor.fetchall()]
