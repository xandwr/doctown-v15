"""File storage operations."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from .schema import IntegrityError, StorageError


def insert_file(
    conn: sqlite3.Connection,
    path: str,
    size_bytes: int,
    sha256_hash: str,
    *,
    content: str | None = None,
    is_binary: bool = False,
    extension: str | None = None,
) -> int:
    """
    Insert a file record into the database.

    Args:
        conn: Database connection
        path: File path (relative to docpack root)
        size_bytes: File size in bytes
        sha256_hash: SHA-256 hash of file content
        content: Text content (None for binary files)
        is_binary: Whether file is binary
        extension: File extension (auto-derived from path if None)

    Returns:
        The file_id of the inserted record

    Raises:
        IntegrityError: If file with this path already exists
    """
    if extension is None:
        _, ext = os.path.splitext(path)
        extension = ext.lower() if ext else None

    try:
        cursor = conn.execute(
            """
            INSERT INTO files (path, extension, size_bytes, is_binary, content, sha256_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (path, extension, size_bytes, int(is_binary), content, sha256_hash),
        )
        conn.commit()
        if cursor.lastrowid is None:
            raise StorageError("Failed to insert file: lastrowid is None")
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        conn.rollback()
        if "UNIQUE constraint" in str(e):
            raise IntegrityError(f"File with path '{path}' already exists") from e
        raise IntegrityError(str(e)) from e


def get_file(conn: sqlite3.Connection, file_id: int) -> dict[str, Any] | None:
    """Retrieve a file by ID."""
    cursor = conn.execute(
        """
        SELECT id, path, extension, size_bytes, is_binary, content, sha256_hash
        FROM files WHERE id = ?
        """,
        (file_id,),
    )
    row = cursor.fetchone()
    if row:
        result = dict(row)
        result["is_binary"] = bool(result["is_binary"])
        return result
    return None


def get_file_by_path(conn: sqlite3.Connection, path: str) -> dict[str, Any] | None:
    """Retrieve a file by path."""
    cursor = conn.execute(
        """
        SELECT id, path, extension, size_bytes, is_binary, content, sha256_hash
        FROM files WHERE path = ?
        """,
        (path,),
    )
    row = cursor.fetchone()
    if row:
        result = dict(row)
        result["is_binary"] = bool(result["is_binary"])
        return result
    return None


def get_all_files(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Retrieve all files (without content for efficiency)."""
    cursor = conn.execute(
        """
        SELECT id, path, extension, size_bytes, is_binary, sha256_hash
        FROM files ORDER BY path
        """
    )
    results = []
    for row in cursor.fetchall():
        result = dict(row)
        result["is_binary"] = bool(result["is_binary"])
        results.append(result)
    return results
