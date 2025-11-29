"""Vector storage operations."""

from __future__ import annotations

import sqlite3
import struct
from typing import Any

from .schema import IntegrityError, StorageError


def insert_vector(
    conn: sqlite3.Connection,
    chunk_id: int,
    dims: int,
    vector: list[float],
) -> int:
    """
    Insert a vector embedding for a chunk.

    Args:
        conn: Database connection
        chunk_id: ID of the parent chunk
        dims: Number of dimensions (for validation)
        vector: The embedding vector as list of floats

    Returns:
        The vector_id of the inserted record

    Raises:
        ValueError: If vector length doesn't match dims
        IntegrityError: On constraint violation
    """
    if len(vector) != dims:
        raise ValueError(f"Vector length {len(vector)} doesn't match dims {dims}")

    # Pack as raw bytes (little-endian float32)
    blob = struct.pack(f"<{dims}f", *vector)

    try:
        cursor = conn.execute(
            """
            INSERT INTO vectors (chunk_id, dims, vector)
            VALUES (?, ?, ?)
            """,
            (chunk_id, dims, blob),
        )
        conn.commit()
        if cursor.lastrowid is None:
            raise StorageError("Failed to insert vector: lastrowid is None")
        return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise IntegrityError(str(e)) from e


def get_vectors(
    conn: sqlite3.Connection,
    chunk_id: int,
) -> list[dict[str, Any]]:
    """
    Retrieve vectors for a specific chunk.

    Args:
        conn: Database connection
        chunk_id: The chunk ID to get vectors for

    Returns:
        List of vector dictionaries with keys:
        id, chunk_id, dims, vector (as list of floats)
    """
    cursor = conn.execute(
        """
        SELECT id, chunk_id, dims, vector
        FROM vectors
        WHERE chunk_id = ?
        """,
        (chunk_id,),
    )

    results = []
    for row in cursor.fetchall():
        blob = row["vector"]
        dims = row["dims"]
        vec = list(struct.unpack(f"<{dims}f", blob))

        results.append(
            {
                "id": row["id"],
                "chunk_id": row["chunk_id"],
                "dims": dims,
                "vector": vec,
            }
        )

    return results
