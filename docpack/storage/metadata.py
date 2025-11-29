"""Metadata and statistics operations."""

from __future__ import annotations

import sqlite3
from typing import Any


def set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Insert or update a metadata key-value pair."""
    conn.execute(
        """
        INSERT INTO metadata (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()


def get_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    """Retrieve a metadata value by key."""
    cursor = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,))
    row = cursor.fetchone()
    return row["value"] if row else None


def get_all_metadata(conn: sqlite3.Connection) -> dict[str, str]:
    """Retrieve all metadata as a dictionary."""
    cursor = conn.execute("SELECT key, value FROM metadata")
    return {row["key"]: row["value"] for row in cursor.fetchall()}


def get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Get summary statistics for the docpack."""
    stats: dict[str, Any] = {}

    cursor = conn.execute("SELECT COUNT(*) as count FROM files")
    stats["total_files"] = cursor.fetchone()["count"]

    cursor = conn.execute("SELECT COUNT(*) as count FROM chunks")
    stats["total_chunks"] = cursor.fetchone()["count"]

    cursor = conn.execute("SELECT COUNT(*) as count FROM vectors")
    stats["total_vectors"] = cursor.fetchone()["count"]

    cursor = conn.execute("SELECT SUM(size_bytes) as total FROM files")
    row = cursor.fetchone()
    stats["total_size_bytes"] = row["total"] or 0

    return stats
