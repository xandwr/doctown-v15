"""Database connection management."""

from __future__ import annotations

import sqlite3

from .schema import SCHEMA_SQL


def init_db(path: str) -> sqlite3.Connection:
    """
    Initialize or open a docpack database.

    Creates all required tables if they don't exist.
    Configures connection for optimal performance.

    Args:
        path: Path to the .docpack SQLite file

    Returns:
        Configured sqlite3.Connection ready for use
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Performance and safety settings
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Create tables
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    return conn
