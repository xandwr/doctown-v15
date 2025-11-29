"""
Marginalia storage operations.

The marginalia is the living layer of a docpack - notes, sessions, and artifacts
that grow with each exploration. Unlike the frozen realm (files, chunks, vectors),
the marginalia can be written to at any time.

Think of it like a scholar's handwriting in the margins of an ancient manuscript.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class Note:
    """A note or finding stored in the marginalia."""

    id: int
    key: str
    content: str
    author: str | None
    created_at: str
    updated_at: str


@dataclass
class Session:
    """An exploration session - tracks who explored this universe."""

    id: str
    started_at: str
    ended_at: str | None
    task: str | None
    tool_calls: int


@dataclass
class Artifact:
    """A generated artifact from a workflow."""

    id: int
    session_id: str | None
    name: str
    content_type: str | None
    content: str
    created_at: str


# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------


def write_note(
    conn: sqlite3.Connection,
    key: str,
    content: str,
    *,
    author: str | None = None,
) -> Note:
    """
    Write or update a note in the marginalia.

    If a note with the given key exists, it will be updated.
    Otherwise, a new note is created.

    Args:
        conn: Database connection
        key: Unique identifier for the note
        content: The note content (markdown supported)
        author: Optional author identifier

    Returns:
        The created or updated Note
    """
    now = datetime.utcnow().isoformat()

    cursor = conn.execute(
        """
        INSERT INTO notes (key, content, author, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            content = excluded.content,
            author = COALESCE(excluded.author, notes.author),
            updated_at = excluded.updated_at
        RETURNING id, key, content, author, created_at, updated_at
        """,
        (key, content, author, now, now),
    )
    row = cursor.fetchone()
    conn.commit()

    return Note(
        id=row[0],
        key=row[1],
        content=row[2],
        author=row[3],
        created_at=row[4],
        updated_at=row[5],
    )


def read_note(conn: sqlite3.Connection, key: str) -> Note | None:
    """Read a note by its key."""
    cursor = conn.execute(
        """
        SELECT id, key, content, author, created_at, updated_at
        FROM notes WHERE key = ?
        """,
        (key,),
    )
    row = cursor.fetchone()
    if row:
        return Note(
            id=row[0],
            key=row[1],
            content=row[2],
            author=row[3],
            created_at=row[4],
            updated_at=row[5],
        )
    return None


def read_all_notes(conn: sqlite3.Connection) -> list[Note]:
    """Read all notes from the marginalia."""
    cursor = conn.execute(
        """
        SELECT id, key, content, author, created_at, updated_at
        FROM notes ORDER BY updated_at DESC
        """
    )
    return [
        Note(
            id=row[0],
            key=row[1],
            content=row[2],
            author=row[3],
            created_at=row[4],
            updated_at=row[5],
        )
        for row in cursor.fetchall()
    ]


def delete_note(conn: sqlite3.Connection, key: str) -> bool:
    """Delete a note by its key. Returns True if deleted, False if not found."""
    cursor = conn.execute("DELETE FROM notes WHERE key = ?", (key,))
    conn.commit()
    return cursor.rowcount > 0


# -----------------------------------------------------------------------------
# Sessions
# -----------------------------------------------------------------------------


def start_session(
    conn: sqlite3.Connection,
    *,
    task: str | None = None,
    session_id: str | None = None,
) -> Session:
    """
    Start a new exploration session.

    Args:
        conn: Database connection
        task: Optional description of what this session is for
        session_id: Optional custom session ID (generates UUID if not provided)

    Returns:
        The new Session
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    now = datetime.utcnow().isoformat()

    conn.execute(
        """
        INSERT INTO sessions (id, started_at, task, tool_calls)
        VALUES (?, ?, ?, 0)
        """,
        (session_id, now, task),
    )
    conn.commit()

    return Session(
        id=session_id,
        started_at=now,
        ended_at=None,
        task=task,
        tool_calls=0,
    )


def end_session(conn: sqlite3.Connection, session_id: str) -> Session | None:
    """End an exploration session."""
    now = datetime.utcnow().isoformat()

    cursor = conn.execute(
        """
        UPDATE sessions SET ended_at = ?
        WHERE id = ? AND ended_at IS NULL
        RETURNING id, started_at, ended_at, task, tool_calls
        """,
        (now, session_id),
    )
    row = cursor.fetchone()
    conn.commit()

    if row:
        return Session(
            id=row[0],
            started_at=row[1],
            ended_at=row[2],
            task=row[3],
            tool_calls=row[4],
        )
    return None


def increment_tool_calls(conn: sqlite3.Connection, session_id: str) -> None:
    """Increment the tool call counter for a session."""
    conn.execute(
        "UPDATE sessions SET tool_calls = tool_calls + 1 WHERE id = ?",
        (session_id,),
    )
    conn.commit()


def get_session(conn: sqlite3.Connection, session_id: str) -> Session | None:
    """Get a session by ID."""
    cursor = conn.execute(
        """
        SELECT id, started_at, ended_at, task, tool_calls
        FROM sessions WHERE id = ?
        """,
        (session_id,),
    )
    row = cursor.fetchone()
    if row:
        return Session(
            id=row[0],
            started_at=row[1],
            ended_at=row[2],
            task=row[3],
            tool_calls=row[4],
        )
    return None


def get_all_sessions(conn: sqlite3.Connection) -> list[Session]:
    """Get all sessions, most recent first."""
    cursor = conn.execute(
        """
        SELECT id, started_at, ended_at, task, tool_calls
        FROM sessions ORDER BY started_at DESC
        """
    )
    return [
        Session(
            id=row[0],
            started_at=row[1],
            ended_at=row[2],
            task=row[3],
            tool_calls=row[4],
        )
        for row in cursor.fetchall()
    ]


# -----------------------------------------------------------------------------
# Artifacts
# -----------------------------------------------------------------------------


def save_artifact(
    conn: sqlite3.Connection,
    name: str,
    content: str,
    *,
    content_type: str | None = None,
    session_id: str | None = None,
) -> Artifact:
    """
    Save an artifact to the marginalia.

    Args:
        conn: Database connection
        name: Name/identifier for the artifact
        content: The artifact content
        content_type: MIME type or format (e.g., "application/json", "text/markdown")
        session_id: Optional session this artifact belongs to

    Returns:
        The saved Artifact
    """
    now = datetime.utcnow().isoformat()

    cursor = conn.execute(
        """
        INSERT INTO artifacts (session_id, name, content_type, content, created_at)
        VALUES (?, ?, ?, ?, ?)
        RETURNING id, session_id, name, content_type, content, created_at
        """,
        (session_id, name, content_type, content, now),
    )
    row = cursor.fetchone()
    conn.commit()

    return Artifact(
        id=row[0],
        session_id=row[1],
        name=row[2],
        content_type=row[3],
        content=row[4],
        created_at=row[5],
    )


def get_artifact(conn: sqlite3.Connection, artifact_id: int) -> Artifact | None:
    """Get an artifact by ID."""
    cursor = conn.execute(
        """
        SELECT id, session_id, name, content_type, content, created_at
        FROM artifacts WHERE id = ?
        """,
        (artifact_id,),
    )
    row = cursor.fetchone()
    if row:
        return Artifact(
            id=row[0],
            session_id=row[1],
            name=row[2],
            content_type=row[3],
            content=row[4],
            created_at=row[5],
        )
    return None


def get_artifacts_by_name(conn: sqlite3.Connection, name: str) -> list[Artifact]:
    """Get all artifacts with a given name."""
    cursor = conn.execute(
        """
        SELECT id, session_id, name, content_type, content, created_at
        FROM artifacts WHERE name = ? ORDER BY created_at DESC
        """,
        (name,),
    )
    return [
        Artifact(
            id=row[0],
            session_id=row[1],
            name=row[2],
            content_type=row[3],
            content=row[4],
            created_at=row[5],
        )
        for row in cursor.fetchall()
    ]


def get_session_artifacts(conn: sqlite3.Connection, session_id: str) -> list[Artifact]:
    """Get all artifacts from a session."""
    cursor = conn.execute(
        """
        SELECT id, session_id, name, content_type, content, created_at
        FROM artifacts WHERE session_id = ? ORDER BY created_at
        """,
        (session_id,),
    )
    return [
        Artifact(
            id=row[0],
            session_id=row[1],
            name=row[2],
            content_type=row[3],
            content=row[4],
            created_at=row[5],
        )
        for row in cursor.fetchall()
    ]


def get_all_artifacts(conn: sqlite3.Connection) -> list[Artifact]:
    """Get all artifacts, most recent first."""
    cursor = conn.execute(
        """
        SELECT id, session_id, name, content_type, content, created_at
        FROM artifacts ORDER BY created_at DESC
        """
    )
    return [
        Artifact(
            id=row[0],
            session_id=row[1],
            name=row[2],
            content_type=row[3],
            content=row[4],
            created_at=row[5],
        )
        for row in cursor.fetchall()
    ]
