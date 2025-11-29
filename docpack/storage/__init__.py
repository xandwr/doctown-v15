"""
Docpack storage layer.

All SQL operations are encapsulated here. No other module should
contain SQL strings or direct database operations.

The storage layer has two realms:
- FROZEN REALM: files, chunks, vectors, metadata (written once during freeze)
- MARGINALIA: notes, sessions, artifacts (grows with each exploration)

Usage:
    from docpack.storage import init_db, insert_file, insert_chunk, insert_vector

    conn = init_db("project.docpack")
    file_id = insert_file(conn, "src/main.py", 1234, "abc123...")
    chunk_id = insert_chunk(conn, file_id, 0, "def main():", 5)
    insert_vector(conn, chunk_id, 384, embedding)

    # Marginalia
    from docpack.storage import write_note, start_session, save_artifact
    write_note(conn, "finding-1", "Found an interesting pattern...")
"""

from .chunks import get_chunks, insert_chunk
from .connection import init_db
from .files import get_all_files, get_file, get_file_by_path, insert_file
from .marginalia import (
    Artifact,
    Note,
    Session,
    delete_note,
    end_session,
    get_all_artifacts,
    get_all_sessions,
    get_artifact,
    get_artifacts_by_name,
    get_session,
    get_session_artifacts,
    increment_tool_calls,
    read_all_notes,
    read_note,
    save_artifact,
    start_session,
    write_note,
)
from .metadata import get_all_metadata, get_metadata, get_stats, set_metadata
from .schema import IntegrityError, NotFoundError, StorageError
from .vectors import get_vectors, insert_vector

__all__ = [
    # Connection
    "init_db",
    # Files (frozen realm)
    "insert_file",
    "get_file",
    "get_file_by_path",
    "get_all_files",
    # Chunks (frozen realm)
    "insert_chunk",
    "get_chunks",
    # Vectors (frozen realm)
    "insert_vector",
    "get_vectors",
    # Metadata (frozen realm)
    "set_metadata",
    "get_metadata",
    "get_all_metadata",
    "get_stats",
    # Marginalia - Notes
    "Note",
    "write_note",
    "read_note",
    "read_all_notes",
    "delete_note",
    # Marginalia - Sessions
    "Session",
    "start_session",
    "end_session",
    "get_session",
    "get_all_sessions",
    "increment_tool_calls",
    # Marginalia - Artifacts
    "Artifact",
    "save_artifact",
    "get_artifact",
    "get_artifacts_by_name",
    "get_session_artifacts",
    "get_all_artifacts",
    # Exceptions
    "StorageError",
    "IntegrityError",
    "NotFoundError",
]
