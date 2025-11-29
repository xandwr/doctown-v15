"""
Structured query tool for safe SQL-like queries against docpack.

This provides SQL-like query power without the risk of SQL injection.
Queries are built using a safe query builder pattern with parameterized values.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class QueryType(str, Enum):
    """Types of queries supported."""

    FILES = "files"
    CHUNKS = "chunks"
    STATS = "stats"


@dataclass
class QueryFilter:
    """Filter conditions for a query."""

    extension: str | None = None
    path_contains: str | None = None
    path_glob: str | None = None
    size_min: int | None = None
    size_max: int | None = None
    is_binary: bool | None = None
    content_contains: str | None = None


@dataclass
class QueryResult:
    """Result of a structured query."""

    rows: list[dict[str, Any]]
    total_count: int
    query_type: str
    truncated: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rows": self.rows,
            "total_count": self.total_count,
            "query_type": self.query_type,
            "truncated": self.truncated,
        }


def structured_query(
    conn: sqlite3.Connection,
    query_type: QueryType | str,
    *,
    filter: QueryFilter | dict | None = None,
    order_by: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> QueryResult:
    """
    Execute a structured query against the docpack.

    Args:
        conn: Database connection
        query_type: Type of query (files, chunks, stats)
        filter: Optional filter conditions
        order_by: Column to order by (path, size, tokens, etc.)
        limit: Maximum rows to return
        offset: Number of rows to skip

    Returns:
        QueryResult with rows and metadata
    """
    if isinstance(query_type, str):
        query_type = QueryType(query_type)

    if isinstance(filter, dict):
        filter = QueryFilter(**filter)

    if filter is None:
        filter = QueryFilter()

    if query_type == QueryType.FILES:
        return _query_files(conn, filter, order_by, limit, offset)
    elif query_type == QueryType.CHUNKS:
        return _query_chunks(conn, filter, order_by, limit, offset)
    elif query_type == QueryType.STATS:
        return _query_stats(conn, filter)
    else:
        raise ValueError(f"Unknown query type: {query_type}")


def _query_files(
    conn: sqlite3.Connection,
    filter: QueryFilter,
    order_by: str | None,
    limit: int,
    offset: int,
) -> QueryResult:
    """Query the files table."""
    conditions = []
    params: list[Any] = []

    if filter.extension is not None:
        # Extensions are stored with the dot (e.g., ".py")
        ext = filter.extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        conditions.append("extension = ?")
        params.append(ext)

    if filter.path_contains is not None:
        conditions.append("path LIKE ?")
        params.append(f"%{filter.path_contains}%")

    if filter.path_glob is not None:
        conditions.append("path GLOB ?")
        params.append(filter.path_glob)

    if filter.size_min is not None:
        conditions.append("size_bytes >= ?")
        params.append(filter.size_min)

    if filter.size_max is not None:
        conditions.append("size_bytes <= ?")
        params.append(filter.size_max)

    if filter.is_binary is not None:
        conditions.append("is_binary = ?")
        params.append(1 if filter.is_binary else 0)

    if filter.content_contains is not None:
        conditions.append("content LIKE ?")
        params.append(f"%{filter.content_contains}%")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Validate order_by to prevent injection
    valid_order_columns = {"path", "size_bytes", "extension", "id"}
    if order_by and order_by.lstrip("-") not in valid_order_columns:
        order_by = "path"

    order_direction = "DESC" if order_by and order_by.startswith("-") else "ASC"
    order_column = order_by.lstrip("-") if order_by else "path"

    # Count total
    count_sql = f"SELECT COUNT(*) FROM files WHERE {where_clause}"
    total_count = conn.execute(count_sql, params).fetchone()[0]

    # Get rows
    sql = f"""
        SELECT id, path, extension, size_bytes, is_binary, sha256_hash
        FROM files
        WHERE {where_clause}
        ORDER BY {order_column} {order_direction}
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    cursor = conn.execute(sql, params)
    rows = []
    for row in cursor.fetchall():
        rows.append(
            {
                "id": row[0],
                "path": row[1],
                "extension": row[2],
                "size_bytes": row[3],
                "is_binary": bool(row[4]),
                "sha256_hash": row[5],
            }
        )

    return QueryResult(
        rows=rows,
        total_count=total_count,
        query_type="files",
        truncated=len(rows) < total_count - offset,
    )


def _query_chunks(
    conn: sqlite3.Connection,
    filter: QueryFilter,
    order_by: str | None,
    limit: int,
    offset: int,
) -> QueryResult:
    """Query the chunks table with file info."""
    conditions = []
    params: list[Any] = []

    if filter.extension is not None:
        # Extensions are stored with the dot (e.g., ".py")
        ext = filter.extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        conditions.append("f.extension = ?")
        params.append(ext)

    if filter.path_contains is not None:
        conditions.append("f.path LIKE ?")
        params.append(f"%{filter.path_contains}%")

    if filter.path_glob is not None:
        conditions.append("f.path GLOB ?")
        params.append(filter.path_glob)

    if filter.content_contains is not None:
        conditions.append("c.text LIKE ?")
        params.append(f"%{filter.content_contains}%")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Validate order_by
    valid_order_columns = {"path", "token_count", "chunk_index", "id"}
    if order_by and order_by.lstrip("-") not in valid_order_columns:
        order_by = "path"

    order_direction = "DESC" if order_by and order_by.startswith("-") else "ASC"
    order_column = order_by.lstrip("-") if order_by else "f.path, c.chunk_index"
    if order_column in ("path",):
        order_column = f"f.{order_column}"
    elif order_column in ("token_count", "chunk_index", "id"):
        order_column = f"c.{order_column}"

    # Count total
    count_sql = f"""
        SELECT COUNT(*)
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        WHERE {where_clause}
    """
    total_count = conn.execute(count_sql, params).fetchone()[0]

    # Get rows
    sql = f"""
        SELECT c.id, f.path, c.chunk_index, c.text, c.token_count, c.start_char, c.end_char
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        WHERE {where_clause}
        ORDER BY {order_column} {order_direction}
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    cursor = conn.execute(sql, params)
    rows = []
    for row in cursor.fetchall():
        rows.append(
            {
                "id": row[0],
                "file_path": row[1],
                "chunk_index": row[2],
                "text": row[3],
                "token_count": row[4],
                "start_char": row[5],
                "end_char": row[6],
            }
        )

    return QueryResult(
        rows=rows,
        total_count=total_count,
        query_type="chunks",
        truncated=len(rows) < total_count - offset,
    )


def _query_stats(
    conn: sqlite3.Connection,
    filter: QueryFilter,
) -> QueryResult:
    """Get aggregate statistics."""
    rows = []

    # Overall stats
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as file_count,
            SUM(size_bytes) as total_bytes,
            SUM(CASE WHEN is_binary = 1 THEN 1 ELSE 0 END) as binary_count,
            SUM(CASE WHEN is_binary = 0 THEN 1 ELSE 0 END) as text_count
        FROM files
        """
    )
    row = cursor.fetchone()
    rows.append(
        {
            "stat": "files_overview",
            "file_count": row[0],
            "total_bytes": row[1],
            "binary_count": row[2],
            "text_count": row[3],
        }
    )

    # Stats by extension
    cursor = conn.execute(
        """
        SELECT
            COALESCE(extension, '(none)') as extension,
            COUNT(*) as count,
            SUM(size_bytes) as total_bytes
        FROM files
        GROUP BY extension
        ORDER BY count DESC
        LIMIT 20
        """
    )
    for row in cursor.fetchall():
        rows.append(
            {
                "stat": "by_extension",
                "extension": row[0],
                "count": row[1],
                "total_bytes": row[2],
            }
        )

    # Chunk stats
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as chunk_count,
            SUM(token_count) as total_tokens,
            AVG(token_count) as avg_tokens_per_chunk
        FROM chunks
        """
    )
    row = cursor.fetchone()
    if row[0]:
        rows.append(
            {
                "stat": "chunks_overview",
                "chunk_count": row[0],
                "total_tokens": row[1],
                "avg_tokens_per_chunk": round(row[2], 1) if row[2] else None,
            }
        )

    # Marginalia stats
    cursor = conn.execute("SELECT COUNT(*) FROM notes")
    note_count = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM sessions")
    session_count = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM artifacts")
    artifact_count = cursor.fetchone()[0]

    if note_count or session_count or artifact_count:
        rows.append(
            {
                "stat": "marginalia",
                "note_count": note_count,
                "session_count": session_count,
                "artifact_count": artifact_count,
            }
        )

    return QueryResult(
        rows=rows,
        total_count=len(rows),
        query_type="stats",
        truncated=False,
    )


def format_query_result(result: QueryResult) -> str:
    """Format query result for display."""
    if not result.rows:
        return f"No results for {result.query_type} query"

    lines = []
    lines.append(f"Query type: {result.query_type}")
    lines.append(f"Total: {result.total_count} rows")
    if result.truncated:
        lines.append(f"(showing {len(result.rows)} rows)")
    lines.append("")

    if result.query_type == "files":
        for row in result.rows:
            status = "[bin]" if row["is_binary"] else "[txt]"
            lines.append(f"{status} {row['path']} ({row['size_bytes']} bytes)")

    elif result.query_type == "chunks":
        for row in result.rows:
            lines.append(
                f"{row['file_path']}:{row['chunk_index']} ({row['token_count']} tokens)"
            )
            # Show first 100 chars of text
            text_preview = row["text"][:100].replace("\n", " ")
            if len(row["text"]) > 100:
                text_preview += "..."
            lines.append(f"  {text_preview}")
            lines.append("")

    elif result.query_type == "stats":
        for row in result.rows:
            stat_type = row.pop("stat")
            lines.append(f"[{stat_type}]")
            for k, v in row.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

    return "\n".join(lines)
