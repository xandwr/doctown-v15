"""
Semantic recall via vector search.

Implements cosine similarity k-nearest neighbors search
against embedded chunks in a docpack.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass

from docpack.embed import embed_texts, DEFAULT_MODEL
from docpack.runtime import RuntimeConfig, get_global_config


@dataclass
class RecallResult:
    """A single recall result with chunk and file metadata."""

    chunk_id: int
    file_id: int
    file_path: str
    chunk_index: int
    text: str
    score: float
    start_char: int | None = None
    end_char: int | None = None
    summary: str | None = None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def load_all_vectors(conn: sqlite3.Connection) -> list[tuple[int, int, list[float]]]:
    """
    Load all vectors from database.

    Returns:
        List of (vector_id, chunk_id, vector) tuples
    """
    cursor = conn.execute(
        """
        SELECT id, chunk_id, dims, vector
        FROM vectors
        """
    )

    results = []
    for row in cursor.fetchall():
        dims = row["dims"]
        blob = row["vector"]
        vec = list(struct.unpack(f"<{dims}f", blob))
        results.append((row["id"], row["chunk_id"], vec))

    return results


def recall(
    conn: sqlite3.Connection,
    query: str,
    k: int = 5,
    model: str = DEFAULT_MODEL,
    threshold: float | None = None,
    config: RuntimeConfig | None = None,
) -> list[RecallResult]:
    """
    Semantic search against embedded chunks.

    Args:
        conn: Database connection to a docpack
        query: Natural language query string
        k: Number of results to return
        model: Embedding model (must match model used during freeze)
        threshold: Optional minimum similarity score (0-1)
        config: Runtime configuration

    Returns:
        List of RecallResult objects sorted by similarity (highest first)
    """
    cfg = config or get_global_config()

    # Embed the query
    query_embedding = embed_texts([query], model=model, config=cfg)[0]

    # Load all vectors
    vectors = load_all_vectors(conn)

    if not vectors:
        return []

    # Compute similarities
    scored: list[tuple[int, float]] = []
    for vector_id, chunk_id, vec in vectors:
        score = cosine_similarity(query_embedding, vec)
        if threshold is None or score >= threshold:
            scored.append((chunk_id, score))

    # Sort by score descending and take top k
    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = scored[:k]

    # Fetch chunk and file metadata for results
    results = []
    for chunk_id, score in top_k:
        cursor = conn.execute(
            """
            SELECT c.id, c.file_id, c.chunk_index, c.text, c.start_char, c.end_char,
                   c.summary, f.path
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.id = ?
            """,
            (chunk_id,),
        )
        row = cursor.fetchone()
        if row:
            results.append(
                RecallResult(
                    chunk_id=row["id"],
                    file_id=row["file_id"],
                    file_path=row["path"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    score=score,
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    summary=row["summary"],
                )
            )

    return results
