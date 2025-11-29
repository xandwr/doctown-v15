"""
Embedding engine for generating vector embeddings from chunks.

Uses sentence-transformers with lazy model loading for efficient
batch inference.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Callable

from docpack.storage import get_chunks, insert_vector, set_metadata

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Default model - google/embeddinggemma-300m produces 1024-dim vectors
DEFAULT_MODEL = "google/embeddinggemma-300m"

# Global model cache for lazy loading
_model_cache: dict[str, "SentenceTransformer"] = {}


def get_model(model_name: str = DEFAULT_MODEL) -> "SentenceTransformer":
    """
    Lazy-load and cache the embedding model.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Loaded SentenceTransformer model
    """
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)

    return _model_cache[model_name]


def get_embedding_dim(model_name: str = DEFAULT_MODEL) -> int:
    """
    Get the embedding dimension for a model.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Number of dimensions in the embedding
    """
    model = get_model(model_name)
    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        raise ValueError(f"Model '{model_name}' did not return a valid embedding dimension.")
    return dim


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    *,
    batch_size: int = 32,
    show_progress: bool = False,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model_name: HuggingFace model identifier
        batch_size: Number of texts to process at once
        show_progress: Show progress bar during encoding

    Returns:
        List of embedding vectors (as lists of floats)
    """
    if not texts:
        return []

    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    # Convert numpy arrays to lists
    return [emb.tolist() for emb in embeddings]


def embed_chunks(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    *,
    batch_size: int = 32,
    show_progress: bool = False,
) -> int:
    """
    Embed a batch of chunks and store vectors in database.

    Args:
        conn: Database connection
        chunk_ids: List of chunk IDs corresponding to texts
        texts: List of chunk texts to embed
        model_name: HuggingFace model identifier
        batch_size: Number of texts to process at once
        show_progress: Show progress bar during encoding

    Returns:
        Number of vectors inserted
    """
    if not texts:
        return 0

    if len(chunk_ids) != len(texts):
        raise ValueError("chunk_ids and texts must have same length")

    embeddings = embed_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    dims = len(embeddings[0]) if embeddings else 0

    for chunk_id, vector in zip(chunk_ids, embeddings):
        insert_vector(conn, chunk_id, dims, vector)

    return len(embeddings)


def embed_all(
    conn: sqlite3.Connection,
    model_name: str = DEFAULT_MODEL,
    *,
    batch_size: int = 32,
    verbose: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """
    Embed all chunks in the database.

    Args:
        conn: Database connection
        model_name: HuggingFace model identifier
        batch_size: Number of chunks to process at once
        verbose: Print progress information
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        Total number of vectors created
    """
    # Get all chunks
    chunks = get_chunks(conn)

    if not chunks:
        if verbose:
            print("No chunks to embed")
        return 0

    if verbose:
        print(f"Embedding {len(chunks)} chunks with {model_name}...")

    # Prepare batch data
    chunk_ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    total_chunks = len(texts)

    # Initial progress callback
    if progress_callback:
        progress_callback(0, total_chunks)

    # Embed in batches
    total_vectors = 0
    dims = get_embedding_dim(model_name)

    for i in range(0, len(texts), batch_size):
        batch_ids = chunk_ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        embeddings = embed_texts(
            batch_texts,
            model_name=model_name,
            batch_size=batch_size,
            show_progress=False,
        )

        for chunk_id, vector in zip(batch_ids, embeddings):
            insert_vector(conn, chunk_id, dims, vector)
            total_vectors += 1

        progress = min(i + batch_size, len(texts))

        if verbose:
            print(f"  Embedded {progress}/{len(texts)} chunks")

        if progress_callback:
            progress_callback(progress, total_chunks)

    # Update metadata
    set_metadata(conn, "vector_count", str(total_vectors))
    set_metadata(conn, "embedding_model", model_name)
    set_metadata(conn, "embedding_dims", str(dims))
    set_metadata(conn, "stage", "embedded")

    if verbose:
        print(f"\nTotal: {total_vectors} vectors ({dims} dimensions)")

    return total_vectors
