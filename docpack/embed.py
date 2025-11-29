"""
Embedding engine using Ollama.

Uses Ollama's embedding API for unified CPU/GPU inference.
Supports parallel processing for CPU mode optimization.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from docpack.runtime import RuntimeConfig, get_global_config
from docpack.storage import get_chunks, insert_vector, set_metadata


# Default model - nomic-embed-text is fast and produces 768-dim vectors
DEFAULT_MODEL = "nomic-embed-text"


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    config: RuntimeConfig | None = None,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Ollama.

    Args:
        texts: List of text strings to embed
        model: Ollama embedding model name
        config: Runtime configuration (uses global if not provided)

    Returns:
        List of embedding vectors (as lists of floats)
    """
    from ollama import embed

    if not texts:
        return []

    cfg = config or get_global_config()

    if cfg.force_cpu:
        # CPU mode: smaller batches processed in parallel
        return _embed_parallel(texts, model, cfg)
    else:
        # GPU mode: let Ollama batch efficiently
        return _embed_batch(texts, model, cfg.embedding_batch_size)


def _embed_batch(
    texts: list[str],
    model: str,
    batch_size: int,
) -> list[list[float]]:
    """Embed texts in batches (GPU-optimized)."""
    from ollama import embed

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = embed(model=model, input=batch)
        all_embeddings.extend(response.embeddings)

    return all_embeddings


def _embed_parallel(
    texts: list[str],
    model: str,
    config: RuntimeConfig,
) -> list[list[float]]:
    """
    Embed texts using parallel workers (CPU-optimized).

    Uses smaller batches across multiple threads to maximize
    CPU utilization without overwhelming memory.
    """
    from ollama import embed

    batch_size = config.embedding_batch_size  # Small batches for CPU
    num_workers = config.parallel_workers

    # Split into small batches
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append((i, texts[i : i + batch_size]))

    # Process in parallel
    results: dict[int, list[list[float]]] = {}

    def process_batch(batch_info: tuple[int, list[str]]) -> tuple[int, list[list[float]]]:
        idx, batch_texts = batch_info
        response = embed(model=model, input=batch_texts)
        return idx, response.embeddings

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch, b) for b in batches]

        for future in as_completed(futures):
            idx, embeddings = future.result()
            results[idx] = embeddings

    # Reassemble in order
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        all_embeddings.extend(results[i])

    return all_embeddings


def get_embedding_dim(model: str = DEFAULT_MODEL) -> int:
    """
    Get the embedding dimension for a model.

    Args:
        model: Ollama embedding model name

    Returns:
        Number of dimensions in the embedding
    """
    from ollama import embed

    # Embed a test string to get dimensions
    response = embed(model=model, input=["test"])
    return len(response.embeddings[0])


def embed_chunks(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
    texts: list[str],
    model: str = DEFAULT_MODEL,
    config: RuntimeConfig | None = None,
) -> int:
    """
    Embed a batch of chunks and store vectors in database.

    Args:
        conn: Database connection
        chunk_ids: List of chunk IDs corresponding to texts
        texts: List of chunk texts to embed
        model: Ollama embedding model name
        config: Runtime configuration

    Returns:
        Number of vectors inserted
    """
    if not texts:
        return 0

    if len(chunk_ids) != len(texts):
        raise ValueError("chunk_ids and texts must have same length")

    embeddings = embed_texts(texts, model=model, config=config)
    dims = len(embeddings[0]) if embeddings else 0

    for chunk_id, vector in zip(chunk_ids, embeddings):
        insert_vector(conn, chunk_id, dims, vector)

    return len(embeddings)


def embed_all(
    conn: sqlite3.Connection,
    model: str = DEFAULT_MODEL,
    config: RuntimeConfig | None = None,
    *,
    verbose: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """
    Embed all chunks in the database.

    Args:
        conn: Database connection
        model: Ollama embedding model name
        config: Runtime configuration
        verbose: Print progress information
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        Total number of vectors created
    """
    cfg = config or get_global_config()
    batch_size = cfg.embedding_batch_size

    # Get all chunks
    chunks = get_chunks(conn)

    if not chunks:
        if verbose:
            print("No chunks to embed")
        return 0

    if verbose:
        mode = "CPU (parallel)" if cfg.force_cpu else "GPU"
        print(f"Embedding {len(chunks)} chunks with {model} [{mode}]...")

    # Prepare batch data
    chunk_ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    total_chunks = len(texts)

    # Initial progress callback
    if progress_callback:
        progress_callback(0, total_chunks)

    # Get embedding dimensions
    dims = get_embedding_dim(model)

    # Process based on mode
    if cfg.force_cpu:
        # CPU mode: process all at once with internal parallelism
        embeddings = embed_texts(texts, model=model, config=cfg)

        for chunk_id, vector in zip(chunk_ids, embeddings):
            insert_vector(conn, chunk_id, dims, vector)

        if progress_callback:
            progress_callback(total_chunks, total_chunks)

        total_vectors = len(embeddings)
    else:
        # GPU mode: process in batches with progress updates
        total_vectors = 0

        for i in range(0, len(texts), batch_size):
            batch_ids = chunk_ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]

            embeddings = embed_texts(batch_texts, model=model, config=cfg)

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
    set_metadata(conn, "embedding_model", model)
    set_metadata(conn, "embedding_dims", str(dims))
    set_metadata(conn, "stage", "embedded")

    if verbose:
        print(f"\nTotal: {total_vectors} vectors ({dims} dimensions)")

    return total_vectors
