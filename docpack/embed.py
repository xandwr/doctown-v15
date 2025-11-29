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


def embed_single(text: str, model: str = DEFAULT_MODEL) -> list[float]:
    """Embed a single text string."""
    from ollama import embed

    response = embed(model=model, input=[text])
    return response.embeddings[0]


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
        # CPU mode: single items processed in parallel
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

    Processes single items across multiple threads.
    """
    from ollama import embed

    num_workers = config.parallel_workers

    # Process in parallel, one text at a time
    results: dict[int, list[float]] = {}

    def process_single(idx_text: tuple[int, str]) -> tuple[int, list[float]]:
        idx, text = idx_text
        response = embed(model=model, input=[text])
        return idx, response.embeddings[0]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single, (i, t)) for i, t in enumerate(texts)]

        for future in as_completed(futures):
            idx, embedding = future.result()
            results[idx] = embedding

    # Reassemble in order
    return [results[i] for i in range(len(texts))]


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

    # Get all chunks
    chunks = get_chunks(conn)

    if not chunks:
        if verbose:
            print("No chunks to embed")
        return 0

    if verbose:
        mode = "CPU (parallel)" if cfg.force_cpu else "GPU"
        print(f"Embedding {len(chunks)} chunks with {model} [{mode}]...")

    # Prepare data
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
        # CPU mode: parallel processing with per-item progress
        total_vectors = _embed_all_parallel(
            conn, chunk_ids, texts, model, dims, cfg,
            verbose, progress_callback
        )
    else:
        # GPU mode: batch processing
        total_vectors = _embed_all_batched(
            conn, chunk_ids, texts, model, dims, cfg,
            verbose, progress_callback
        )

    # Update metadata
    set_metadata(conn, "vector_count", str(total_vectors))
    set_metadata(conn, "embedding_model", model)
    set_metadata(conn, "embedding_dims", str(dims))
    set_metadata(conn, "stage", "embedded")

    if verbose:
        print(f"\nTotal: {total_vectors} vectors ({dims} dimensions)")

    return total_vectors


def _embed_all_parallel(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
    texts: list[str],
    model: str,
    dims: int,
    config: RuntimeConfig,
    verbose: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> int:
    """Embed all chunks in parallel with per-item progress (CPU mode)."""
    from ollama import embed

    total = len(texts)
    completed = 0

    def process_single(item: tuple[int, str]) -> tuple[int, list[float]]:
        """Embed a single text, return (chunk_id, embedding)."""
        chunk_id, text = item
        response = embed(model=model, input=[text])
        return chunk_id, response.embeddings[0]

    # Prepare work items
    work_items = list(zip(chunk_ids, texts))

    with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures = {executor.submit(process_single, item): item for item in work_items}

        for future in as_completed(futures):
            chunk_id, vector = future.result()
            insert_vector(conn, chunk_id, dims, vector)
            completed += 1

            if progress_callback:
                progress_callback(completed, total)

            if verbose and completed % 10 == 0:
                print(f"  Embedded {completed}/{total} chunks")

    return completed


def _embed_all_batched(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
    texts: list[str],
    model: str,
    dims: int,
    config: RuntimeConfig,
    verbose: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> int:
    """Embed all chunks in batches with progress (GPU mode)."""
    batch_size = config.embedding_batch_size
    total = len(texts)
    total_vectors = 0

    for i in range(0, total, batch_size):
        batch_ids = chunk_ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        embeddings = embed_texts(batch_texts, model=model, config=config)

        for chunk_id, vector in zip(batch_ids, embeddings):
            insert_vector(conn, chunk_id, dims, vector)
            total_vectors += 1

        progress = min(i + batch_size, total)

        if verbose:
            print(f"  Embedded {progress}/{total} chunks")

        if progress_callback:
            progress_callback(progress, total)

    return total_vectors
