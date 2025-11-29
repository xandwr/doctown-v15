"""
LLM-powered summarization for chunks.

Uses Ollama with structured output for reliable summarization.
Supports parallel processing for CPU mode.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from pydantic import BaseModel

from docpack.storage import set_metadata
from docpack.runtime import RuntimeConfig, get_global_config


DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_BATCH_SIZE = 8


class ChunkSummary(BaseModel):
    """Summary for a single chunk."""
    s: str  # summary only - we use position for matching


class SingleSummary(BaseModel):
    """Summary for a single chunk (CPU mode)."""
    summary: str


class BatchSummaries(BaseModel):
    """Batch of chunk summaries in order."""
    summaries: list[ChunkSummary]


def build_batch_prompt(chunks: list[dict]) -> str:
    """Build a prompt for batch summarization."""
    chunk_texts = []
    for i, c in enumerate(chunks):
        text = c['text'][:1024] if len(c['text']) > 1024 else c['text']
        chunk_texts.append(f"### {i+1}. {c['file_path']}\n{text}")

    return f"""Summarize each of the {len(chunks)} code chunks below in 1 sentence each. Return exactly {len(chunks)} summaries in order.

{chr(10).join(chunk_texts)}"""


def build_single_prompt(chunk: dict) -> str:
    """Build a prompt for single chunk summarization."""
    text = chunk['text'][:1024] if len(chunk['text']) > 1024 else chunk['text']
    return f"""Summarize this code chunk in 1 sentence.

### {chunk['file_path']}
{text}"""


def summarize_single(
    chunk: dict,
    model: str = DEFAULT_MODEL,
) -> str:
    """Summarize a single chunk."""
    from ollama import chat

    prompt = build_single_prompt(chunk)

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=SingleSummary.model_json_schema(),
        think=False,
    )

    if response.message.content is None:
        raise ValueError("No content returned from model response.")
    result = SingleSummary.model_validate_json(response.message.content)
    return result.summary


def summarize_batch(
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """Summarize a batch of chunks, returning summaries in order."""
    from ollama import chat

    prompt = build_batch_prompt(chunks)

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=BatchSummaries.model_json_schema(),
        think=False,
    )

    if response.message.content is None:
        raise ValueError("No content returned from model response.")
    result = BatchSummaries.model_validate_json(response.message.content)
    return [item.s for item in result.summaries]


def summarize_all(
    conn: sqlite3.Connection,
    model: str = DEFAULT_MODEL,
    config: RuntimeConfig | None = None,
    *,
    batch_size: int | None = None,
    verbose: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Generate summaries for all chunks using LLM calls."""
    cfg = config or get_global_config()

    # Use config batch size if not explicitly provided
    if batch_size is None:
        batch_size = cfg.summarize_batch_size

    cursor = conn.execute(
        """
        SELECT c.id, c.file_id, c.chunk_index, c.text, f.path
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        ORDER BY c.file_id, c.chunk_index
        """
    )

    all_chunks = cursor.fetchall()
    total = len(all_chunks)

    if not all_chunks:
        if verbose:
            print("No chunks to summarize")
        return 0

    mode = "CPU (parallel)" if cfg.force_cpu else "GPU"
    if verbose:
        print(f"Summarizing {total} chunks with {model} [{mode}]...")

    if progress_callback:
        progress_callback(0, total)

    # CPU mode: parallel single-item processing
    if cfg.force_cpu:
        return _summarize_parallel(
            conn, all_chunks, model, cfg, verbose, progress_callback
        )

    # GPU mode: batch processing
    return _summarize_batched(
        conn, all_chunks, model, batch_size, verbose, progress_callback
    )


def _summarize_parallel(
    conn: sqlite3.Connection,
    all_chunks: list,
    model: str,
    config: RuntimeConfig,
    verbose: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> int:
    """Summarize chunks in parallel (CPU mode)."""
    total = len(all_chunks)
    summarized = 0
    errors = 0

    # Prepare work items
    work_items = []
    for row in all_chunks:
        work_items.append({
            "id": row["id"],
            "file_path": row["path"],
            "text": row["text"],
        })

    def process_single(item: dict) -> tuple[int, str | None, str | None]:
        """Process a single chunk, return (id, summary, error)."""
        try:
            summary = summarize_single(
                {"file_path": item["file_path"], "text": item["text"]},
                model=model
            )
            return item["id"], summary, None
        except Exception as e:
            return item["id"], None, str(e)

    # Process in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures = {executor.submit(process_single, item): item for item in work_items}

        for future in as_completed(futures):
            chunk_id, summary, error = future.result()
            completed += 1

            if summary:
                conn.execute(
                    "UPDATE chunks SET summary = ? WHERE id = ?",
                    (summary, chunk_id),
                )
                summarized += 1
            else:
                errors += 1
                if verbose:
                    print(f"  [{completed}/{total}] ERROR: {error}")

            # Fire progress after every single item
            if progress_callback:
                progress_callback(completed, total)

            # Commit periodically (every 10) to avoid too many disk writes
            if completed % 10 == 0:
                conn.commit()

    conn.commit()

    set_metadata(conn, "summary_count", str(summarized))
    set_metadata(conn, "summary_model", model)
    set_metadata(conn, "stage", "summarized")

    if verbose:
        print(f"\nDone: {summarized}/{total} summarized")
        if errors:
            print(f"Errors: {errors}")

    return summarized


def _summarize_batched(
    conn: sqlite3.Connection,
    all_chunks: list,
    model: str,
    batch_size: int,
    verbose: bool,
    progress_callback: Callable[[int, int], None] | None,
) -> int:
    """Summarize chunks in batches (GPU mode)."""
    total = len(all_chunks)
    summarized = 0
    errors = 0

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_rows = all_chunks[batch_start:batch_end]

        # Build batch with chunk IDs for later update
        batch_chunks = []
        chunk_ids = []

        for row in batch_rows:
            batch_chunks.append({
                "file_path": row["path"],
                "text": row["text"],
            })
            chunk_ids.append(row["id"])

        try:
            summaries = summarize_batch(batch_chunks, model=model)

            # Match summaries to chunks by position
            for i, summary in enumerate(summaries):
                if i < len(chunk_ids):
                    conn.execute(
                        "UPDATE chunks SET summary = ? WHERE id = ?",
                        (summary, chunk_ids[i]),
                    )
                    summarized += 1

            conn.commit()

            if verbose:
                print(f"  [{batch_end}/{total}] +{len(summaries)} summaries")

        except Exception as e:
            errors += len(batch_chunks)
            if verbose:
                print(f"  [{batch_end}/{total}] ERROR: {e}")

        if progress_callback:
            progress_callback(batch_end, total)

    set_metadata(conn, "summary_count", str(summarized))
    set_metadata(conn, "summary_model", model)
    set_metadata(conn, "stage", "summarized")

    if verbose:
        print(f"\nDone: {summarized}/{total} summarized")
        if errors:
            print(f"Errors: {errors}")

    return summarized
