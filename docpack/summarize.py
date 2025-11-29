"""
LLM-powered summarization for chunks.

Uses Ollama with structured output for reliable batch summarization.
"""

from __future__ import annotations

import sqlite3
from typing import Callable

from pydantic import BaseModel

from docpack.storage import set_metadata


DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_BATCH_SIZE = 8


class ChunkSummary(BaseModel):
    """Summary for a single chunk."""
    s: str  # summary only - we use position for matching


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
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Generate summaries for all chunks using batch LLM calls."""
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

    if verbose:
        print(f"Summarizing {total} chunks with {model} (batch={batch_size})...")

    if progress_callback:
        progress_callback(0, total)

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
