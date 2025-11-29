"""
Chunking module for splitting file content into semantic chunks.

This module provides functions to split text content into paragraph-based
chunks suitable for embedding and semantic search.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from docpack.storage import get_all_files, get_file, insert_chunk, set_metadata


@dataclass
class Chunk:
    """Represents a chunk of text with position information."""

    text: str
    index: int
    start_char: int
    end_char: int

    @property
    def char_count(self) -> int:
        """Number of characters in the chunk."""
        return len(self.text)

    def estimate_tokens(self) -> int:
        """
        Estimate token count using simple heuristic.

        Roughly 4 characters per token for English text.
        This is a fast approximation - actual tokenization would
        require a tokenizer library.
        """
        return max(1, len(self.text) // 4)


def split_into_chunks(
    text: str,
    *,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 2000,
) -> list[Chunk]:
    """
    Split text into paragraph-based chunks.

    The algorithm:
    1. Split on paragraph boundaries (double newlines)
    2. Merge small paragraphs together until min_chunk_chars
    3. Split large paragraphs at sentence boundaries if > max_chunk_chars

    Args:
        text: The text content to split
        min_chunk_chars: Minimum characters per chunk (merge smaller ones)
        max_chunk_chars: Maximum characters per chunk (split larger ones)

    Returns:
        List of Chunk objects with text and position information
    """
    if not text or not text.strip():
        return []

    # Split on paragraph boundaries (2+ newlines)
    paragraphs = re.split(r"\n\s*\n", text)

    # Filter empty paragraphs and track positions
    para_spans: list[tuple[str, int, int]] = []
    pos = 0
    for para in paragraphs:
        # Find actual position in original text
        start = text.find(para, pos)
        if start == -1:
            start = pos
        end = start + len(para)
        if para.strip():
            para_spans.append((para.strip(), start, end))
        pos = end

    if not para_spans:
        return []

    chunks: list[Chunk] = []
    current_text = ""
    current_start = para_spans[0][1]
    current_end = para_spans[0][1]

    for para_text, para_start, para_end in para_spans:
        # Would adding this paragraph exceed max?
        combined = f"{current_text}\n\n{para_text}" if current_text else para_text

        if len(combined) > max_chunk_chars and current_text:
            # Finalize current chunk
            chunks.append(
                Chunk(
                    text=current_text,
                    index=len(chunks),
                    start_char=current_start,
                    end_char=current_end,
                )
            )
            current_text = para_text
            current_start = para_start
            current_end = para_end
        else:
            current_text = combined
            if not current_text.strip():
                current_start = para_start
            current_end = para_end

    # Handle remaining text
    if current_text.strip():
        # If still too large, split on sentences
        if len(current_text) > max_chunk_chars:
            sentence_chunks = _split_large_chunk(
                current_text, current_start, len(chunks), max_chunk_chars
            )
            chunks.extend(sentence_chunks)
        else:
            chunks.append(
                Chunk(
                    text=current_text,
                    index=len(chunks),
                    start_char=current_start,
                    end_char=current_end,
                )
            )

    # Merge small chunks
    chunks = _merge_small_chunks(chunks, min_chunk_chars, max_chunk_chars)

    # Re-index after merging
    for i, chunk in enumerate(chunks):
        chunk.index = i

    return chunks


def _split_large_chunk(
    text: str,
    start_offset: int,
    start_index: int,
    max_chars: int,
) -> list[Chunk]:
    """Split a large chunk on sentence boundaries."""
    # Sentence boundaries: . ! ? followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[Chunk] = []
    current_text = ""
    current_start = start_offset
    pos = start_offset

    for sentence in sentences:
        if not sentence.strip():
            continue

        combined = f"{current_text} {sentence}" if current_text else sentence

        if len(combined) > max_chars and current_text:
            chunks.append(
                Chunk(
                    text=current_text.strip(),
                    index=start_index + len(chunks),
                    start_char=current_start,
                    end_char=pos,
                )
            )
            current_text = sentence
            current_start = pos
        else:
            current_text = combined

        pos += len(sentence) + 1  # +1 for space

    if current_text.strip():
        chunks.append(
            Chunk(
                text=current_text.strip(),
                index=start_index + len(chunks),
                start_char=current_start,
                end_char=start_offset + len(text),
            )
        )

    return chunks


def _merge_small_chunks(
    chunks: list[Chunk],
    min_chars: int,
    max_chars: int,
) -> list[Chunk]:
    """Merge chunks that are too small."""
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    current: Chunk | None = None

    for chunk in chunks:
        if current is None:
            current = chunk
            continue

        # Can we merge with current?
        combined_len = len(current.text) + len(chunk.text) + 2  # +2 for \n\n
        if len(current.text) < min_chars and combined_len <= max_chars:
            # Merge
            current = Chunk(
                text=f"{current.text}\n\n{chunk.text}",
                index=current.index,
                start_char=current.start_char,
                end_char=chunk.end_char,
            )
        else:
            merged.append(current)
            current = chunk

    if current is not None:
        merged.append(current)

    return merged


def chunk_file(
    conn: sqlite3.Connection,
    file_id: int,
    *,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 2000,
) -> int:
    """
    Chunk a single file and store chunks in database.

    Args:
        conn: Database connection
        file_id: ID of the file to chunk
        min_chunk_chars: Minimum characters per chunk
        max_chunk_chars: Maximum characters per chunk

    Returns:
        Number of chunks created

    Raises:
        ValueError: If file not found or is binary
    """
    file_record = get_file(conn, file_id)
    if file_record is None:
        raise ValueError(f"File with id {file_id} not found")

    if file_record["is_binary"]:
        return 0  # Skip binary files

    content = file_record.get("content")
    if not content:
        return 0

    chunks = split_into_chunks(
        content,
        min_chunk_chars=min_chunk_chars,
        max_chunk_chars=max_chunk_chars,
    )

    for chunk in chunks:
        insert_chunk(
            conn,
            file_id=file_id,
            order_index=chunk.index,
            text=chunk.text,
            token_count=chunk.estimate_tokens(),
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )

    return len(chunks)


def chunk_all(
    conn: sqlite3.Connection,
    *,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 2000,
    verbose: bool = False,
) -> int:
    """
    Chunk all text files in the database.

    Args:
        conn: Database connection
        min_chunk_chars: Minimum characters per chunk
        max_chunk_chars: Maximum characters per chunk
        verbose: Print progress information

    Returns:
        Total number of chunks created
    """
    files = get_all_files(conn)
    total_chunks = 0

    for file_record in files:
        if file_record["is_binary"]:
            continue

        file_id = file_record["id"]
        chunk_count = chunk_file(
            conn,
            file_id,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
        )

        if verbose and chunk_count > 0:
            print(f"  {file_record['path']}: {chunk_count} chunks")

        total_chunks += chunk_count

    # Update metadata
    set_metadata(conn, "chunk_count", str(total_chunks))
    set_metadata(conn, "stage", "chunked")

    if verbose:
        print(f"\nTotal: {total_chunks} chunks")

    return total_chunks
