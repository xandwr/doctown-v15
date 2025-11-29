"""
Freeze command implementation.

Ingests files from various sources and stores them in a .docpack file.
This is the first stage of the pipeline - raw file ingestion without
chunking or embedding.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from docpack import FORMAT_VERSION, __version__
from docpack.chunk import chunk_all
from docpack.embed import embed_all
from docpack.summarize import summarize_all
from docpack.storage import init_db, insert_file, set_metadata

from .sources import DirectorySource, URLSource, ZipSource
from .vfs import VirtualFS

if TYPE_CHECKING:
    import sqlite3


# -----------------------------------------------------------------------------
# Source detection
# -----------------------------------------------------------------------------


def detect_source(target: str) -> VirtualFS:
    """
    Detect and create appropriate source for target.

    Handles:
    - Local directories
    - Local zip files
    - URLs (http://, https://)

    Args:
        target: Path or URL to ingest

    Returns:
        Appropriate VirtualFS implementation

    Raises:
        ValueError: If target type cannot be determined
        FileNotFoundError: If local path doesn't exist
    """
    # URL
    if target.startswith(("http://", "https://")):
        return URLSource(target)

    # Local path
    path = Path(target)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    if path.is_dir():
        return DirectorySource(path)

    if path.is_file():
        # Check if zip
        if path.suffix.lower() == ".zip":
            return ZipSource(path)

        # Try to detect zip by magic bytes
        try:
            with open(path, "rb") as f:
                if f.read(2) == b"PK":
                    return ZipSource(path)
        except OSError:
            pass

        raise ValueError(f"Unsupported file type: {target}")

    raise ValueError(f"Cannot determine source type: {target}")


# -----------------------------------------------------------------------------
# Freeze
# -----------------------------------------------------------------------------


def freeze(
    target: str,
    output: str | None = None,
    *,
    use_temp: bool = False,
    verbose: bool = False,
    skip_chunking: bool = False,
    skip_embedding: bool = False,
    skip_summarize: bool = False,
    embedding_model: str | None = None,
    summarize_model: str | None = None,
) -> Path:
    """
    Freeze a target into a .docpack file.

    Ingests all files from the target source (directory, zip, URL)
    and stores them in a SQLite database. By default, text files are
    also chunked, embedded, and summarized for semantic search.

    Args:
        target: Path or URL to ingest
        output: Output .docpack path (default: derived from target)
        use_temp: If True, create in system temp dir (auto-cleanup)
        verbose: Print progress information
        skip_chunking: If True, skip the chunking step (raw ingestion only)
        skip_embedding: If True, skip the embedding step
        skip_summarize: If True, skip LLM summarization (requires Ollama)
        embedding_model: HuggingFace model to use for embeddings
        summarize_model: LLM model for summarization (default: qwen3:4b)

    Returns:
        Path to created .docpack file

    Example:
        # Freeze a directory
        path = freeze("./my-project", "project.docpack")

        # Freeze to temp (for one-shot operations)
        path = freeze("./my-project", use_temp=True)

        # Freeze a URL
        path = freeze("https://github.com/user/repo/archive/main.zip")

        # Freeze with LLM summaries
        path = freeze("./my-project", summarize_model="qwen3:4b")
    """
    # Determine output path
    if output:
        output_path = Path(output)
    elif use_temp:
        # Create in temp directory
        fd, temp_path = tempfile.mkstemp(suffix=".docpack")
        import os

        os.close(fd)
        output_path = Path(temp_path)
    else:
        # Derive from target name
        if target.startswith(("http://", "https://")):
            from urllib.parse import urlparse

            parsed = urlparse(target)
            name = parsed.path.rsplit("/", 1)[-1] or "download"
            name = name.rsplit(".", 1)[0]  # Remove extension
        else:
            name = Path(target).stem

        output_path = Path(f"{name}.docpack")

    # Initialize database
    conn = init_db(str(output_path))

    try:
        # Get source and ingest
        source = detect_source(target)

        with source:
            file_count = 0
            total_bytes = 0

            for vfile in source.walk():
                # Read content
                content_bytes = vfile.read_bytes()
                is_binary = vfile.is_binary()

                # Decode text content
                text_content: str | None = None
                if not is_binary:
                    text_content = content_bytes.decode("utf-8", errors="replace")

                # Insert into database
                insert_file(
                    conn,
                    path=vfile.path,
                    size_bytes=vfile.size,
                    sha256_hash=vfile.sha256(),
                    content=text_content,
                    is_binary=is_binary,
                    extension=vfile.extension,
                )

                file_count += 1
                total_bytes += vfile.size

                if verbose:
                    status = "binary" if is_binary else "text"
                    print(f"  [{status}] {vfile.path} ({vfile.size} bytes)")

            # Store metadata
            set_metadata(conn, "format_version", FORMAT_VERSION)
            set_metadata(conn, "docpack_version", __version__)
            set_metadata(conn, "created_at", datetime.now(timezone.utc).isoformat())
            set_metadata(conn, "source", target)
            set_metadata(conn, "file_count", str(file_count))
            set_metadata(conn, "total_bytes", str(total_bytes))

            # Store config flags
            if skip_chunking:
                set_metadata(conn, "config.skip_chunking", "true")
            if skip_embedding:
                set_metadata(conn, "config.skip_embedding", "true")
            if skip_summarize:
                set_metadata(conn, "config.skip_summarize", "true")
            if embedding_model:
                set_metadata(conn, "config.embedding_model", embedding_model)
            if summarize_model:
                set_metadata(conn, "config.summarize_model", summarize_model)

            if verbose:
                print(f"\nFroze {file_count} files ({total_bytes:,} bytes)")

            # Chunk text files
            if not skip_chunking:
                if verbose:
                    print("\nChunking text files...")
                chunk_count = chunk_all(conn, verbose=verbose)
                if verbose:
                    print(f"Created {chunk_count} chunks")

                # Embed chunks
                if not skip_embedding and chunk_count > 0:
                    if verbose:
                        print("\nEmbedding chunks...")
                    embed_kwargs = {"verbose": verbose}
                    if embedding_model:
                        embed_kwargs["model_name"] = embedding_model
                    embed_all(conn, **embed_kwargs)

                    # Summarize chunks with LLM
                    if not skip_summarize:
                        if verbose:
                            print("\nSummarizing chunks with LLM...")
                        summarize_kwargs = {"verbose": verbose}
                        if summarize_model:
                            summarize_kwargs["model"] = summarize_model
                        try:
                            summarize_all(conn, **summarize_kwargs)
                        except Exception as e:
                            if verbose:
                                print(f"Warning: Summarization failed: {e}")
                                print("Continuing without summaries (Ollama may not be running)")
            else:
                set_metadata(conn, "stage", "frozen")  # No chunks yet

            if verbose:
                print(f"\nOutput: {output_path}")

    finally:
        conn.close()

    return output_path


def freeze_to_connection(
    target: str,
    conn: "sqlite3.Connection",
    *,
    verbose: bool = False,
) -> int:
    """
    Freeze a target into an existing database connection.

    Useful for programmatic use where you want control over the connection.

    Args:
        target: Path or URL to ingest
        conn: Open database connection
        verbose: Print progress information

    Returns:
        Number of files ingested
    """
    source = detect_source(target)

    with source:
        file_count = 0
        total_bytes = 0

        for vfile in source.walk():
            content_bytes = vfile.read_bytes()
            is_binary = vfile.is_binary()

            text_content: str | None = None
            if not is_binary:
                text_content = content_bytes.decode("utf-8", errors="replace")

            insert_file(
                conn,
                path=vfile.path,
                size_bytes=vfile.size,
                sha256_hash=vfile.sha256(),
                content=text_content,
                is_binary=is_binary,
                extension=vfile.extension,
            )

            file_count += 1
            total_bytes += vfile.size

            if verbose:
                status = "binary" if is_binary else "text"
                print(f"  [{status}] {vfile.path} ({vfile.size} bytes)")

        set_metadata(conn, "format_version", FORMAT_VERSION)
        set_metadata(conn, "docpack_version", __version__)
        set_metadata(conn, "created_at", datetime.now(timezone.utc).isoformat())
        set_metadata(conn, "source", target)
        set_metadata(conn, "file_count", str(file_count))
        set_metadata(conn, "total_bytes", str(total_bytes))
        set_metadata(conn, "stage", "frozen")

    return file_count
