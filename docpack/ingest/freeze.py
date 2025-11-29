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
from docpack.extract import EXTRACTABLE_EXTENSIONS, can_extract, extract_document
from docpack.summarize import summarize_all
from docpack.storage import init_db, insert_file, insert_image, set_metadata
from docpack.runtime import RuntimeConfig, get_global_config
from docpack.vision import vision_all, DEFAULT_VISION_MODEL

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
    skip_vision: bool = False,
    embedding_model: str | None = None,
    summarize_model: str | None = None,
    vision_model: str | None = None,
    config: RuntimeConfig | None = None,
) -> Path:
    """
    Freeze a target into a .docpack file.

    Ingests all files from the target source (directory, zip, URL)
    and stores them in a SQLite database. By default, text files are
    also chunked, embedded, and summarized for semantic search.

    PDF, DOCX, and PPTX files are automatically extracted, with text
    becoming searchable and images analyzed by a vision model.

    Args:
        target: Path or URL to ingest
        output: Output .docpack path (default: derived from target)
        use_temp: If True, create in system temp dir (auto-cleanup)
        verbose: Print progress information
        skip_chunking: If True, skip the chunking step (raw ingestion only)
        skip_embedding: If True, skip the embedding step
        skip_summarize: If True, skip LLM summarization (requires Ollama)
        skip_vision: If True, skip vision model analysis of images
        embedding_model: HuggingFace model to use for embeddings
        summarize_model: LLM model for summarization (default: qwen3:4b)
        vision_model: Vision model for image analysis (default: qwen3-vl:2b)

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
            image_count = 0

            for vfile in source.walk():
                # Read content
                content_bytes = vfile.read_bytes()
                is_binary = vfile.is_binary()

                # Check if this is an extractable document (PDF, DOCX, PPTX)
                if can_extract(vfile.extension or ""):
                    try:
                        extracted = extract_document(content_bytes, vfile.extension or "")

                        # Insert file with extracted text
                        file_id = insert_file(
                            conn,
                            path=vfile.path,
                            size_bytes=vfile.size,
                            sha256_hash=vfile.sha256(),
                            content=extracted.text,
                            is_binary=False,  # Text was extracted
                            extension=vfile.extension,
                        )

                        # Store extracted images
                        for img in extracted.images:
                            insert_image(
                                conn,
                                file_id=file_id,
                                image_index=img.image_index,
                                format=img.format,
                                width=img.width,
                                height=img.height,
                                image_data=img.data,
                                page_number=img.page_number,
                                context=img.context,
                            )
                            image_count += 1

                        file_count += 1
                        total_bytes += vfile.size

                        if verbose:
                            img_info = f", {len(extracted.images)} images" if extracted.images else ""
                            page_info = f", {extracted.page_count} pages" if extracted.page_count else ""
                            print(f"  [extracted] {vfile.path} ({vfile.size} bytes{page_info}{img_info})")

                    except Exception as e:
                        # Fall back to binary storage if extraction fails
                        if verbose:
                            print(f"  [warn] Failed to extract {vfile.path}: {e}")

                        insert_file(
                            conn,
                            path=vfile.path,
                            size_bytes=vfile.size,
                            sha256_hash=vfile.sha256(),
                            content=None,
                            is_binary=True,
                            extension=vfile.extension,
                        )
                        file_count += 1
                        total_bytes += vfile.size

                else:
                    # Regular file handling
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

            # Store metadata
            set_metadata(conn, "format_version", FORMAT_VERSION)
            set_metadata(conn, "docpack_version", __version__)
            set_metadata(conn, "created_at", datetime.now(timezone.utc).isoformat())
            set_metadata(conn, "source", target)
            set_metadata(conn, "file_count", str(file_count))
            set_metadata(conn, "total_bytes", str(total_bytes))
            if image_count > 0:
                set_metadata(conn, "image_count", str(image_count))

            # Store config flags
            if skip_chunking:
                set_metadata(conn, "config.skip_chunking", "true")
            if skip_embedding:
                set_metadata(conn, "config.skip_embedding", "true")
            if skip_summarize:
                set_metadata(conn, "config.skip_summarize", "true")
            if skip_vision:
                set_metadata(conn, "config.skip_vision", "true")
            if embedding_model:
                set_metadata(conn, "config.embedding_model", embedding_model)
            if summarize_model:
                set_metadata(conn, "config.summarize_model", summarize_model)
            if vision_model:
                set_metadata(conn, "config.vision_model", vision_model)

            if verbose:
                img_info = f", {image_count} images extracted" if image_count > 0 else ""
                print(f"\nFroze {file_count} files ({total_bytes:,} bytes{img_info})")

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
                    cfg = config or get_global_config()
                    embed_kwargs: dict = {"verbose": verbose, "config": cfg}
                    if embedding_model:
                        embed_kwargs["model"] = embedding_model
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

                    # Vision analysis of images
                    if not skip_vision and image_count > 0:
                        if verbose:
                            print("\nAnalyzing images with vision model...")
                        cfg = config or get_global_config()
                        vision_kwargs: dict = {"verbose": verbose, "config": cfg}
                        if vision_model:
                            vision_kwargs["model"] = vision_model
                        try:
                            analyzed = vision_all(conn, **vision_kwargs)
                            # Re-embed the new image description chunks
                            if analyzed > 0 and not skip_embedding:
                                if verbose:
                                    print("\nEmbedding image descriptions...")
                                embed_kwargs_vision: dict = {"verbose": verbose, "config": cfg}
                                if embedding_model:
                                    embed_kwargs_vision["model"] = embedding_model
                                embed_all(conn, **embed_kwargs_vision)
                        except Exception as e:
                            if verbose:
                                print(f"Warning: Vision analysis failed: {e}")
                                print("Continuing without image descriptions (Ollama may not be running)")
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
