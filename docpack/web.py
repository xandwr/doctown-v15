"""
Doctown web server.

A FastAPI server that serves the Svelte frontend and provides
API endpoints for semantic search and document processing.

Usage:
    docpack web                    # Start server on localhost:8000
    docpack web -p 3000            # Custom port
    docpack web --static-dir ./build  # Custom static directory
"""

from __future__ import annotations

import asyncio
import signal
import sqlite3
import sys
import tempfile
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# =============================================================================
# State Management
# =============================================================================


@dataclass
class AppState:
    """Global application state."""

    stage: str = "idle"  # idle, freezing, chunking, embedding, summarizing, ready, error
    docpack_path: Path | None = None
    error: str | None = None

    # Progress tracking
    progress_phase: str = ""
    progress_current: int = 0
    progress_total: int = 0

    # Stats cache
    stats: dict[str, int] = field(default_factory=dict)

    # SSE subscribers
    subscribers: list[asyncio.Queue] = field(default_factory=list)

    # Timing tracking (timestamps in seconds since epoch)
    pipeline_start_time: float | None = None
    phase_start_time: float | None = None
    phase_times: dict[str, float] = field(default_factory=dict)  # phase -> duration in seconds

    # File staging for uploads
    stage_dir: Path | None = None
    staged_files: dict[str, int] = field(default_factory=dict)  # relative_path -> size_bytes

    def get_connection(self) -> sqlite3.Connection | None:
        """Get a new database connection (thread-safe)."""
        if self.docpack_path is None:
            return None
        conn = sqlite3.connect(str(self.docpack_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def clear_staging(self):
        """Clear all staged files and remove temp directory."""
        import shutil
        if self.stage_dir and self.stage_dir.exists():
            shutil.rmtree(self.stage_dir, ignore_errors=True)
        self.stage_dir = None
        self.staged_files = {}


# Global state instance
state = AppState()


async def broadcast_state():
    """Send current state to all SSE subscribers."""
    import time

    # Calculate current phase elapsed time
    phase_elapsed = None
    if state.phase_start_time is not None:
        phase_elapsed = time.time() - state.phase_start_time

    # Calculate total pipeline elapsed time
    pipeline_elapsed = None
    if state.pipeline_start_time is not None:
        pipeline_elapsed = time.time() - state.pipeline_start_time

    data = {
        "stage": state.stage,
        "progress": {
            "phase": state.progress_phase,
            "current": state.progress_current,
            "total": state.progress_total,
        },
        "stats": state.stats,
        "error": state.error,
        "timing": {
            "phase_elapsed": phase_elapsed,
            "pipeline_elapsed": pipeline_elapsed,
            "phase_times": state.phase_times,
        },
    }
    for queue in state.subscribers:
        await queue.put(data)


def update_progress(phase: str, current: int, total: int):
    """Update progress and schedule broadcast."""
    state.progress_phase = phase
    state.progress_current = current
    state.progress_total = total


# =============================================================================
# Pydantic Models
# =============================================================================


class ProcessRequest(BaseModel):
    """Request to process a directory or staged files."""

    path: str | None = None
    use_staged: bool = False


class StagedFileModel(BaseModel):
    """A staged file."""

    name: str
    size: int


class StagedFilesResponse(BaseModel):
    """Response with staged files list."""

    files: list[StagedFileModel]
    total_size: int
    file_count: int


class SearchRequest(BaseModel):
    """Semantic search request."""

    query: str
    k: int = 10


class SearchResult(BaseModel):
    """A single search result."""

    file_path: str
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    """Search response."""

    results: list[SearchResult]


class AskRequest(BaseModel):
    """Ask a question with citation-backed answer."""

    query: str


class CitationModel(BaseModel):
    """A citation reference."""

    id: int
    file_path: str
    chunk_index: int
    quote: str
    start_char: int | None = None
    end_char: int | None = None


class AskResponse(BaseModel):
    """Answer with citations."""

    answer: str
    citations: list[CitationModel]
    confidence: str
    sources_retrieved: int
    sources_used: int


# =============================================================================
# Pipeline Runner
# =============================================================================


def _run_pipeline_sync(target_path: str, output_path: Path, progress_update_fn):
    """
    Synchronous pipeline runner (runs in a thread).

    This does all the heavy lifting in a single thread to avoid SQLite threading issues.
    """
    from docpack.chunk import chunk_all
    from docpack.embed import embed_all
    from docpack.extract import can_extract, extract_document
    from docpack.ingest.freeze import detect_source
    from docpack.storage import init_db, insert_file, insert_image, set_metadata, get_stats
    from docpack.runtime import get_global_config
    from docpack import FORMAT_VERSION, __version__
    from datetime import datetime, timezone

    config = get_global_config()

    # Initialize database with check_same_thread=False for safety
    conn = sqlite3.connect(str(output_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Run the schema initialization
    from docpack.storage.schema import SCHEMA_SQL
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    # Detect source and count files first
    source = detect_source(target_path)

    with source:
        files_list = list(source.walk())
        total_files = len(files_list)

    progress_update_fn("freezing", 0, total_files)

    # Re-detect source since we consumed the iterator
    source = detect_source(target_path)

    file_count = 0
    total_bytes = 0
    image_count = 0

    with source:
        for vfile in source.walk():
            content_bytes = vfile.read_bytes()
            is_binary = vfile.is_binary()

            # Check if this is an extractable document (PDF, DOCX, PPTX) or image
            if can_extract(vfile.extension or ""):
                try:
                    extracted = extract_document(content_bytes, vfile.extension or "", vfile.path)

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

                except Exception as e:
                    # Fall back to binary storage if extraction fails
                    print(f"[warn] Failed to extract {vfile.path}: {e}")
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

            if file_count % 10 == 0 or file_count == total_files:
                progress_update_fn("freezing", file_count, total_files)

    # Store metadata
    set_metadata(conn, "format_version", FORMAT_VERSION)
    set_metadata(conn, "docpack_version", __version__)
    set_metadata(conn, "created_at", datetime.now(timezone.utc).isoformat())
    set_metadata(conn, "source", target_path)
    set_metadata(conn, "file_count", str(file_count))
    set_metadata(conn, "total_bytes", str(total_bytes))
    if image_count > 0:
        set_metadata(conn, "image_count", str(image_count))

    stats = get_stats(conn)
    progress_update_fn("freezing_done", file_count, total_files, stats)

    # Chunking phase
    progress_update_fn("chunking", 0, file_count)
    chunk_count = chunk_all(conn, verbose=False)
    stats = get_stats(conn)
    progress_update_fn("chunking_done", chunk_count, chunk_count, stats)

    # Embedding phase
    if chunk_count > 0:
        progress_update_fn("embedding", 0, chunk_count)

        def embed_progress(current: int, total: int):
            progress_update_fn("embedding", current, total)

        embed_all(conn, config=config, verbose=False, progress_callback=embed_progress)
        stats = get_stats(conn)
        progress_update_fn("embedding_done", chunk_count, chunk_count, stats)

    # Summarization phase - generate chunk summaries for better answers
    if chunk_count > 0:
        from docpack.summarize import summarize_all

        progress_update_fn("summarizing", 0, chunk_count)

        def summarize_progress(current: int, total: int):
            progress_update_fn("summarizing", current, total)

        try:
            summarize_all(conn, config=config, verbose=False, progress_callback=summarize_progress)
        except Exception as e:
            # Summarization is optional - continue if it fails (e.g., no Ollama)
            print(f"Summarization skipped: {e}")

        stats = get_stats(conn)
        progress_update_fn("summarizing_done", chunk_count, chunk_count, stats)

    # Vision analysis phase - analyze images from PDFs/DOCX/PPTX
    if image_count > 0:
        from docpack.vision import vision_all

        progress_update_fn("vision", 0, image_count)

        def vision_progress(current: int, total: int):
            progress_update_fn("vision", current, total)

        try:
            analyzed = vision_all(conn, config=config, verbose=False, progress_callback=vision_progress)
            # Re-embed the new image description chunks
            if analyzed > 0:
                embed_all(conn, config=config, verbose=False)
        except Exception as e:
            # Vision is optional - continue if it fails (e.g., no Ollama)
            print(f"Vision analysis skipped: {e}")

        stats = get_stats(conn)
        progress_update_fn("vision_done", image_count, image_count, stats)

    conn.close()
    return stats


async def run_pipeline(target_path: str):
    """
    Run the freeze -> chunk -> embed pipeline with progress updates.

    This wraps the existing docpack functions and broadcasts progress via SSE.
    """
    import os
    import time

    try:
        # Initialize timing
        state.pipeline_start_time = time.time()
        state.phase_start_time = time.time()
        state.phase_times = {}

        state.stage = "freezing"
        state.error = None
        await broadcast_state()

        # Create temp docpack file
        fd, temp_path = tempfile.mkstemp(suffix=".docpack")
        os.close(fd)
        output_path = Path(temp_path)
        state.docpack_path = output_path

        # Progress callback that updates state and queues broadcast
        loop = asyncio.get_event_loop()

        def progress_update_fn(phase: str, current: int, total: int, stats: dict | None = None):
            if phase.endswith("_done"):
                base_phase = phase.replace("_done", "")
                # Record the completed phase's duration
                if state.phase_start_time is not None:
                    state.phase_times[base_phase] = time.time() - state.phase_start_time

                # Transition to next phase and reset phase timer
                # Note: vision phase only runs if there are images, so summarizing
                # may go directly to ready if no vision phase follows
                if base_phase == "freezing":
                    state.stage = "chunking"
                    state.phase_start_time = time.time()
                elif base_phase == "chunking":
                    state.stage = "embedding"
                    state.phase_start_time = time.time()
                elif base_phase == "embedding":
                    state.stage = "summarizing"
                    state.phase_start_time = time.time()
                elif base_phase == "summarizing":
                    # Vision phase will set its own stage if it runs
                    # Otherwise we'll go to ready after the pipeline finishes
                    state.phase_start_time = time.time()
                elif base_phase == "vision":
                    state.stage = "ready"
                    state.phase_start_time = None  # Pipeline complete
            else:
                state.stage = phase
                state.progress_phase = phase
                state.progress_current = current
                state.progress_total = total

            if stats:
                state.stats = stats

            # Schedule broadcast from the thread
            asyncio.run_coroutine_threadsafe(broadcast_state(), loop)

        # Run the entire pipeline in a thread
        final_stats = await asyncio.to_thread(
            _run_pipeline_sync, target_path, output_path, progress_update_fn
        )

        # Save timing data to database
        if state.phase_times:
            import json
            conn = state.get_connection()
            if conn:
                from docpack.storage import set_metadata
                set_metadata(conn, "phase_times", json.dumps(state.phase_times))
                total_time = sum(state.phase_times.values())
                set_metadata(conn, "total_pipeline_time", str(total_time))
                conn.close()

        # Done!
        state.stage = "ready"
        state.stats = final_stats
        update_progress("", 0, 0)
        await broadcast_state()

    except Exception as e:
        state.stage = "error"
        state.error = str(e)
        await broadcast_state()
        raise


# =============================================================================
# FastAPI App
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """App lifespan handler for startup/shutdown."""
    print("Doctown server starting...")
    yield
    print("Shutting down...")
    # Clean up temp file if it exists
    if state.docpack_path and state.docpack_path.exists():
        try:
            state.docpack_path.unlink()
        except OSError:
            pass


app = FastAPI(
    title="Doctown",
    description="Semantic search for your data",
    lifespan=lifespan,
)

# Add CORS middleware for mobile/cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Routes
# =============================================================================


@app.get("/api/status")
async def get_status():
    """Get current server status."""
    return {
        "stage": state.stage,
        "stats": state.stats,
        "error": state.error,
    }


# =============================================================================
# File Staging API
# =============================================================================


@app.post("/api/upload", response_model=StagedFilesResponse)
async def upload_files(files: list[UploadFile]):
    """
    Upload files to staging area for processing.

    Accepts multiple files via multipart/form-data.
    Files are stored temporarily until processing begins.
    """
    print(f"[upload] Received upload request with {len(files)} file(s)")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Create staging directory if needed
    if state.stage_dir is None:
        state.stage_dir = Path(tempfile.mkdtemp(prefix="doctown-stage-"))

    for file in files:
        if not file.filename:
            continue

        # Use the filename directly from the File object
        relative_path = file.filename.lstrip("/")

        # Create parent directories if needed
        file_path = state.stage_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file content
        print(f"[upload] Reading file: {relative_path}")
        content = await file.read()
        print(f"[upload] Writing file: {relative_path} ({len(content)} bytes)")
        file_path.write_bytes(content)

        # Track in state
        state.staged_files[relative_path] = len(content)

    # Build response
    staged_list = [
        StagedFileModel(name=name, size=size)
        for name, size in sorted(state.staged_files.items())
    ]

    print(f"[upload] Upload complete: {len(state.staged_files)} files staged")
    return StagedFilesResponse(
        files=staged_list,
        total_size=sum(state.staged_files.values()),
        file_count=len(state.staged_files),
    )


@app.get("/api/staged", response_model=StagedFilesResponse)
async def get_staged_files():
    """Get list of currently staged files."""
    staged_list = [
        StagedFileModel(name=name, size=size)
        for name, size in sorted(state.staged_files.items())
    ]

    return StagedFilesResponse(
        files=staged_list,
        total_size=sum(state.staged_files.values()),
        file_count=len(state.staged_files),
    )


@app.delete("/api/staged/{file_path:path}")
async def remove_staged_file(file_path: str):
    """Remove a single file from staging."""
    if file_path not in state.staged_files:
        raise HTTPException(status_code=404, detail=f"File not staged: {file_path}")

    # Remove from disk
    if state.stage_dir:
        actual_path = state.stage_dir / file_path
        if actual_path.exists():
            actual_path.unlink()

            # Clean up empty parent directories
            parent = actual_path.parent
            while parent != state.stage_dir:
                if not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break

    # Remove from tracking
    del state.staged_files[file_path]

    return {"status": "removed", "name": file_path}


@app.delete("/api/staged")
async def clear_staged_files():
    """Clear all staged files."""
    state.clear_staging()
    return {"status": "cleared"}


@app.post("/api/unload")
async def unload_docpack():
    """Unload the current docpack and reset to idle state."""
    if state.stage not in ("ready", "error"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot unload in current state: {state.stage}"
        )

    # Clean up docpack file
    if state.docpack_path and state.docpack_path.exists():
        try:
            state.docpack_path.unlink()
        except OSError:
            pass

    # Reset state
    state.docpack_path = None
    state.stage = "idle"
    state.error = None
    state.progress_phase = ""
    state.progress_current = 0
    state.progress_total = 0
    state.stats = {}
    state.pipeline_start_time = None
    state.phase_start_time = None
    state.phase_times = {}

    # Also clear staged files
    state.clear_staging()

    # Broadcast the reset state
    await broadcast_state()

    return {"status": "unloaded"}


@app.get("/api/events")
async def sse_events():
    """Server-Sent Events endpoint for real-time updates."""

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        state.subscribers.append(queue)

        try:
            # Send initial state
            import json
            import time

            # Calculate timing for initial state
            phase_elapsed = None
            if state.phase_start_time is not None:
                phase_elapsed = time.time() - state.phase_start_time
            pipeline_elapsed = None
            if state.pipeline_start_time is not None:
                pipeline_elapsed = time.time() - state.pipeline_start_time

            # Try to load timing from database if not in memory but we have a docpack
            phase_times = state.phase_times
            if not phase_times and state.stage == "ready" and state.docpack_path:
                try:
                    conn = state.get_connection()
                    if conn:
                        from docpack.storage import get_metadata
                        phase_times_json = get_metadata(conn, "phase_times")
                        if phase_times_json:
                            phase_times = json.loads(phase_times_json)
                            state.phase_times = phase_times  # Cache it
                        conn.close()
                except Exception:
                    pass  # Ignore errors loading timing

            initial = {
                "stage": state.stage,
                "progress": {
                    "phase": state.progress_phase,
                    "current": state.progress_current,
                    "total": state.progress_total,
                },
                "stats": state.stats,
                "error": state.error,
                "timing": {
                    "phase_elapsed": phase_elapsed,
                    "pipeline_elapsed": pipeline_elapsed,
                    "phase_times": phase_times,
                },
            }
            yield f"data: {json.dumps(initial)}\n\n"

            # Stream updates
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
        finally:
            state.subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/process")
async def process_directory(request: ProcessRequest):
    """Start processing a directory or staged files."""
    if state.stage not in ("idle", "ready", "error"):
        raise HTTPException(status_code=409, detail="Processing already in progress")

    # Determine source path
    if request.use_staged:
        # Use staged files
        if not state.stage_dir or not state.staged_files:
            raise HTTPException(status_code=400, detail="No files staged for processing")
        process_path = state.stage_dir
        source_desc = f"staged files ({len(state.staged_files)} files)"
    elif request.path:
        # Use filesystem path (existing behavior)
        process_path = Path(request.path).expanduser().resolve()
        if not process_path.exists():
            raise HTTPException(status_code=400, detail=f"Path not found: {request.path}")
        source_desc = str(process_path)
    else:
        raise HTTPException(status_code=400, detail="Either 'path' or 'use_staged' must be provided")

    # Clean up old docpack if any
    if state.docpack_path and state.docpack_path.exists():
        try:
            state.docpack_path.unlink()
        except OSError:
            pass
    state.docpack_path = None

    # Start pipeline in background
    asyncio.create_task(run_pipeline(str(process_path)))

    return {"status": "started", "path": source_desc}


def _search_sync(docpack_path: Path, query: str, k: int):
    """Run search in a thread with its own connection."""
    from docpack.recall import recall

    conn = sqlite3.connect(str(docpack_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        return recall(conn, query, k=k)
    finally:
        conn.close()


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic search against the loaded docpack."""
    if state.stage != "ready" or state.docpack_path is None:
        raise HTTPException(status_code=400, detail="No docpack loaded. Process a directory first.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = await asyncio.to_thread(
            _search_sync,
            state.docpack_path,
            request.query,
            request.k,
        )

        return SearchResponse(
            results=[
                SearchResult(
                    file_path=r.file_path,
                    chunk_index=r.chunk_index,
                    text=r.text,
                    score=r.score,
                )
                for r in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _ask_sync(docpack_path: Path, query: str):
    """Run answer generation in a thread with its own connection."""
    from docpack.answer import generate_answer

    conn = sqlite3.connect(str(docpack_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        return generate_answer(conn, query)
    finally:
        conn.close()


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Ask a question and get a citation-backed answer.

    Returns a concise, accurate answer with citations to source material.
    Only states facts directly supported by the documents.
    """
    if state.stage != "ready" or state.docpack_path is None:
        raise HTTPException(status_code=400, detail="No docpack loaded. Process a directory first.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = await asyncio.to_thread(
            _ask_sync,
            state.docpack_path,
            request.query,
        )

        return AskResponse(
            answer=result.answer,
            citations=[
                CitationModel(
                    id=c.id,
                    file_path=c.file_path,
                    chunk_index=c.chunk_index,
                    quote=c.quote,
                    start_char=c.start_char,
                    end_char=c.end_char,
                )
                for c in result.citations
            ],
            confidence=result.confidence,
            sources_retrieved=result.sources_retrieved,
            sources_used=result.sources_used,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Server Runner
# =============================================================================


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    static_dir: str | None = None,
    open_browser: bool = True,
):
    """
    Run the Doctown web server.

    Args:
        host: Host to bind to
        port: Port to bind to
        static_dir: Path to Svelte build directory (optional)
        open_browser: Open browser on startup
    """
    import uvicorn

    # Mount static files if directory exists
    if static_dir:
        static_path = Path(static_dir)
        if static_path.exists():
            app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
        else:
            print(f"Warning: Static directory not found: {static_dir}", file=sys.stderr)

    # Open browser after a short delay
    if open_browser:
        def open_browser_delayed():
            import time
            time.sleep(1)
            webbrowser.open(f"http://{host}:{port}")

        import threading
        threading.Thread(target=open_browser_delayed, daemon=True).start()

    print(f"Doctown server running at http://{host}:{port}")
    print("Press Ctrl+C to stop")

    # Run server
    uvicorn.run(app, host=host, port=port, log_level="warning")
