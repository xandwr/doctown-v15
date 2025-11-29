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

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# =============================================================================
# State Management
# =============================================================================


@dataclass
class AppState:
    """Global application state."""

    stage: str = "idle"  # idle, freezing, chunking, embedding, ready, error
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

    def get_connection(self) -> sqlite3.Connection | None:
        """Get a new database connection (thread-safe)."""
        if self.docpack_path is None:
            return None
        conn = sqlite3.connect(str(self.docpack_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn


# Global state instance
state = AppState()


async def broadcast_state():
    """Send current state to all SSE subscribers."""
    data = {
        "stage": state.stage,
        "progress": {
            "phase": state.progress_phase,
            "current": state.progress_current,
            "total": state.progress_total,
        },
        "stats": state.stats,
        "error": state.error,
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
    """Request to process a directory."""

    path: str


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
    from docpack.ingest.freeze import detect_source
    from docpack.storage import init_db, insert_file, set_metadata, get_stats
    from docpack import FORMAT_VERSION, __version__
    from datetime import datetime, timezone

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

    with source:
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

            if file_count % 10 == 0 or file_count == total_files:
                progress_update_fn("freezing", file_count, total_files)

    # Store metadata
    set_metadata(conn, "format_version", FORMAT_VERSION)
    set_metadata(conn, "docpack_version", __version__)
    set_metadata(conn, "created_at", datetime.now(timezone.utc).isoformat())
    set_metadata(conn, "source", target_path)
    set_metadata(conn, "file_count", str(file_count))
    set_metadata(conn, "total_bytes", str(total_bytes))

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

        embed_all(conn, verbose=False, progress_callback=embed_progress)
        stats = get_stats(conn)
        progress_update_fn("embedding_done", chunk_count, chunk_count, stats)

    conn.close()
    return stats


async def run_pipeline(target_path: str):
    """
    Run the freeze -> chunk -> embed pipeline with progress updates.

    This wraps the existing docpack functions and broadcasts progress via SSE.
    """
    import os

    try:
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
                if base_phase == "freezing":
                    state.stage = "chunking"
                elif base_phase == "chunking":
                    state.stage = "embedding"
                elif base_phase == "embedding":
                    state.stage = "ready"
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


@app.get("/api/events")
async def sse_events():
    """Server-Sent Events endpoint for real-time updates."""

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        state.subscribers.append(queue)

        try:
            # Send initial state
            import json

            initial = {
                "stage": state.stage,
                "progress": {
                    "phase": state.progress_phase,
                    "current": state.progress_current,
                    "total": state.progress_total,
                },
                "stats": state.stats,
                "error": state.error,
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
    """Start processing a directory."""
    # Validate path
    path = Path(request.path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {request.path}")

    if state.stage not in ("idle", "ready", "error"):
        raise HTTPException(status_code=409, detail="Processing already in progress")

    # Clean up old docpack if any
    if state.docpack_path and state.docpack_path.exists():
        try:
            state.docpack_path.unlink()
        except OSError:
            pass
    state.docpack_path = None

    # Start pipeline in background
    asyncio.create_task(run_pipeline(str(path)))

    return {"status": "started", "path": str(path)}


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
