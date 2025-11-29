"""
MCP Server for docpack.

Exposes a frozen docpack to AI agents via the Model Context Protocol.
The docpack is a portable universe with two realms:
- Frozen realm: files, chunks, vectors (read-only)
- Marginalia: notes, sessions, artifacts (agent-writable)

Tools:
    ls              - List all files in the docpack
    read            - Read file contents by path
    recall          - Semantic search against embedded chunks
    grep            - Search file contents with context
    regex_search    - Full Python regex search
    query           - Structured queries (files, chunks, stats)
    summarize       - Generate summaries of content
    write_note      - Write a note to the marginalia
    read_notes      - Read all notes from the marginalia
    get_stats       - Get docpack statistics including marginalia
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from docpack.recall import recall as do_recall
from docpack.storage import (
    get_all_files,
    get_file_by_path,
    init_db,
    read_all_notes,
    read_note,
    write_note,
)
from docpack.tools import (
    format_grep_result,
    format_query_result,
    format_regex_result,
    format_summary_result,
    grep,
    regex_search,
    structured_query,
    summarize,
)


def create_server(docpack_path: str) -> Server:
    """
    Create an MCP server for a docpack.

    Args:
        docpack_path: Path to the .docpack file

    Returns:
        Configured MCP Server instance
    """
    path = Path(docpack_path)
    if not path.exists():
        raise FileNotFoundError(f"Docpack not found: {docpack_path}")

    server = Server("docpack")

    # Keep connection open for the server lifetime
    conn: sqlite3.Connection | None = None

    def get_conn() -> sqlite3.Connection:
        nonlocal conn
        if conn is None:
            conn = init_db(str(path))
        return conn

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            # === Frozen Realm Tools (read-only) ===
            Tool(
                name="ls",
                description="List all files in the docpack with their paths and sizes",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="read",
                description="Read the contents of a file by its path",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path within the docpack",
                        },
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="recall",
                description="Semantic search for relevant code chunks using natural language",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="grep",
                description="Search file contents for a pattern, returns matching lines with context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Text pattern to search for",
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Lines of context around matches (default: 2)",
                            "default": 2,
                        },
                        "path_filter": {
                            "type": "string",
                            "description": "Glob pattern to filter files (e.g., '*.py', 'src/**/*.ts')",
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Whether to match case (default: true)",
                            "default": True,
                        },
                    },
                    "required": ["pattern"],
                },
            ),
            Tool(
                name="regex_search",
                description="Search file contents using Python regular expressions with capture groups",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Python regex pattern",
                        },
                        "flags": {
                            "type": "string",
                            "description": "Regex flags: i=ignorecase, m=multiline, s=dotall",
                            "default": "",
                        },
                        "path_filter": {
                            "type": "string",
                            "description": "Glob pattern to filter files",
                        },
                    },
                    "required": ["pattern"],
                },
            ),
            Tool(
                name="query",
                description="Execute structured queries against the docpack (files, chunks, or stats)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["files", "chunks", "stats"],
                            "description": "Type of query",
                        },
                        "extension": {
                            "type": "string",
                            "description": "Filter by file extension (e.g., 'py', '.ts')",
                        },
                        "path_contains": {
                            "type": "string",
                            "description": "Filter paths containing this substring",
                        },
                        "content_contains": {
                            "type": "string",
                            "description": "Filter by content containing this text",
                        },
                        "order_by": {
                            "type": "string",
                            "description": "Column to order by (path, size_bytes, -size_bytes for desc)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results to return (default: 100)",
                            "default": 100,
                        },
                    },
                    "required": ["type"],
                },
            ),
            Tool(
                name="summarize",
                description="Generate a summary of a file or content",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to file to summarize",
                        },
                        "content": {
                            "type": "string",
                            "description": "Direct content to summarize (alternative to path)",
                        },
                        "style": {
                            "type": "string",
                            "enum": ["brief", "detailed", "bullets"],
                            "description": "Summary style (default: brief)",
                            "default": "brief",
                        },
                    },
                },
            ),
            # === Marginalia Tools (agent-writable) ===
            Tool(
                name="write_note",
                description="Write a note to the docpack's marginalia (persists inside the docpack)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Unique key/identifier for the note",
                        },
                        "content": {
                            "type": "string",
                            "description": "Note content (markdown supported)",
                        },
                    },
                    "required": ["key", "content"],
                },
            ),
            Tool(
                name="read_notes",
                description="Read all notes from the docpack's marginalia",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Optional: read a specific note by key",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        db = get_conn()

        # === Frozen Realm Tools ===

        if name == "ls":
            files = get_all_files(db)
            if not files:
                return [TextContent(type="text", text="No files in docpack")]

            lines = []
            for f in files:
                status = "[bin]" if f["is_binary"] else "[txt]"
                lines.append(f"{status} {f['path']} ({f['size_bytes']} bytes)")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "read":
            file_path = arguments.get("path", "")
            if not file_path:
                return [TextContent(type="text", text="Error: path is required")]

            file_record = get_file_by_path(db, file_path)
            if not file_record:
                return [TextContent(type="text", text=f"Error: File not found: {file_path}")]

            if file_record["is_binary"]:
                return [TextContent(type="text", text=f"Error: Cannot read binary file: {file_path}")]

            content = file_record.get("content", "")
            return [TextContent(type="text", text=content or "(empty file)")]

        elif name == "recall":
            query = arguments.get("query", "")
            if not query:
                return [TextContent(type="text", text="Error: query is required")]

            k = arguments.get("k", 5)
            results = do_recall(db, query, k=k)

            if not results:
                return [TextContent(type="text", text="No results found")]

            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"[{i}] {r.file_path} (chunk {r.chunk_index}) — score: {r.score:.4f}")
                lines.append("-" * 60)
                lines.append(r.text)
                lines.append("")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "grep":
            pattern = arguments.get("pattern", "")
            if not pattern:
                return [TextContent(type="text", text="Error: pattern is required")]

            result = grep(
                db,
                pattern,
                context_lines=arguments.get("context_lines", 2),
                path_filter=arguments.get("path_filter"),
                case_sensitive=arguments.get("case_sensitive", True),
            )
            return [TextContent(type="text", text=format_grep_result(result))]

        elif name == "regex_search":
            pattern = arguments.get("pattern", "")
            if not pattern:
                return [TextContent(type="text", text="Error: pattern is required")]

            result = regex_search(
                db,
                pattern,
                flags=arguments.get("flags", ""),
                path_filter=arguments.get("path_filter"),
            )
            return [TextContent(type="text", text=format_regex_result(result))]

        elif name == "query":
            query_type = arguments.get("type", "stats")

            # Build filter from arguments
            filter_dict = {}
            for key in ["extension", "path_contains", "content_contains"]:
                if key in arguments:
                    filter_dict[key] = arguments[key]

            result = structured_query(
                db,
                query_type,
                filter=filter_dict if filter_dict else None,
                order_by=arguments.get("order_by"),
                limit=arguments.get("limit", 100),
            )
            return [TextContent(type="text", text=format_query_result(result))]

        elif name == "summarize":
            result = summarize(
                db,
                path=arguments.get("path"),
                content=arguments.get("content"),
                style=arguments.get("style", "brief"),
            )
            return [TextContent(type="text", text=format_summary_result(result))]

        # === Marginalia Tools ===

        elif name == "write_note":
            key = arguments.get("key", "")
            content = arguments.get("content", "")
            if not key or not content:
                return [TextContent(type="text", text="Error: key and content are required")]

            note = write_note(db, key, content)
            return [TextContent(type="text", text=f"Note saved: {note.key} (updated: {note.updated_at})")]

        elif name == "read_notes":
            key = arguments.get("key")
            if key:
                note = read_note(db, key)
                if note:
                    return [TextContent(type="text", text=f"[{note.key}]\n{note.content}")]
                return [TextContent(type="text", text=f"Note not found: {key}")]

            notes = read_all_notes(db)
            if not notes:
                return [TextContent(type="text", text="No notes in marginalia")]

            lines = []
            for note in notes:
                lines.append(f"## {note.key}")
                lines.append(f"*Updated: {note.updated_at}*")
                lines.append("")
                lines.append(note.content)
                lines.append("")
                lines.append("---")
                lines.append("")

            return [TextContent(type="text", text="\n".join(lines))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def run_server(docpack_path: str) -> None:
    """
    Run the MCP server over stdio.

    Args:
        docpack_path: Path to the .docpack file
    """
    server = create_server(docpack_path)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
