"""
Docpack CLI.

Commands:
    freeze     Turn folders/zips/URLs into a .docpack
    deck       Launch the Flight Deck (TUI / web cockpit)
    recall     Semantic search against a docpack
    serve      Expose docpack via MCP for AI agents
    info       Inspect metadata, stats, counts
    run        Freeze + serve a temp docpack (one-shot)
    task       Execute a natural language task against a docpack

Examples:
    docpack freeze ./my-project -o project.docpack
    docpack freeze https://github.com/user/repo/archive/main.zip
    docpack info project.docpack
    docpack recall project.docpack "where is jwt validation?"
    docpack task project.docpack "find all TODO comments and summarize them"
"""

from __future__ import annotations

import argparse
import sys


def cmd_freeze(args: argparse.Namespace) -> int:
    """Handle freeze command."""
    from pathlib import Path

    from docpack.ingest import freeze

    # Check if output file exists and prompt for overwrite
    output = args.output
    if output and not args.temp:
        output_path = Path(output)
        if output_path.exists():
            if args.yes:
                output_path.unlink()
            else:
                response = input(f"'{output}' exists. Overwrite? [y/N] ").strip().lower()
                if response in ("y", "yes"):
                    output_path.unlink()
                else:
                    print("Aborted.")
                    return 1

    try:
        output_path = freeze(
            args.target,
            output=output,
            use_temp=args.temp,
            verbose=args.verbose,
            skip_chunking=args.no_chunk,
            skip_embedding=args.no_embed,
            skip_summarize=args.no_summarize,
            embedding_model=args.model,
            summarize_model=args.summarize_model,
        )
        if not args.verbose:
            print(f"Created: {output_path}")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Handle info command."""
    from pathlib import Path

    from docpack.storage import get_all_metadata, get_stats, init_db

    docpack_path = Path(args.docpack)
    if not docpack_path.exists():
        print(f"Error: File not found: {docpack_path}", file=sys.stderr)
        return 1

    conn = init_db(str(docpack_path))
    try:
        stats = get_stats(conn)
        metadata = get_all_metadata(conn)

        # Header
        print(f"Docpack: {docpack_path}")
        print(f"Size: {docpack_path.stat().st_size:,} bytes")

        # Version info
        if "format_version" in metadata or "docpack_version" in metadata:
            print()
            print("Version:")
            if "format_version" in metadata:
                print(f"  Format: {metadata['format_version']}")
            if "docpack_version" in metadata:
                print(f"  Docpack: {metadata['docpack_version']}")

        # Timestamps
        if "created_at" in metadata:
            print()
            print("Created:")
            print(f"  {metadata['created_at']}")

        # Stats
        print()
        print("Stats:")
        print(f"  Files: {stats.get('total_files', 0)}")
        print(f"  Chunks: {stats.get('total_chunks', 0)}")
        print(f"  Vectors: {stats.get('total_vectors', 0)}")
        print(f"  Content size: {stats.get('total_size_bytes', 0):,} bytes")

        # Embedding info
        if "embedding_model" in metadata:
            print()
            print("Embedding:")
            print(f"  Model: {metadata.get('embedding_model', 'N/A')}")
            print(f"  Dimensions: {metadata.get('embedding_dims', 'N/A')}")

        # Summary info
        if "summary_model" in metadata:
            print()
            print("Summaries:")
            print(f"  Model: {metadata.get('summary_model', 'N/A')}")
            print(f"  Count: {metadata.get('summary_count', 'N/A')}")

        # Config flags
        config_keys = [k for k in metadata if k.startswith("config.")]
        if config_keys:
            print()
            print("Config:")
            for key in sorted(config_keys):
                short_key = key.replace("config.", "")
                print(f"  {short_key}: {metadata[key]}")

        # Source
        if "source" in metadata:
            print()
            print("Source:")
            print(f"  {metadata['source']}")

        # Stage
        if "stage" in metadata:
            print()
            print(f"Stage: {metadata['stage']}")

        return 0
    finally:
        conn.close()


def cmd_recall(args: argparse.Namespace) -> int:
    """Handle recall command."""
    from pathlib import Path

    from docpack.recall import recall
    from docpack.storage import init_db

    docpack_path = Path(args.docpack)
    if not docpack_path.exists():
        print(f"Error: File not found: {docpack_path}", file=sys.stderr)
        return 1

    conn = init_db(str(docpack_path))
    try:
        results = recall(
            conn,
            args.query,
            k=args.k,
            model_name=args.model,
            threshold=args.threshold,
        )

        if not results:
            print("No results found.")
            return 0

        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r.file_path} (chunk {r.chunk_index}) — score: {r.score:.4f}")
            print("-" * 60)
            # Show summary if available
            if r.summary:
                print(f"Summary: {r.summary}")
                print("-" * 60)
            # Truncate long text for display
            text = r.text
            if len(text) > 500 and not args.full:
                text = text[:500] + "..."
            print(text)

        return 0
    finally:
        conn.close()


def cmd_serve(args: argparse.Namespace) -> int:
    """Handle serve command."""
    import asyncio
    from pathlib import Path

    from docpack.serve import run_server

    docpack_path = Path(args.docpack)
    if not docpack_path.exists():
        print(f"Error: File not found: {docpack_path}", file=sys.stderr)
        return 1

    try:
        print(f"Serving: {docpack_path}", file=sys.stderr)
        print("MCP server ready on stdio (waiting for client connection)", file=sys.stderr)
        print("Tip: This command is meant to be invoked by an MCP client.", file=sys.stderr)
        print("     Press Ctrl+C to exit.", file=sys.stderr)
        asyncio.run(run_server(str(docpack_path)))
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Handle run command (freeze + serve)."""
    import asyncio

    from docpack.ingest import freeze
    from docpack.serve import run_server

    try:
        # Freeze to temp file
        print(f"Freezing {args.target}...", file=sys.stderr)
        output_path = freeze(
            args.target,
            use_temp=True,
            verbose=args.verbose,
            embedding_model=args.model,
        )
        print(f"Created: {output_path}", file=sys.stderr)
        print("MCP server ready on stdio (waiting for client connection)", file=sys.stderr)
        print("Tip: This command is meant to be invoked by an MCP client.", file=sys.stderr)
        print("     Press Ctrl+C to exit.", file=sys.stderr)

        # Serve it - blocks waiting for MCP protocol messages on stdin
        asyncio.run(run_server(str(output_path)))
        return 0
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_task(args: argparse.Namespace) -> int:
    """Handle task command - execute a workflow against a docpack."""
    import asyncio
    import json
    from pathlib import Path

    from docpack.workflow import run_task

    docpack_path = Path(args.docpack)
    if not docpack_path.exists():
        print(f"Error: File not found: {docpack_path}", file=sys.stderr)
        return 1

    try:
        if args.verbose:
            print(f"Running task: {args.task}", file=sys.stderr)
            print(f"Docpack: {docpack_path}", file=sys.stderr)
            print(file=sys.stderr)

        result = asyncio.run(
            run_task(
                str(docpack_path),
                args.task,
                model=args.model,
            )
        )

        if args.verbose:
            print(f"Status: {result.status}", file=sys.stderr)
            print(f"Duration: {result.duration_ms}ms", file=sys.stderr)
            print(f"Steps: {len(result.step_results)}", file=sys.stderr)
            print(file=sys.stderr)

            for sr in result.step_results:
                status = "[OK]" if sr.success else "[FAIL]"
                print(f"  {status} {sr.step.tool}: {sr.step.description}", file=sys.stderr)

            print(file=sys.stderr)

        # Output format
        if args.format == "json":
            print(result.to_json(indent=2))
        else:
            # Human-readable output
            print(f"Task: {result.task}")
            print(f"Status: {result.status}")
            print()

            if result.error:
                print(f"Error: {result.error}")
                return 1

            print("Output:")
            for key, value in result.output.items():
                print(f"\n## {key}")
                if isinstance(value, dict):
                    print(json.dumps(value, indent=2, default=str))
                elif isinstance(value, list):
                    for item in value[:10]:  # Limit output
                        if isinstance(item, dict):
                            print(f"  - {item}")
                        else:
                            print(f"  - {item}")
                    if len(value) > 10:
                        print(f"  ... and {len(value) - 10} more")
                else:
                    print(value)

        return 0 if result.status in ("completed", "partial") else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_web(args: argparse.Namespace) -> int:
    """Handle web command - start the web server."""
    from pathlib import Path

    from docpack.web import run_server

    # Find static directory
    static_dir = args.static_dir
    if not static_dir:
        # Try to find it relative to this file
        pkg_dir = Path(__file__).parent.parent
        candidates = [
            pkg_dir / "doctown_website" / "build",
            Path.cwd() / "doctown_website" / "build",
        ]
        for candidate in candidates:
            if candidate.exists():
                static_dir = str(candidate)
                break

    if static_dir and not Path(static_dir).exists():
        print(f"Warning: Static directory not found: {static_dir}", file=sys.stderr)
        print("Run 'npm run build' in doctown_website/ first", file=sys.stderr)
        static_dir = None

    try:
        run_server(
            host=args.host,
            port=args.port,
            static_dir=static_dir,
            open_browser=not args.no_browser,
        )
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_not_implemented(name: str) -> int:
    """Placeholder for unimplemented commands."""
    print(f"Command '{name}' not yet implemented.", file=sys.stderr)
    return 1


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="docpack",
        description="Freeze codebases for semantic search and AI consumption.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # freeze
    freeze_parser = subparsers.add_parser(
        "freeze",
        help="Turn folders/zips/URLs into a .docpack",
    )
    freeze_parser.add_argument(
        "target",
        help="Path or URL to ingest",
    )
    freeze_parser.add_argument(
        "-o",
        "--output",
        help="Output .docpack path",
    )
    freeze_parser.add_argument(
        "-t",
        "--temp",
        action="store_true",
        help="Create in temp directory (auto-cleanup)",
    )
    freeze_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information",
    )
    freeze_parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Skip chunking (raw ingestion only)",
    )
    freeze_parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding (chunking only)",
    )
    freeze_parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Skip LLM summarization (requires Ollama with qwen3:4b)",
    )
    freeze_parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Embedding model (default: google/embeddinggemma-300m)",
    )
    freeze_parser.add_argument(
        "--summarize-model",
        default=None,
        help="LLM model for summaries (default: qwen3:4b via Ollama)",
    )
    freeze_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Overwrite existing output file without prompting",
    )

    # info
    info_parser = subparsers.add_parser(
        "info",
        help="Inspect metadata, stats, counts",
    )
    info_parser.add_argument(
        "docpack",
        help="Path to .docpack file",
    )

    # recall
    recall_parser = subparsers.add_parser(
        "recall",
        help="Semantic search against a docpack",
    )
    recall_parser.add_argument(
        "docpack",
        help="Path to .docpack file",
    )
    recall_parser.add_argument(
        "query",
        help="Natural language search query",
    )
    recall_parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    recall_parser.add_argument(
        "-m",
        "--model",
        default="google/embeddinggemma-300m",
        help="Embedding model (must match freeze model)",
    )
    recall_parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="Minimum similarity score (0-1)",
    )
    recall_parser.add_argument(
        "--full",
        action="store_true",
        help="Show full chunk text (don't truncate)",
    )

    # serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Expose docpack via MCP",
    )
    serve_parser.add_argument(
        "docpack",
        help="Path to .docpack file",
    )

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Freeze + serve (one-shot)",
    )
    run_parser.add_argument(
        "target",
        help="Path or URL to ingest",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information",
    )
    run_parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Embedding model (default: google/embeddinggemma-300m)",
    )

    # deck
    deck_parser = subparsers.add_parser("deck", help="Launch the Flight Deck (TUI)")
    deck_parser.add_argument(
        "docpack",
        nargs="?",
        help="Path to .docpack file (optional, shows README if omitted)",
    )

    # web
    web_parser = subparsers.add_parser(
        "web",
        help="Start the web interface",
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    web_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    web_parser.add_argument(
        "--static-dir",
        help="Path to Svelte build directory (auto-detected if not specified)",
    )
    web_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    # task
    task_parser = subparsers.add_parser(
        "task",
        help="Execute a natural language task against a docpack",
    )
    task_parser.add_argument(
        "docpack",
        help="Path to .docpack file",
    )
    task_parser.add_argument(
        "task",
        help="Natural language task description",
    )
    task_parser.add_argument(
        "-f",
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    task_parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="LLM model for planning (default: qwen3:4b via Ollama)",
    )
    task_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show execution details",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "freeze":
        return cmd_freeze(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "recall":
        return cmd_recall(args)
    elif args.command == "serve":
        return cmd_serve(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "task":
        return cmd_task(args)
    elif args.command == "web":
        return cmd_web(args)
    else:
        return cmd_not_implemented(args.command)


if __name__ == "__main__":
    sys.exit(main())