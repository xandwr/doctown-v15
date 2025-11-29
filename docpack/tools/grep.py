"""
Grep tool for searching file contents.

Searches for patterns in file contents and returns matching lines
with surrounding context, similar to `grep -C`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from fnmatch import fnmatch


@dataclass
class GrepMatch:
    """A single grep match with context."""

    file_path: str
    line_number: int
    line_content: str
    context_before: list[str]
    context_after: list[str]
    match_start: int  # Character offset within line
    match_end: int  # Character offset within line

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "match_start": self.match_start,
            "match_end": self.match_end,
        }


@dataclass
class GrepResult:
    """Result of a grep operation."""

    matches: list[GrepMatch]
    total_matches: int
    files_searched: int
    files_matched: int
    truncated: bool  # True if results were limited

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "matches": [m.to_dict() for m in self.matches],
            "total_matches": self.total_matches,
            "files_searched": self.files_searched,
            "files_matched": self.files_matched,
            "truncated": self.truncated,
        }


def grep(
    conn: sqlite3.Connection,
    pattern: str,
    *,
    context_lines: int = 2,
    path_filter: str | None = None,
    case_sensitive: bool = True,
    max_results: int = 100,
) -> GrepResult:
    """
    Search for a pattern in file contents.

    Args:
        conn: Database connection
        pattern: Text pattern to search for
        context_lines: Number of lines of context before/after matches
        path_filter: Optional glob pattern to filter files (e.g., "*.py", "src/**/*.ts")
        case_sensitive: Whether to match case (default True)
        max_results: Maximum number of matches to return

    Returns:
        GrepResult with matches and metadata
    """
    # Build query to find files containing the pattern
    search_pattern = pattern if case_sensitive else pattern.lower()

    cursor = conn.execute(
        """
        SELECT id, path, content
        FROM files
        WHERE is_binary = 0 AND content IS NOT NULL
        """,
    )

    matches: list[GrepMatch] = []
    files_searched = 0
    files_matched = 0
    total_matches = 0
    truncated = False

    for row in cursor.fetchall():
        file_id, file_path, content = row

        # Apply path filter if specified
        if path_filter and not _matches_glob(file_path, path_filter):
            continue

        files_searched += 1

        # Search for pattern in content
        file_matches = _search_content(
            file_path,
            content,
            pattern,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
        )

        if file_matches:
            files_matched += 1
            total_matches += len(file_matches)

            # Add matches up to limit
            remaining = max_results - len(matches)
            if remaining <= 0:
                truncated = True
                break

            matches.extend(file_matches[:remaining])
            if len(file_matches) > remaining:
                truncated = True
                break

    return GrepResult(
        matches=matches,
        total_matches=total_matches,
        files_searched=files_searched,
        files_matched=files_matched,
        truncated=truncated,
    )


def _matches_glob(path: str, pattern: str) -> bool:
    """Check if path matches a glob pattern."""
    # Handle ** for recursive matching
    if "**" in pattern:
        # Split pattern into parts
        parts = pattern.split("**")
        if len(parts) == 2:
            prefix, suffix = parts
            # Check if path starts with prefix (if any) and ends with suffix pattern
            if prefix and not path.startswith(prefix.rstrip("/")):
                return False
            if suffix:
                # Match the suffix part
                suffix = suffix.lstrip("/")
                # Check if any suffix of path matches
                path_parts = path.split("/")
                for i in range(len(path_parts)):
                    remaining = "/".join(path_parts[i:])
                    if fnmatch(remaining, suffix) or fnmatch(path_parts[-1], suffix):
                        return True
                return False
            return True
        return fnmatch(path, pattern)
    return fnmatch(path, pattern)


def _search_content(
    file_path: str,
    content: str,
    pattern: str,
    *,
    case_sensitive: bool,
    context_lines: int,
) -> list[GrepMatch]:
    """Search content for pattern and return matches with context."""
    lines = content.split("\n")
    matches: list[GrepMatch] = []

    search_pattern = pattern if case_sensitive else pattern.lower()

    for i, line in enumerate(lines):
        search_line = line if case_sensitive else line.lower()

        # Find all occurrences in this line
        start = 0
        while True:
            pos = search_line.find(search_pattern, start)
            if pos == -1:
                break

            # Get context lines
            context_start = max(0, i - context_lines)
            context_end = min(len(lines), i + context_lines + 1)

            matches.append(
                GrepMatch(
                    file_path=file_path,
                    line_number=i + 1,  # 1-indexed
                    line_content=line,
                    context_before=lines[context_start:i],
                    context_after=lines[i + 1 : context_end],
                    match_start=pos,
                    match_end=pos + len(pattern),
                )
            )

            # Move past this match to find more in same line
            start = pos + 1

    return matches


def format_grep_result(result: GrepResult) -> str:
    """Format grep result for display."""
    if not result.matches:
        return f"No matches found (searched {result.files_searched} files)"

    lines = []
    lines.append(
        f"Found {result.total_matches} matches in {result.files_matched} files"
    )
    if result.truncated:
        lines.append(f"(showing first {len(result.matches)} matches)")
    lines.append("")

    current_file = None
    for match in result.matches:
        if match.file_path != current_file:
            if current_file is not None:
                lines.append("")
            lines.append(f"=== {match.file_path} ===")
            current_file = match.file_path

        # Show context before
        for j, ctx_line in enumerate(match.context_before):
            ctx_line_num = match.line_number - len(match.context_before) + j
            lines.append(f"  {ctx_line_num:4d}  {ctx_line}")

        # Show matching line with highlight marker
        lines.append(f"> {match.line_number:4d}  {match.line_content}")

        # Show context after
        for j, ctx_line in enumerate(match.context_after):
            ctx_line_num = match.line_number + j + 1
            lines.append(f"  {ctx_line_num:4d}  {ctx_line}")

        lines.append("")

    return "\n".join(lines)
