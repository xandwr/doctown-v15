"""
Regex search tool for searching file contents with full Python regex support.

Unlike grep (simple string matching), this uses Python's re module for
full regex power including capture groups, lookahead, etc.

Includes timeout protection against ReDoS attacks.
"""

from __future__ import annotations

import re
import signal
import sqlite3
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any


class RegexTimeout(Exception):
    """Raised when regex operation times out."""

    pass


@dataclass
class RegexMatch:
    """A single regex match."""

    file_path: str
    line_number: int
    line_content: str
    match_text: str
    match_start: int  # Character offset within line
    match_end: int  # Character offset within line
    groups: tuple[str, ...]  # Captured groups
    named_groups: dict[str, str]  # Named captured groups

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "match_text": self.match_text,
            "match_start": self.match_start,
            "match_end": self.match_end,
            "groups": list(self.groups),
            "named_groups": self.named_groups,
        }


@dataclass
class RegexResult:
    """Result of a regex search operation."""

    matches: list[RegexMatch]
    total_matches: int
    files_searched: int
    files_matched: int
    truncated: bool
    pattern: str
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "matches": [m.to_dict() for m in self.matches],
            "total_matches": self.total_matches,
            "files_searched": self.files_searched,
            "files_matched": self.files_matched,
            "truncated": self.truncated,
            "pattern": self.pattern,
            "error": self.error,
        }


def regex_search(
    conn: sqlite3.Connection,
    pattern: str,
    *,
    flags: str = "",
    path_filter: str | None = None,
    max_results: int = 100,
    timeout_seconds: int = 5,
    multiline: bool = False,
) -> RegexResult:
    """
    Search for a regex pattern in file contents.

    Args:
        conn: Database connection
        pattern: Python regex pattern
        flags: Regex flags string (i=ignorecase, m=multiline, s=dotall)
        path_filter: Optional glob pattern to filter files
        max_results: Maximum number of matches to return
        timeout_seconds: Timeout for regex operations (ReDoS protection)
        multiline: If True, search across lines (pattern can match newlines)

    Returns:
        RegexResult with matches and metadata
    """
    # Parse flags
    re_flags = 0
    for flag in flags.lower():
        if flag == "i":
            re_flags |= re.IGNORECASE
        elif flag == "m":
            re_flags |= re.MULTILINE
        elif flag == "s":
            re_flags |= re.DOTALL

    # Compile pattern
    try:
        compiled = re.compile(pattern, re_flags)
    except re.error as e:
        return RegexResult(
            matches=[],
            total_matches=0,
            files_searched=0,
            files_matched=0,
            truncated=False,
            pattern=pattern,
            error=f"Invalid regex pattern: {e}",
        )

    cursor = conn.execute(
        """
        SELECT id, path, content
        FROM files
        WHERE is_binary = 0 AND content IS NOT NULL
        """,
    )

    matches: list[RegexMatch] = []
    files_searched = 0
    files_matched = 0
    total_matches = 0
    truncated = False
    error = None

    for row in cursor.fetchall():
        file_id, file_path, content = row

        # Apply path filter if specified
        if path_filter and not _matches_glob(file_path, path_filter):
            continue

        files_searched += 1

        try:
            file_matches = _search_with_regex(
                file_path,
                content,
                compiled,
                multiline=multiline,
                timeout_seconds=timeout_seconds,
            )
        except RegexTimeout:
            error = f"Regex timed out after {timeout_seconds}s (possible ReDoS pattern)"
            truncated = True
            break

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

    return RegexResult(
        matches=matches,
        total_matches=total_matches,
        files_searched=files_searched,
        files_matched=files_matched,
        truncated=truncated,
        pattern=pattern,
        error=error,
    )


def _matches_glob(path: str, pattern: str) -> bool:
    """Check if path matches a glob pattern."""
    if "**" in pattern:
        parts = pattern.split("**")
        if len(parts) == 2:
            prefix, suffix = parts
            if prefix and not path.startswith(prefix.rstrip("/")):
                return False
            if suffix:
                suffix = suffix.lstrip("/")
                path_parts = path.split("/")
                for i in range(len(path_parts)):
                    remaining = "/".join(path_parts[i:])
                    if fnmatch(remaining, suffix) or fnmatch(path_parts[-1], suffix):
                        return True
                return False
            return True
        return fnmatch(path, pattern)
    return fnmatch(path, pattern)


def _timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for regex timeout."""
    raise RegexTimeout("Regex operation timed out")


def _search_with_regex(
    file_path: str,
    content: str,
    compiled: re.Pattern,
    *,
    multiline: bool,
    timeout_seconds: int,
) -> list[RegexMatch]:
    """Search content with compiled regex pattern."""
    matches: list[RegexMatch] = []

    # Set up timeout (only works on Unix)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    except (ValueError, AttributeError):
        # Signal not available (Windows or non-main thread)
        pass

    try:
        if multiline:
            # Search entire content
            for m in compiled.finditer(content):
                # Find line number for match start
                line_num = content[: m.start()].count("\n") + 1
                # Get the line containing the match start
                line_start = content.rfind("\n", 0, m.start()) + 1
                line_end = content.find("\n", m.start())
                if line_end == -1:
                    line_end = len(content)
                line_content = content[line_start:line_end]

                matches.append(
                    RegexMatch(
                        file_path=file_path,
                        line_number=line_num,
                        line_content=line_content,
                        match_text=m.group(0),
                        match_start=m.start() - line_start,
                        match_end=m.end() - line_start,
                        groups=m.groups(),
                        named_groups=m.groupdict(),
                    )
                )
        else:
            # Search line by line
            lines = content.split("\n")
            for i, line in enumerate(lines):
                for m in compiled.finditer(line):
                    matches.append(
                        RegexMatch(
                            file_path=file_path,
                            line_number=i + 1,
                            line_content=line,
                            match_text=m.group(0),
                            match_start=m.start(),
                            match_end=m.end(),
                            groups=m.groups(),
                            named_groups=m.groupdict(),
                        )
                    )
    finally:
        # Cancel timeout
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (ValueError, AttributeError):
            pass

    return matches


def format_regex_result(result: RegexResult) -> str:
    """Format regex result for display."""
    if result.error:
        return f"Error: {result.error}"

    if not result.matches:
        return f"No matches for /{result.pattern}/ (searched {result.files_searched} files)"

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

        # Show matching line with match highlighted
        lines.append(f"{match.line_number:4d}  {match.line_content}")
        if match.match_text != match.line_content:
            # Show the actual match if it's different from the whole line
            lines.append(f"      match: {repr(match.match_text)}")
        if match.groups:
            lines.append(f"      groups: {match.groups}")
        if match.named_groups:
            lines.append(f"      named: {match.named_groups}")
        lines.append("")

    return "\n".join(lines)
