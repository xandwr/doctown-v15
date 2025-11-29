"""
Docpack tools module.

These are the MCP tools that let agents explore and interact with a docpack.
Each tool is designed to be composable - output from one can feed into another.

Tools are organized into categories:
- Search: grep, regex_search (find content)
- Query: structured_query (SQL-like queries)
- Transform: summarize (compress information)
- Marginalia: write_note, read_notes, etc. (persist findings via storage module)
"""

from .grep import grep, GrepMatch, GrepResult, format_grep_result
from .query import structured_query, QueryFilter, QueryResult, QueryType, format_query_result
from .regex import regex_search, RegexMatch, RegexResult, format_regex_result
from .summarize import summarize, SummaryResult, format_summary_result

__all__ = [
    # Search tools
    "grep",
    "GrepMatch",
    "GrepResult",
    "format_grep_result",
    "regex_search",
    "RegexMatch",
    "RegexResult",
    "format_regex_result",
    # Query tools
    "structured_query",
    "QueryFilter",
    "QueryResult",
    "QueryType",
    "format_query_result",
    # Transform tools
    "summarize",
    "SummaryResult",
    "format_summary_result",
]
