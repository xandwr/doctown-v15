"""
Summarize tool for generating summaries of content.

Supports two modes:
1. Extractive: Simple first-N-sentences extraction (fast, no LLM needed)
2. LLM-powered: Uses Ollama for intelligent summarization (requires setup)
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Literal

from docpack.storage import get_file_by_path


@dataclass
class SummaryResult:
    """Result of a summarization operation."""

    summary: str
    source: str  # file path or "inline"
    method: str  # "extractive" or "llm"
    original_length: int
    summary_length: int
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": self.summary,
            "source": self.source,
            "method": self.method,
            "original_length": self.original_length,
            "summary_length": self.summary_length,
            "error": self.error,
        }


def summarize(
    conn: sqlite3.Connection | None,
    *,
    path: str | None = None,
    content: str | None = None,
    style: Literal["brief", "detailed", "bullets"] = "brief",
    method: Literal["extractive", "llm"] = "extractive",
    max_sentences: int | None = None,
    model: str = "qwen3:4b",
) -> SummaryResult:
    """
    Generate a summary of content.

    Args:
        conn: Database connection (required if using path)
        path: Path to file to summarize
        content: Direct content to summarize (alternative to path)
        style: Summary style (brief=3 sentences, detailed=5-7, bullets=key points)
        method: Summarization method (extractive or llm)
        max_sentences: Override default sentence count for extractive
        model: Ollama model for LLM method

    Returns:
        SummaryResult with the summary and metadata
    """
    # Get content to summarize
    if path and conn:
        file_record = get_file_by_path(conn, path)
        if not file_record:
            return SummaryResult(
                summary="",
                source=path,
                method=method,
                original_length=0,
                summary_length=0,
                error=f"File not found: {path}",
            )
        if file_record["is_binary"]:
            return SummaryResult(
                summary="",
                source=path,
                method=method,
                original_length=0,
                summary_length=0,
                error=f"Cannot summarize binary file: {path}",
            )
        text = file_record.get("content", "")
        source = path
    elif content:
        text = content
        source = "inline"
    else:
        return SummaryResult(
            summary="",
            source="",
            method=method,
            original_length=0,
            summary_length=0,
            error="Either path or content must be provided",
        )

    if not text.strip():
        return SummaryResult(
            summary="(empty content)",
            source=source,
            method=method,
            original_length=0,
            summary_length=0,
        )

    # Choose summarization method
    if method == "extractive":
        summary = _extractive_summary(text, style, max_sentences)
    else:
        summary = _llm_summary(text, style, model)

    return SummaryResult(
        summary=summary,
        source=source,
        method=method,
        original_length=len(text),
        summary_length=len(summary),
    )


def _extractive_summary(
    text: str,
    style: Literal["brief", "detailed", "bullets"],
    max_sentences: int | None = None,
) -> str:
    """
    Generate an extractive summary using simple heuristics.

    This extracts the first N sentences, which works surprisingly well
    for many types of content (code comments, documentation, etc.)
    """
    # Determine number of sentences based on style
    if max_sentences is not None:
        n_sentences = max_sentences
    elif style == "brief":
        n_sentences = 3
    elif style == "detailed":
        n_sentences = 7
    else:  # bullets
        n_sentences = 5

    # Split into sentences
    sentences = _split_sentences(text)

    if not sentences:
        return text[:500] + ("..." if len(text) > 500 else "")

    if style == "bullets":
        # For bullets, try to extract key points
        key_sentences = _extract_key_sentences(sentences, n_sentences)
        return "\n".join(f"- {s}" for s in key_sentences)
    else:
        # For brief/detailed, take first N sentences
        selected = sentences[:n_sentences]
        return " ".join(selected)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Normalize whitespace
    text = " ".join(text.split())

    # Split on sentence boundaries
    # This regex handles common cases like "Dr.", "Mr.", etc.
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    raw_sentences = re.split(sentence_pattern, text)

    # Clean up sentences
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) > 10:  # Skip very short fragments
            sentences.append(s)

    return sentences


def _extract_key_sentences(sentences: list[str], n: int) -> list[str]:
    """Extract key sentences using simple heuristics."""
    # Score sentences by various signals
    scored = []
    for i, sentence in enumerate(sentences):
        score = 0

        # First sentences are usually important
        if i < 3:
            score += 10 - i * 3

        # Sentences with keywords are important
        keywords = [
            "important",
            "key",
            "main",
            "primary",
            "essential",
            "note",
            "summary",
            "overview",
            "purpose",
            "goal",
        ]
        for kw in keywords:
            if kw in sentence.lower():
                score += 5

        # Longer sentences often contain more info (but not too long)
        length = len(sentence)
        if 50 < length < 200:
            score += 3
        elif length > 200:
            score += 1

        # Sentences with numbers might be important
        if re.search(r"\d+", sentence):
            score += 2

        scored.append((score, i, sentence))

    # Sort by score descending, then by position
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Take top N and sort by original position
    top_n = sorted(scored[:n], key=lambda x: x[1])

    return [s[2] for s in top_n]


def _llm_summary(
    text: str,
    style: Literal["brief", "detailed", "bullets"],
    model: str,
) -> str:
    """Generate summary using Ollama LLM."""
    try:
        from ollama import chat
    except ImportError:
        return "(LLM summarization requires ollama package: pip install ollama)"

    style_prompts = {
        "brief": "Summarize the following in 2-3 concise sentences:",
        "detailed": "Provide a detailed summary of the following in 5-7 sentences:",
        "bullets": "Summarize the following as 3-5 bullet points with key takeaways:",
    }

    prompt = style_prompts.get(style, style_prompts["brief"])

    # Truncate very long content
    max_chars = 10000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Content truncated...]"

    try:
        response = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates clear, accurate summaries.",
                },
                {"role": "user", "content": f"{prompt}\n\n{text}"},
            ],
        )
        return response.message.content
    except Exception as e:
        return f"(LLM error: {e})"


def format_summary_result(result: SummaryResult) -> str:
    """Format summary result for display."""
    if result.error:
        return f"Error: {result.error}"

    lines = []
    lines.append(f"Source: {result.source}")
    lines.append(f"Method: {result.method}")
    lines.append(
        f"Compression: {result.original_length} -> {result.summary_length} chars "
        f"({100 * result.summary_length / result.original_length:.1f}%)"
        if result.original_length
        else ""
    )
    lines.append("")
    lines.append(result.summary)

    return "\n".join(lines)
