"""
Answer generation with citations.

Takes a query, retrieves relevant chunks via semantic search,
and generates a concise, accurate answer with clickable citations.
Only states what is directly supported by the source material.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel

from docpack.recall import recall, RecallResult


# Default model for answer generation - use the 8B model for better reasoning
DEFAULT_ANSWER_MODEL = "qwen3:8b"


class Citation(BaseModel):
    """A citation reference to source material."""

    id: int  # Citation number [1], [2], etc.
    file_path: str
    chunk_index: int
    quote: str  # Brief quote or paraphrase from source
    start_char: int | None = None
    end_char: int | None = None


class AnswerResponse(BaseModel):
    """Structured answer with citations."""

    answer: str  # The answer text with [1], [2] citation markers
    citations: list[Citation]
    confidence: str  # "high", "medium", "low", or "none"


class LLMAnswer(BaseModel):
    """LLM-generated answer structure."""

    answer: str
    used_sources: list[int]  # 0-indexed source numbers that were used
    confidence: str


@dataclass
class AnswerResult:
    """Complete answer with metadata."""

    query: str
    answer: str
    citations: list[Citation]
    confidence: str
    sources_retrieved: int
    sources_used: int


def build_answer_prompt(query: str, sources: list[RecallResult]) -> str:
    """Build the prompt for answer generation."""

    source_texts = []
    for i, src in enumerate(sources):
        # Show the actual text content - truncate if very long
        content = src.text[:800] if len(src.text) > 800 else src.text
        source_texts.append(
            f"[{i}] {src.file_path}\n```\n{content}\n```"
        )

    sources_block = "\n\n".join(source_texts)

    return f"""Answer the question based on the source documents below.

SOURCES:
{sources_block}

QUESTION: {query}

Instructions:
- Answer directly and concisely (1-3 sentences)
- Include the source indices you used in "used_sources" (0-indexed numbers)
- Set confidence: "high" if clearly answered, "medium" if partial, "low" if uncertain
- If the sources genuinely don't help answer the question, set confidence to "none" and explain why"""


def generate_answer(
    conn: sqlite3.Connection,
    query: str,
    *,
    k: int = 8,
    model: str = DEFAULT_ANSWER_MODEL,
    threshold: float = 0.3,
) -> AnswerResult:
    """
    Generate a citation-backed answer to a query.

    Args:
        conn: Database connection to a docpack
        query: Natural language question
        k: Number of sources to retrieve
        model: LLM model for answer generation
        threshold: Minimum similarity score for sources

    Returns:
        AnswerResult with answer, citations, and confidence
    """
    from ollama import chat

    # Step 1: Retrieve relevant chunks
    sources = recall(conn, query, k=k, threshold=threshold)

    if not sources:
        return AnswerResult(
            query=query,
            answer="I couldn't find any relevant information in the loaded documents.",
            citations=[],
            confidence="none",
            sources_retrieved=0,
            sources_used=0,
        )

    # Step 2: Generate answer with LLM
    prompt = build_answer_prompt(query, sources)

    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=LLMAnswer.model_json_schema(),
            options={"temperature": 0.3},
            think=False,  # Disable thinking mode for faster, direct responses
        )

        llm_answer = LLMAnswer.model_validate_json(response.message.content)

    except Exception as e:
        # Fallback: return sources without LLM synthesis
        return AnswerResult(
            query=query,
            answer=f"Error generating answer: {e}. Showing top sources instead.",
            citations=[
                Citation(
                    id=i + 1,
                    file_path=src.file_path,
                    chunk_index=src.chunk_index,
                    quote=src.summary or src.text[:100],
                    start_char=src.start_char,
                    end_char=src.end_char,
                )
                for i, src in enumerate(sources[:3])
            ],
            confidence="none",
            sources_retrieved=len(sources),
            sources_used=0,
        )

    # Step 3: Build citations from used sources
    citations = []
    for i, src_idx in enumerate(llm_answer.used_sources):
        if 0 <= src_idx < len(sources):
            src = sources[src_idx]
            citations.append(
                Citation(
                    id=i + 1,
                    file_path=src.file_path,
                    chunk_index=src.chunk_index,
                    quote=src.summary or src.text[:150],
                    start_char=src.start_char,
                    end_char=src.end_char,
                )
            )

    # Step 4: Format answer with citation numbers
    answer_text = llm_answer.answer

    # If there are citations, add reference markers
    if citations and "[" not in answer_text:
        # Add citation reference at the end if LLM didn't include markers
        refs = ", ".join(f"[{c.id}]" for c in citations)
        answer_text = f"{answer_text} {refs}"

    return AnswerResult(
        query=query,
        answer=answer_text,
        citations=citations,
        confidence=llm_answer.confidence,
        sources_retrieved=len(sources),
        sources_used=len(citations),
    )


def format_answer_plaintext(result: AnswerResult) -> str:
    """
    Format an answer result as clean plaintext with citations.

    Returns a nicely formatted string suitable for CLI or simple display.
    """
    lines = []

    # Main answer
    lines.append(result.answer)
    lines.append("")

    # Citations section
    if result.citations:
        lines.append("Sources:")
        for citation in result.citations:
            # Format: [1] path/to/file.py:chunk_index - "quote preview..."
            quote_preview = citation.quote[:80] + "..." if len(citation.quote) > 80 else citation.quote
            quote_preview = quote_preview.replace("\n", " ")
            lines.append(f"  [{citation.id}] {citation.file_path} - \"{quote_preview}\"")
    else:
        lines.append("(No sources cited)")

    # Confidence indicator
    confidence_icons = {
        "high": "++",
        "medium": "+",
        "low": "~",
        "none": "-",
    }
    icon = confidence_icons.get(result.confidence, "?")
    lines.append(f"\nConfidence: {icon} {result.confidence}")

    return "\n".join(lines)


def format_answer_markdown(result: AnswerResult) -> str:
    """
    Format an answer result as markdown with clickable-style citations.

    Returns markdown suitable for rich display.
    """
    lines = []

    # Main answer
    lines.append(result.answer)
    lines.append("")

    # Citations as a list with file links
    if result.citations:
        lines.append("---")
        lines.append("**Sources:**")
        for citation in result.citations:
            quote_preview = citation.quote[:100] + "..." if len(citation.quote) > 100 else citation.quote
            quote_preview = quote_preview.replace("\n", " ").replace('"', "'")

            # Format with line info if available
            if citation.start_char is not None:
                lines.append(f"- **[{citation.id}]** `{citation.file_path}` (chars {citation.start_char}-{citation.end_char})")
            else:
                lines.append(f"- **[{citation.id}]** `{citation.file_path}` (chunk {citation.chunk_index})")
            lines.append(f"  > {quote_preview}")

    return "\n".join(lines)
