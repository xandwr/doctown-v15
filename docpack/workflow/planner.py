"""
Workflow planner - transforms natural language tasks into execution plans.

The planner uses an LLM to decompose a task into a sequence of tool calls
that can be executed against a docpack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .llm import chat_with_llm, LLMError


@dataclass
class ExecutionStep:
    """A single step in an execution plan."""

    tool: str
    arguments: dict[str, Any]
    output_key: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "output_key": self.output_key,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionStep":
        return cls(
            tool=data["tool"],
            arguments=data.get("arguments", {}),
            output_key=data.get("output_key", "result"),
            description=data.get("description", ""),
        )


@dataclass
class ExecutionPlan:
    """A complete execution plan for a task."""

    task: str
    steps: list[ExecutionStep]
    output_format: str = "json"
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "output_format": self.output_format,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionPlan":
        return cls(
            task=data.get("task", ""),
            steps=[ExecutionStep.from_dict(s) for s in data.get("steps", [])],
            output_format=data.get("output_format", "json"),
            rationale=data.get("rationale", ""),
        )


# System prompt for the planner LLM
PLANNER_SYSTEM_PROMPT = """You are a workflow planner for docpack, a tool that explores frozen codebases and documentation.

Your job is to take a natural language task and decompose it into a sequence of tool calls.

## Available Tools

### Search & Read
- **ls**: List all files. No arguments needed.
- **read**: Read a file. Arguments: {path: string}
- **recall**: Semantic search. Arguments: {query: string, k: int (default 5)}
- **grep**: Text search with context. Arguments: {pattern: string, context_lines: int, path_filter: string, case_sensitive: bool}
- **regex_search**: Regex search. Arguments: {pattern: string, flags: string, path_filter: string}

### Query & Analyze
- **query**: Structured query. Arguments: {type: "files"|"chunks"|"stats", extension: string, path_contains: string, content_contains: string, order_by: string, limit: int}
- **summarize**: Generate summary. Arguments: {path: string, content: string, style: "brief"|"detailed"|"bullets"}

### Marginalia (Notes)
- **write_note**: Save a finding. Arguments: {key: string, content: string}
- **read_notes**: Read saved notes. Arguments: {key: string (optional)}

## Output Format

Return a JSON object with this structure:
{
  "task": "the original task",
  "rationale": "brief explanation of your approach",
  "steps": [
    {
      "tool": "tool_name",
      "arguments": {...},
      "output_key": "result_1",
      "description": "what this step does"
    }
  ],
  "output_format": "json"
}

## Guidelines

1. **Start simple**: Use ls or query first to understand what's in the docpack
2. **Be specific**: Use grep/regex for precise searches, recall for semantic understanding
3. **Chain results**: Each step can build on previous results
4. **Summarize findings**: Use summarize or write_note to consolidate insights
5. **Keep it focused**: 3-7 steps is usually enough

## Example

Task: "Find all TODO comments and summarize them"

{
  "task": "Find all TODO comments and summarize them",
  "rationale": "Search for TODO patterns, then summarize findings",
  "steps": [
    {
      "tool": "grep",
      "arguments": {"pattern": "TODO", "case_sensitive": false},
      "output_key": "todos",
      "description": "Find all TODO comments"
    },
    {
      "tool": "query",
      "arguments": {"type": "stats"},
      "output_key": "stats",
      "description": "Get file statistics for context"
    },
    {
      "tool": "write_note",
      "arguments": {"key": "todo-summary", "content": "Summary of TODOs found"},
      "output_key": "saved",
      "description": "Save summary to marginalia"
    }
  ],
  "output_format": "json"
}
"""


def plan_task(task: str, model: str | None = None) -> ExecutionPlan:
    """
    Use LLM to create an execution plan for a task.

    Args:
        task: Natural language task description
        model: LLM model to use (default: qwen3:4b)

    Returns:
        ExecutionPlan ready for execution
    """
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
    ]

    try:
        response = chat_with_llm(messages, model=model, format="json")
        content = response.content

        # Parse JSON response
        # Handle thinking models that output <think>...</think> before JSON
        if "<think>" in content:
            # Extract content after </think>
            parts = content.split("</think>")
            if len(parts) > 1:
                content = parts[-1].strip()

        # Find JSON in response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            data = json.loads(json_str)
            return ExecutionPlan.from_dict(data)

        raise LLMError(f"Could not parse plan from response: {content[:200]}")

    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON in response: {e}")


def plan_task_simple(task: str) -> ExecutionPlan:
    """
    Create a simple execution plan without LLM.

    Useful as a fallback when LLM is not available.
    """
    # Simple heuristics based on task keywords
    steps = []

    task_lower = task.lower()

    # Always start with ls to understand the structure
    steps.append(
        ExecutionStep(
            tool="ls",
            arguments={},
            output_key="files",
            description="List all files",
        )
    )

    # Add stats for context
    steps.append(
        ExecutionStep(
            tool="query",
            arguments={"type": "stats"},
            output_key="stats",
            description="Get docpack statistics",
        )
    )

    # Look for search-related keywords
    if any(kw in task_lower for kw in ["find", "search", "look for", "grep"]):
        # Try to extract a search term
        # This is very basic - just use the last quoted string or last word
        words = task.split()
        search_term = words[-1] if words else "TODO"
        steps.append(
            ExecutionStep(
                tool="grep",
                arguments={"pattern": search_term},
                output_key="matches",
                description=f"Search for '{search_term}'",
            )
        )

    # Look for semantic search keywords
    if any(kw in task_lower for kw in ["about", "related to", "understand", "explain"]):
        steps.append(
            ExecutionStep(
                tool="recall",
                arguments={"query": task, "k": 5},
                output_key="relevant",
                description="Semantic search for relevant content",
            )
        )

    return ExecutionPlan(
        task=task,
        steps=steps,
        rationale="Fallback plan without LLM",
    )
