"""
Workflow executor - runs execution plans against a docpack.

The executor takes an execution plan and runs each step, collecting results
and building the final output.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from docpack.storage import (
    init_db,
    read_all_notes,
    read_note,
    start_session,
    end_session,
    save_artifact,
    write_note,
    get_all_files,
    get_file_by_path,
)
from docpack.tools import (
    grep,
    regex_search,
    structured_query,
    summarize,
)
from docpack.recall import recall

from .planner import ExecutionPlan, ExecutionStep, plan_task, plan_task_simple
from .llm import is_llm_available, LLMError


@dataclass
class StepResult:
    """Result of executing a single step."""

    step: ExecutionStep
    success: bool
    output: Any
    error: str | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "step": self.step.to_dict(),
            "success": self.success,
            "output": self.output if isinstance(self.output, (dict, list, str, int, float, bool, type(None))) else str(self.output),
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class TaskResult:
    """Result of executing a complete task."""

    task: str
    status: str  # "completed", "partial", "failed"
    plan: ExecutionPlan
    step_results: list[StepResult]
    output: dict[str, Any]
    session_id: str | None = None
    started_at: str = ""
    completed_at: str = ""
    duration_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "status": self.status,
            "plan": self.plan.to_dict(),
            "step_results": [r.to_dict() for r in self.step_results],
            "output": self.output,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


async def run_task(
    docpack_path: str,
    task: str,
    *,
    model: str | None = None,
    save_artifact_result: bool = True,
) -> TaskResult:
    """
    Execute a natural language task against a docpack.

    This is the main entry point for workflow execution.

    Args:
        docpack_path: Path to the .docpack file
        task: Natural language task description
        model: LLM model for planning (default: qwen3:4b)
        save_artifact_result: Whether to save result as artifact in marginalia

    Returns:
        TaskResult with execution details and output
    """
    path = Path(docpack_path)
    if not path.exists():
        return TaskResult(
            task=task,
            status="failed",
            plan=ExecutionPlan(task=task, steps=[]),
            step_results=[],
            output={},
            error=f"Docpack not found: {docpack_path}",
        )

    conn = init_db(str(path))
    started_at = datetime.utcnow().isoformat()
    start_time = time.monotonic()

    # Start a session
    session = start_session(conn, task=task)

    try:
        # Create execution plan
        if is_llm_available():
            try:
                plan = plan_task(task, model=model)
            except LLMError as e:
                # Fall back to simple plan
                plan = plan_task_simple(task)
                plan.rationale = f"Fallback plan (LLM error: {e})"
        else:
            plan = plan_task_simple(task)
            plan.rationale = "Fallback plan (LLM not available)"

        # Execute plan
        step_results: list[StepResult] = []
        context: dict[str, Any] = {}  # Accumulated results

        for step in plan.steps:
            result = await _execute_step(conn, step, context)
            step_results.append(result)

            if result.success:
                context[step.output_key] = result.output
            else:
                # Continue on error but note it
                context[step.output_key] = None

        # Determine status
        success_count = sum(1 for r in step_results if r.success)
        if success_count == len(step_results):
            status = "completed"
        elif success_count > 0:
            status = "partial"
        else:
            status = "failed"

        completed_at = datetime.utcnow().isoformat()
        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Build output from context
        output = {k: v for k, v in context.items() if v is not None}

        result = TaskResult(
            task=task,
            status=status,
            plan=plan,
            step_results=step_results,
            output=output,
            session_id=session.id,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
        )

        # Save result as artifact
        if save_artifact_result:
            save_artifact(
                conn,
                name=f"task-result-{session.id[:8]}",
                content=result.to_json(),
                content_type="application/json",
                session_id=session.id,
            )

        # End session
        end_session(conn, session.id)

        return result

    except Exception as e:
        end_session(conn, session.id)
        return TaskResult(
            task=task,
            status="failed",
            plan=ExecutionPlan(task=task, steps=[]),
            step_results=[],
            output={},
            session_id=session.id,
            started_at=started_at,
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=int((time.monotonic() - start_time) * 1000),
            error=str(e),
        )
    finally:
        conn.close()


async def _execute_step(
    conn: sqlite3.Connection,
    step: ExecutionStep,
    context: dict[str, Any],
) -> StepResult:
    """Execute a single step."""
    start_time = time.monotonic()

    try:
        output = await _call_tool(conn, step.tool, step.arguments, context)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        return StepResult(
            step=step,
            success=True,
            output=output,
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        return StepResult(
            step=step,
            success=False,
            output=None,
            error=str(e),
            duration_ms=duration_ms,
        )


async def _call_tool(
    conn: sqlite3.Connection,
    tool: str,
    arguments: dict[str, Any],
    context: dict[str, Any],
) -> Any:
    """Call a tool and return its output."""

    # Resolve any context references in arguments
    resolved_args = _resolve_context_refs(arguments, context)

    if tool == "ls":
        files = get_all_files(conn)
        return [
            {"path": f["path"], "size": f["size_bytes"], "binary": f["is_binary"]}
            for f in files
        ]

    elif tool == "read":
        path = resolved_args.get("path", "")
        record = get_file_by_path(conn, path)
        if not record:
            raise ValueError(f"File not found: {path}")
        if record["is_binary"]:
            raise ValueError(f"Cannot read binary file: {path}")
        return record.get("content", "")

    elif tool == "recall":
        query = resolved_args.get("query", "")
        k = resolved_args.get("k", 5)
        results = recall(conn, query, k=k)
        return [
            {
                "file_path": r.file_path,
                "chunk_index": r.chunk_index,
                "text": r.text,
                "score": r.score,
            }
            for r in results
        ]

    elif tool == "grep":
        result = grep(
            conn,
            resolved_args.get("pattern", ""),
            context_lines=resolved_args.get("context_lines", 2),
            path_filter=resolved_args.get("path_filter"),
            case_sensitive=resolved_args.get("case_sensitive", True),
        )
        return result.to_dict()

    elif tool == "regex_search":
        result = regex_search(
            conn,
            resolved_args.get("pattern", ""),
            flags=resolved_args.get("flags", ""),
            path_filter=resolved_args.get("path_filter"),
        )
        return result.to_dict()

    elif tool == "query":
        query_type = resolved_args.get("type", "stats")
        filter_dict = {
            k: v
            for k, v in resolved_args.items()
            if k in ["extension", "path_contains", "content_contains"] and v
        }
        result = structured_query(
            conn,
            query_type,
            filter=filter_dict if filter_dict else None,
            order_by=resolved_args.get("order_by"),
            limit=resolved_args.get("limit", 100),
        )
        return result.to_dict()

    elif tool == "summarize":
        result = summarize(
            conn,
            path=resolved_args.get("path"),
            content=resolved_args.get("content"),
            style=resolved_args.get("style", "brief"),
        )
        return result.to_dict()

    elif tool == "write_note":
        key = resolved_args.get("key", "")
        content = resolved_args.get("content", "")
        # If content references a context variable, expand it
        if isinstance(content, str) and content.startswith("$"):
            var_name = content[1:]
            if var_name in context:
                content = json.dumps(context[var_name], indent=2, default=str)
        note = write_note(conn, key, content)
        return {"key": note.key, "updated_at": note.updated_at}

    elif tool == "read_notes":
        key = resolved_args.get("key")
        if key:
            note = read_note(conn, key)
            if note:
                return {"key": note.key, "content": note.content}
            return None
        notes = read_all_notes(conn)
        return [{"key": n.key, "content": n.content} for n in notes]

    else:
        raise ValueError(f"Unknown tool: {tool}")


def _resolve_context_refs(
    arguments: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Resolve references to context variables in arguments."""
    resolved = {}
    for key, value in arguments.items():
        if isinstance(value, str) and value.startswith("$"):
            var_name = value[1:]
            if var_name in context:
                resolved[key] = context[var_name]
            else:
                resolved[key] = value  # Keep original if not found
        else:
            resolved[key] = value
    return resolved
