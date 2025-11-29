"""
Docpack Workflow Engine.

Transforms natural language tasks into tool execution plans using a local LLM.
This is the magic that makes `docpack run --task "..."` work.

Architecture:
    Task String → [Planner] → Execution Plan → [Executor] → JSON Output
                     ↑                              ↑
              (Local LLM)                    (tool calls)

Usage:
    from docpack.workflow import run_task

    result = await run_task(
        docpack_path="project.docpack",
        task="classify all files by purpose"
    )
    print(result.output)
"""

from .executor import run_task, TaskResult
from .planner import ExecutionPlan, ExecutionStep

__all__ = [
    "run_task",
    "TaskResult",
    "ExecutionPlan",
    "ExecutionStep",
]
