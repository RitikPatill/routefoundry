"""The bundled auto-gradable task suite and its loader.

A task is only included when a deterministic grader can score it, because the whole point
of autopilot is to produce a routing matrix from measurement rather than from an LLM judge
whose own bias would then need auditing.

That constraint has a consequence worth stating plainly wherever results are shown: this
suite measures **verifiable short-answer ability**, not open-ended generation quality.
Conclusions drawn from it apply to workloads that look like it.

Difficulty is deliberately spread. If every model scores identically there is nothing for a
router to learn, so the suite mixes items a 1.5B model handles with items that defeat it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Final

from routefoundry.graders import GRADER_NAMES

SUITE_VERSION: Final = "routefoundry.tasks.v1"
BUNDLED_SUITE_FILE: Final = "suite_v1.jsonl"
MAX_SUITE_BYTES: Final = 4 * 1024 * 1024
MAX_PROMPT_CHARS: Final = 2_000


class TaskSuiteError(ValueError):
    """Raised when a task suite file cannot be used as written."""


@dataclass(frozen=True, slots=True)
class Task:
    """One auto-gradable prompt."""

    id: str
    task: str
    prompt: str
    expected: str
    grader: str
    grader_arg: str | None = None

    @property
    def num_predict(self) -> int:
        """Token budget for this task.

        Reasoning models spend most of their output on a scratchpad before answering, so a
        budget tuned for a one-word answer would truncate them mid-thought and score a
        capable model as wrong. The ceiling is generous for that reason; it exists only to
        stop a degenerate repetition loop from stalling a run.
        """

        return 800


def _require_str(row: dict[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskSuiteError(f"{key} must be a non-empty string ({context})")
    return value.strip()


def parse_task(row: dict[str, Any], context: str) -> Task:
    """Validate one suite row, failing closed on anything ambiguous."""

    if not isinstance(row, dict):
        raise TaskSuiteError(f"task must be a JSON object ({context})")

    unknown = set(row) - {"id", "task", "prompt", "expected", "grader", "grader_arg"}
    if unknown:
        raise TaskSuiteError(f"unknown task fields {sorted(unknown)} ({context})")

    grader = _require_str(row, "grader", context)
    if grader not in GRADER_NAMES:
        raise TaskSuiteError(f"unknown grader {grader!r} ({context})")

    prompt = _require_str(row, "prompt", context)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise TaskSuiteError(f"prompt exceeds {MAX_PROMPT_CHARS} characters ({context})")

    grader_arg = row.get("grader_arg")
    if grader_arg is not None and not isinstance(grader_arg, str):
        raise TaskSuiteError(f"grader_arg must be a string when present ({context})")

    return Task(
        id=_require_str(row, "id", context),
        task=_require_str(row, "task", context),
        prompt=prompt,
        expected=_require_str(row, "expected", context),
        grader=grader,
        grader_arg=grader_arg.strip() if isinstance(grader_arg, str) and grader_arg.strip() else None,
    )


def _iter_rows(text: str, source: str) -> Iterator[tuple[dict[str, Any], str]]:
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        context = f"{source} line {number}"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise TaskSuiteError(f"invalid JSON ({context}): {error.msg}") from None
        if not isinstance(parsed, dict):
            raise TaskSuiteError(f"task must be a JSON object ({context})")
        yield parsed, context


def load_tasks(path: str | Path | None = None) -> tuple[Task, ...]:
    """Load the bundled suite, or a caller-supplied JSONL file of the same shape."""

    if path is None:
        text = resources.files("routefoundry.data").joinpath(BUNDLED_SUITE_FILE).read_text("utf-8")
        source = f"bundled {BUNDLED_SUITE_FILE}"
    else:
        file_path = Path(path)
        size = file_path.stat().st_size
        if size > MAX_SUITE_BYTES:
            raise TaskSuiteError(f"task suite exceeds {MAX_SUITE_BYTES} bytes")
        text = file_path.read_text("utf-8")
        source = str(file_path)

    tasks = tuple(parse_task(row, context) for row, context in _iter_rows(text, source))
    if not tasks:
        raise TaskSuiteError(f"no tasks found in {source}")

    identifiers = [task.id for task in tasks]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        raise TaskSuiteError(f"duplicate task ids in {source}: {', '.join(duplicates)}")
    return tasks


def select_tasks(tasks: Sequence[Task], *, limit: int | None = None) -> tuple[Task, ...]:
    """Take a balanced subset, round-robin across task types.

    Truncating the suite in file order would silently drop whole categories and make a
    quick run unrepresentative, so selection rotates through task types instead.
    """

    if limit is None:
        return tuple(tasks)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise TaskSuiteError("limit must be a positive integer")

    by_type: dict[str, list[Task]] = {}
    for task in tasks:
        by_type.setdefault(task.task, []).append(task)

    selected: list[Task] = []
    while len(selected) < limit:
        added = False
        for bucket in by_type.values():
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return tuple(selected)


def suite_summary(tasks: Sequence[Task]) -> dict[str, int]:
    """Count tasks per task type, for run previews and reports."""

    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.task] = counts.get(task.task, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "BUNDLED_SUITE_FILE",
    "MAX_PROMPT_CHARS",
    "SUITE_VERSION",
    "Task",
    "TaskSuiteError",
    "load_tasks",
    "parse_task",
    "select_tasks",
    "suite_summary",
]
