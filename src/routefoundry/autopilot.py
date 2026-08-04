"""Generate a routing observation matrix by measuring locally installed models.

RouteFoundry's audit needs a complete prompt x model matrix of graded results. Producing
one by hand is the reason most people never get to run an audit at all, so autopilot builds
it directly: discover installed Ollama models, run the bundled auto-gradable suite against
each, grade deterministically, and write observations the existing pipeline already accepts.

Three properties are dictated by measurements on consumer hardware rather than by taste:

* **Work is grouped by model.** A cold load costs 7-14 s on a 16 GB laptop, so per-prompt
  model switching would spend most of a run loading weights instead of answering.
* **Runs are resumable.** A complete suite across a large fleet takes tens of minutes; a
  crash or an interrupt must not discard finished work, so rows are appended and re-loaded.
* **Progress is reported per prompt.** A silent multi-minute command is indistinguishable
  from a hung one.

Honesty constraints carried through to the output: timings are backend-reported and
OS-cache-uncontrolled, so they are never labelled cold-start; quality is verifiable
short-answer correctness, not open-ended generation quality; and a failed or timed-out
generation is recorded as an explicit error rather than silently scored zero, because a
timeout is a property of the run, not of the model's ability.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
import psutil  # type: ignore[import-untyped]

from routefoundry.graders import grade
from routefoundry.ollama import DEFAULT_BASE_URL, OllamaAPIError, OllamaProfiler
from routefoundry.tasks import SUITE_VERSION, Task

RUN_SCHEMA_VERSION: Final = "routefoundry.autopilot.v1"
TRACE_ID: Final = "autopilot-suite"
DEFAULT_TIMEOUT_SECONDS: Final = 300.0
DEFAULT_QUICK_LIMIT: Final = 12

# Local execution has no per-token price. Recording 0.0 keeps the schema honest: the audit
# then compares latency and quality, and never invents a monetary saving that did not occur.
LOCAL_COST_USD: Final = 0.0


class AutopilotError(RuntimeError):
    """Raised when a run cannot produce a usable matrix."""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """An installed model as reported by Ollama."""

    name: str
    size_bytes: int | None
    parameter_size: str | None
    quantization: str | None

    @property
    def size_gb(self) -> float | None:
        return None if self.size_bytes is None else round(self.size_bytes / 1e9, 2)


@dataclass(slots=True)
class RunProgress:
    """A single completed measurement, for live reporting."""

    model: str
    task_id: str
    index: int
    total: int
    correct: bool
    latency_ms: float
    error: str | None = None


@dataclass(slots=True)
class RunStats:
    """Totals accumulated across a run."""

    measured: int = 0
    resumed: int = 0
    errors: int = 0
    wall_seconds: float = 0.0
    dropped_prompts: tuple[str, ...] = ()
    per_model_correct: dict[str, int] = field(default_factory=dict)
    per_model_total: dict[str, int] = field(default_factory=dict)


def discover_models(*, base_url: str = DEFAULT_BASE_URL) -> tuple[ModelInfo, ...]:
    """List installed models. Never pulls: a run measures what is already present."""

    with OllamaProfiler(base_url=base_url) as profiler:
        tags = profiler.list_tags()

    models: list[ModelInfo] = []
    for tag in tags:
        name = tag.get("name") or tag.get("model")
        if not isinstance(name, str) or not name:
            continue
        details = tag.get("details") if isinstance(tag.get("details"), dict) else {}
        size = tag.get("size")
        models.append(
            ModelInfo(
                name=name,
                size_bytes=size if isinstance(size, int) and not isinstance(size, bool) else None,
                parameter_size=details.get("parameter_size"),
                quantization=details.get("quantization_level"),
            )
        )
    return tuple(sorted(models, key=lambda item: item.name))


def estimate_duration_seconds(model_count: int, task_count: int) -> float:
    """Rough pre-run estimate so a user can decide before committing minutes to a run.

    The constants come from measurements on a 16 GB laptop CPU (~15 s per generation,
    ~10 s per model load). They are a planning aid, not a promise: a GPU fleet is far
    faster and a large reasoning model is slower.
    """

    per_generation = 15.0
    per_model_load = 10.0
    return model_count * (per_model_load + task_count * per_generation)


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    return f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.1f} h"


class _Generator:
    """Minimal Ollama generation client scoped to what a measured run needs."""

    def __init__(self, base_url: str, timeout: float) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, trust_env=False)

    def __enter__(self) -> _Generator:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._client.close()

    def unload(self, model: str) -> None:
        """Evict a model so the next call measures a load rather than a warm hit."""

        try:
            self._client.post("/api/generate", json={"model": model, "keep_alive": 0, "stream": False})
        except httpx.HTTPError:
            # Eviction is an optimisation for measurement fidelity, not a correctness
            # requirement; a failure here only means the next load time is a warm one.
            pass

    def generate(self, model: str, prompt: str, num_predict: int) -> dict[str, Any]:
        response = self._client.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0, "seed": 0, "num_predict": num_predict},
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
            raise AutopilotError(f"Ollama returned an unusable response for model {model!r}")
        return payload


@dataclass(frozen=True, slots=True)
class Trial:
    """One measured (model, task) pair, before it becomes an observation row."""

    task_id: str
    model: str
    quality: float
    correct: bool
    generation_ms: float
    load_ms: float
    grade_reason: str
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "quality": self.quality,
            "correct": self.correct,
            "generation_ms": round(self.generation_ms, 3),
            "load_ms": round(self.load_ms, 3),
            "grade_reason": self.grade_reason,
            "error": self.error,
        }


class ConcurrentRunError(AutopilotError):
    """Raised when another run is already measuring into the same output."""


@contextmanager
def _run_lock(output_path: Path) -> Iterator[None]:
    """Refuse to start when another live run owns this output.

    Two concurrent runs interleave writes *and* compete for the same CPU, so every latency
    they record is inflated. That silently produces plausible-looking numbers that are not
    evidence of anything, which is worse than crashing.
    """

    lock = output_path.with_suffix(output_path.suffix + ".lock")
    if lock.exists():
        try:
            owner = int(lock.read_text("utf-8").strip() or "0")
        except (OSError, ValueError):
            owner = 0
        if owner and owner != os.getpid() and psutil.pid_exists(owner):
            raise ConcurrentRunError(
                f"another autopilot run (pid {owner}) is writing to {output_path.name}. "
                "Concurrent runs corrupt timings; wait for it, or choose a different --output."
            )
        lock.unlink(missing_ok=True)  # stale lock from a killed run

    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def trials_path_for(output_path: str | Path) -> Path:
    """Sidecar path holding per-(model, task) detail.

    Per-model detail cannot live in an observation's ``metadata``: the validator treats
    metadata as a property of the *prompt* and rejects a dataset whose rows disagree.
    """

    path = Path(output_path)
    return path.with_name(path.stem + ".trials.jsonl")


def run_autopilot(
    models: Sequence[str],
    tasks: Sequence[Task],
    output_path: str | Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    resume: bool = True,
    on_progress: Callable[[RunProgress], None] | None = None,
) -> RunStats:
    """Measure every (model, task) pair, then write a validated observation matrix.

    Trials are appended to a sidecar as they complete, so an interrupt leaves finished work
    intact and a later invocation resumes from it. The observation matrix is derived at the
    end because two of its fields are only knowable once a model's whole set is measured.
    """

    if not models:
        raise AutopilotError("at least one model is required")
    if not tasks:
        raise AutopilotError("at least one task is required")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trials_path = trials_path_for(path)
    completed = _load_trials(trials_path) if resume else {}

    stats = RunStats()
    started = time.perf_counter()
    total = len(models) * len(tasks)
    index = 0
    trials: list[Trial] = []

    with trials_path.open("a", encoding="utf-8", newline="\n") as sidecar:
        with _Generator(base_url, timeout) as generator:
            for model in models:
                stats.per_model_total.setdefault(model, 0)
                stats.per_model_correct.setdefault(model, 0)
                if any((task.id, model) not in completed for task in tasks):
                    # One eviction per model, not per prompt: the first generation then
                    # carries an honest load cost and the rest measure steady state.
                    generator.unload(model)

                for task in tasks:
                    index += 1
                    cached = completed.get((task.id, model))
                    if cached is not None:
                        trials.append(cached)
                        stats.resumed += 1
                    else:
                        trial = _measure_one(generator, model, task)
                        trials.append(trial)
                        sidecar.write(json.dumps(trial.as_json(), sort_keys=True) + "\n")
                        sidecar.flush()
                        stats.measured += 1
                        if trial.error:
                            stats.errors += 1
                        if on_progress is not None:
                            on_progress(
                                RunProgress(
                                    model,
                                    task.id,
                                    index,
                                    total,
                                    trial.correct,
                                    trial.generation_ms,
                                    trial.error,
                                )
                            )
                        cached = trial

                    stats.per_model_total[model] += 1
                    if cached.correct:
                        stats.per_model_correct[model] += 1

    stats.dropped_prompts = write_observations(trials, tasks, models, path)
    stats.wall_seconds = time.perf_counter() - started
    return stats


def _measure_one(generator: _Generator, model: str, task: Task) -> Trial:
    """Run and grade one pair."""

    try:
        payload = generator.generate(model, task.prompt, task.num_predict)
    except (httpx.HTTPError, OllamaAPIError, AutopilotError) as error:
        return Trial(
            task_id=task.id,
            model=model,
            quality=0.0,
            correct=False,
            generation_ms=0.0,
            load_ms=0.0,
            grade_reason="not graded",
            error=type(error).__name__,
        )

    total_ms = float(payload.get("total_duration", 0)) / 1e6
    load_ms = float(payload.get("load_duration", 0)) / 1e6
    result = grade(payload["response"], task.expected, task.grader, task.grader_arg)
    return Trial(
        task_id=task.id,
        model=model,
        quality=float(result.score),
        correct=result.correct,
        # Load is reported separately and must not be double-counted inside latency: the
        # audit adds a model's load cost only when routing actually switches models.
        generation_ms=max(0.0, total_ms - load_ms),
        load_ms=load_ms,
        grade_reason=result.reason,
    )


def write_observations(
    trials: Sequence[Trial],
    tasks: Sequence[Task],
    models: Sequence[str],
    output_path: str | Path,
) -> tuple[str, ...]:
    """Derive and write the observation matrix. Returns the prompt ids that were dropped.

    Two rules make the result trustworthy rather than merely well-formed:

    * A prompt is emitted only when **every** model produced a graded result. A timeout is
      a property of the run, not evidence that a model cannot answer, so scoring it zero
      would understate that model. Incomplete prompts are dropped and reported instead.
    * ``load_ms`` is the model's cost to *become resident* -- the cold load observed once
      per model -- replicated across that model's rows. Using each call's own value would
      put ~0 on every warm call and silently erase the switch penalty from the audit.
    """

    by_prompt: dict[str, dict[str, Trial]] = {}
    for trial in trials:
        by_prompt.setdefault(trial.task_id, {})[trial.model] = trial

    cold_load_ms: dict[str, float] = {}
    for model in models:
        observed = [t.load_ms for t in trials if t.model == model and t.load_ms > 0]
        cold_load_ms[model] = max(observed) if observed else 0.0

    usable, dropped = [], []
    for task in tasks:
        row_set = by_prompt.get(task.id, {})
        if all(model in row_set and row_set[model].error is None for model in models):
            usable.append(task)
        else:
            dropped.append(task.id)

    path = Path(output_path)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        # sequence_index must stay contiguous within the trace, so positions are assigned
        # after dropping rather than carried over from the original suite order.
        for position, task in enumerate(usable):
            for model in models:
                trial = by_prompt[task.id][model]
                handle.write(
                    json.dumps(
                        {
                            "prompt_id": task.id,
                            "model": model,
                            "prompt": task.prompt,
                            "task": task.task,
                            "quality": trial.quality,
                            "latency_ms": round(trial.generation_ms, 3),
                            "load_ms": round(cold_load_ms[model], 3),
                            "cost_usd": LOCAL_COST_USD,
                            "trace_id": TRACE_ID,
                            "sequence_index": position,
                            # Prompt-scoped only: the validator requires every row of a
                            # prompt to agree, so per-model detail lives in the sidecar.
                            "metadata": {"grader": task.grader, "suite": SUITE_VERSION},
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return tuple(dropped)


def _load_trials(path: Path) -> dict[tuple[str, str], Trial]:
    """Read completed trials so an interrupted run can resume."""

    if not path.exists():
        return {}
    resumed: dict[tuple[str, str], Trial] = {}
    for line in path.read_text("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            # A partial final line is expected after an abrupt interrupt.
            continue
        if not isinstance(row, dict) or not row.get("task_id") or not row.get("model"):
            continue
        if row.get("error"):
            continue  # retry failures rather than resuming them
        resumed[(str(row["task_id"]), str(row["model"]))] = Trial(
            task_id=str(row["task_id"]),
            model=str(row["model"]),
            quality=float(row.get("quality", 0.0)),
            correct=bool(row.get("correct", False)),
            generation_ms=float(row.get("generation_ms", 0.0)),
            load_ms=float(row.get("load_ms", 0.0)),
            grade_reason=str(row.get("grade_reason", "")),
        )
    return resumed


def iter_observation_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield rows from a written matrix."""

    for line in Path(path).read_text("utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            yield json.loads(stripped)


__all__ = [
    "ConcurrentRunError",
    "Trial",
    "trials_path_for",
    "write_observations",
    "DEFAULT_QUICK_LIMIT",
    "DEFAULT_TIMEOUT_SECONDS",
    "RUN_SCHEMA_VERSION",
    "AutopilotError",
    "ModelInfo",
    "RunProgress",
    "RunStats",
    "discover_models",
    "estimate_duration_seconds",
    "format_duration",
    "iter_observation_rows",
    "run_autopilot",
]
