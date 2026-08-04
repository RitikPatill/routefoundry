"""Strict, dependency-free input validation for RouteFoundry observations.

The complete prompt/model matrix is an intentional v0.1 constraint.  It prevents a
missing model result from being mistaken for a bad result and makes every baseline
directly comparable on exactly the same prompts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

OBSERVATION_SCHEMA_VERSION = "routefoundry.observation.v1"

# Resource limits are part of the public parser contract.  They prevent an accidental or
# hostile input from consuming unbounded memory before validation can reject it.
MAX_JSONL_FILE_BYTES = 64 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 1024 * 1024
MAX_JSONL_ROWS = 1_000_000

JsonValue: TypeAlias = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None

_REQUIRED_FIELDS = {"prompt_id", "model", "quality", "latency_ms", "cost_usd"}
_OPTIONAL_FIELDS = {
    "prompt",
    "task",
    "trace_id",
    "sequence_index",
    "load_ms",
    "metadata",
}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


class ValidationError(ValueError):
    """Raised when input cannot support a valid, comparable RouteFoundry audit."""


def _label(context: str | None) -> str:
    return f" ({context})" if context else ""


def _required_string(value: object, name: str, context: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string{_label(context)}")
    if value != value.strip():
        raise ValidationError(
            f"{name} may not have leading or trailing whitespace{_label(context)}"
        )
    return value


def _optional_string(value: object, name: str, context: str | None = None) -> str | None:
    if value is None:
        return None
    return _required_string(value, name, context)


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    context: str | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{name} must be a JSON number{_label(context)}")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite{_label(context)}")
    if number < minimum or (maximum is not None and number > maximum):
        interval = f"[{minimum:g}, {maximum:g}]" if maximum is not None else f">= {minimum:g}"
        raise ValidationError(f"{name} must be {interval}{_label(context)}")
    return number


def _nonnegative_integer(value: object, name: str, context: str | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a non-negative integer{_label(context)}")
    return value


def _validate_json(value: object, path: str = "metadata") -> JsonValue:
    """Return a detached JSON value while rejecting non-finite or exotic values."""

    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_validate_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{path} keys must be strings")
            result[key] = _validate_json(item, f"{path}.{key}")
        return result
    raise ValidationError(f"{path} must contain only JSON-compatible values")


@dataclass(frozen=True, slots=True)
class Observation:
    """A measured result for one prompt/model pair."""

    prompt_id: str
    model: str
    quality: float
    latency_ms: float
    cost_usd: float
    prompt: str | None = None
    task: str | None = None
    trace_id: str | None = None
    sequence_index: int | None = None
    load_ms: float = 0.0
    metadata: Mapping[str, JsonValue] = field(default_factory=dict, compare=True, repr=False)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, context: str | None = None
    ) -> Observation:
        if any(not isinstance(key, str) for key in value):
            raise ValidationError(f"observation field names must be strings{_label(context)}")
        keys = set(value)
        missing = sorted(_REQUIRED_FIELDS - keys)
        unknown = sorted(keys - _ALLOWED_FIELDS)
        if missing:
            raise ValidationError(
                f"missing required field(s) {', '.join(missing)}{_label(context)}"
            )
        if unknown:
            raise ValidationError(f"unknown field(s) {', '.join(unknown)}{_label(context)}")

        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValidationError(f"metadata must be an object{_label(context)}")
        checked_metadata = _validate_json(raw_metadata)
        assert isinstance(checked_metadata, dict)

        has_trace_id = "trace_id" in value
        has_sequence_index = "sequence_index" in value
        if has_trace_id != has_sequence_index:
            raise ValidationError(
                f"trace_id and sequence_index must be supplied together{_label(context)}"
            )
        trace_id = (
            _required_string(value["trace_id"], "trace_id", context) if has_trace_id else None
        )
        sequence_index = (
            _nonnegative_integer(value["sequence_index"], "sequence_index", context)
            if has_sequence_index
            else None
        )

        return cls(
            prompt_id=_required_string(value["prompt_id"], "prompt_id", context),
            model=_required_string(value["model"], "model", context),
            quality=_finite_number(
                value["quality"], "quality", minimum=0.0, maximum=1.0, context=context
            ),
            latency_ms=_finite_number(
                value["latency_ms"], "latency_ms", minimum=0.0, context=context
            ),
            cost_usd=_finite_number(value["cost_usd"], "cost_usd", minimum=0.0, context=context),
            prompt=_optional_string(value.get("prompt"), "prompt", context),
            task=_optional_string(value.get("task"), "task", context),
            trace_id=trace_id,
            sequence_index=sequence_index,
            load_ms=_finite_number(
                value.get("load_ms", 0.0), "load_ms", minimum=0.0, context=context
            ),
            metadata=MappingProxyType(checked_metadata),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "prompt_id": self.prompt_id,
            "model": self.model,
            "quality": self.quality,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }
        if self.prompt is not None:
            result["prompt"] = self.prompt
        if self.task is not None:
            result["task"] = self.task
        if (self.trace_id is None) != (self.sequence_index is None):
            raise ValidationError("trace_id and sequence_index must be supplied together")
        if self.trace_id is not None and self.sequence_index is not None:
            result["trace_id"] = self.trace_id
            result["sequence_index"] = self.sequence_index
        if self.load_ms:
            result["load_ms"] = self.load_ms
        if self.metadata:
            # JSON round-tripping is a small, reliable deep-copy for a JSON-only mapping.
            result["metadata"] = json.loads(
                json.dumps(dict(self.metadata), ensure_ascii=False, allow_nan=False)
            )
        return result


@dataclass(frozen=True, slots=True)
class Dataset:
    """A validated and complete observation matrix."""

    observations: tuple[Observation, ...]
    prompt_ids: tuple[str, ...]
    models: tuple[str, ...]
    _trace_groups: tuple[tuple[str, tuple[str, ...]], ...] = field(default=(), repr=False)
    trace_complete: bool = False

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def prompt_count(self) -> int:
        return len(self.prompt_ids)

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def trace_prompt_ids(self) -> Mapping[str, tuple[str, ...]]:
        """Prompt IDs ordered by ``sequence_index`` within each explicit trace."""

        return MappingProxyType(dict(self._trace_groups))

    @property
    def trace_ordered_prompt_ids(self) -> tuple[str, ...]:
        """Flattened deterministic trace order; traces themselves sort by ID."""

        return tuple(
            prompt_id
            for _, prompt_ids in self._trace_groups
            for prompt_id in prompt_ids
        )

    @property
    def trace_coverage(self) -> float:
        if not self.prompt_ids:
            return 0.0
        traced = sum(len(prompt_ids) for _, prompt_ids in self._trace_groups)
        return traced / len(self.prompt_ids)

    @property
    def trace_completeness(self) -> Mapping[str, bool]:
        """Whether each trace has a gap-free sequence (the first index may be nonzero)."""

        result: dict[str, bool] = {}
        for trace_id, prompt_ids in self._trace_groups:
            indices = [self.for_prompt(prompt_id)[0].sequence_index for prompt_id in prompt_ids]
            assert all(index is not None for index in indices)
            integer_indices = [index for index in indices if index is not None]
            expected = list(
                range(integer_indices[0], integer_indices[0] + len(integer_indices))
            )
            result[trace_id] = integer_indices == expected
        return MappingProxyType(result)

    @property
    def traces_complete(self) -> bool:
        """Compatibility alias for the all-prompts-have-order completeness flag."""

        return self.trace_complete

    @property
    def workload_fingerprint(self) -> str:
        return workload_fingerprint(self)

    def for_prompt(self, prompt_id: str) -> tuple[Observation, ...]:
        return tuple(row for row in self.observations if row.prompt_id == prompt_id)

    def row(self, prompt_id: str, model: str) -> Observation:
        for observation in self.observations:
            if observation.prompt_id == prompt_id and observation.model == model:
                return observation
        raise KeyError((prompt_id, model))

    def subset(self, prompt_ids: Iterable[str]) -> Dataset:
        selected = set(prompt_ids)
        unknown = selected - set(self.prompt_ids)
        if unknown:
            raise KeyError(f"unknown prompt_id(s): {', '.join(sorted(unknown))}")
        if not selected:
            return Dataset((), (), self.models)
        rows = [row for row in self.observations if row.prompt_id in selected]
        return validate_observations(rows)

    def to_records(self) -> list[dict[str, JsonValue]]:
        return [row.to_dict() for row in self.observations]


def _as_observation(value: Observation | Mapping[str, object], index: int) -> Observation:
    if isinstance(value, Observation):
        # Revalidate constructed dataclasses too; callers must not be able to inject NaN.
        try:
            record = value.to_dict()
        except (TypeError, ValueError) as error:
            raise ValidationError(f"invalid Observation object (row {index}): {error}") from error
        return Observation.from_mapping(record, context=f"row {index}")
    if not isinstance(value, Mapping):
        raise ValidationError(f"row {index} must be a JSON object")
    return Observation.from_mapping(value, context=f"row {index}")


def validate_observations(
    rows: Iterable[Observation | Mapping[str, object]],
    *,
    require_signal: bool = False,
    require_complete_matrix: bool = True,
) -> Dataset:
    """Validate observations and return a canonical, order-independent dataset.

    ``require_signal`` is used by compilation: every prompt must have either explicit
    task metadata or prompt text from which a task can be classified.
    """

    observations = [_as_observation(row, index) for index, row in enumerate(rows, start=1)]
    if not observations:
        raise ValidationError("the observation dataset is empty")

    pairs: set[tuple[str, str]] = set()
    prompt_metadata: dict[
        str, tuple[str | None, str | None, str | None, int | None, str]
    ] = {}
    prompt_models: dict[str, set[str]] = {}
    all_models: set[str] = set()
    trace_positions: dict[tuple[str, int], str] = {}

    for row in observations:
        pair = (row.prompt_id, row.model)
        if pair in pairs:
            raise ValidationError(
                f"duplicate observation for prompt_id={row.prompt_id!r}, model={row.model!r}"
            )
        pairs.add(pair)
        all_models.add(row.model)
        prompt_models.setdefault(row.prompt_id, set()).add(row.model)

        metadata_key = json.dumps(
            dict(row.metadata),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        signature = (row.prompt, row.task, row.trace_id, row.sequence_index, metadata_key)
        prior = prompt_metadata.setdefault(row.prompt_id, signature)
        if prior != signature:
            raise ValidationError(
                f"inconsistent prompt/task/trace/metadata for prompt_id={row.prompt_id!r}"
            )
        if row.trace_id is not None:
            assert row.sequence_index is not None
            position = (row.trace_id, row.sequence_index)
            prior_prompt = trace_positions.setdefault(position, row.prompt_id)
            if prior_prompt != row.prompt_id:
                raise ValidationError(
                    f"duplicate sequence_index={row.sequence_index} in trace_id={row.trace_id!r}"
                )
        if require_signal and row.prompt is None and row.task is None:
            raise ValidationError(
                f"prompt_id={row.prompt_id!r} has neither prompt text nor an explicit task"
            )

    if require_complete_matrix:
        for prompt_id, models in prompt_models.items():
            if models != all_models:
                missing = ", ".join(sorted(all_models - models))
                raise ValidationError(
                    f"incomplete prompt/model matrix: prompt_id={prompt_id!r} is missing {missing}"
                )

    canonical = tuple(sorted(observations, key=lambda row: (row.prompt_id, row.model)))
    prompt_heads = {row.prompt_id: row for row in canonical}
    trace_members: dict[str, list[tuple[int, str]]] = {}
    for prompt_id in sorted(prompt_heads):
        row = prompt_heads[prompt_id]
        if row.trace_id is not None:
            assert row.sequence_index is not None
            trace_members.setdefault(row.trace_id, []).append((row.sequence_index, prompt_id))
    trace_groups = tuple(
        (
            trace_id,
            tuple(prompt_id for _, prompt_id in sorted(members)),
        )
        for trace_id, members in sorted(trace_members.items())
    )
    all_prompts_traced = bool(prompt_models) and len(trace_positions) == len(prompt_models)
    dataset = Dataset(
        canonical,
        tuple(sorted(prompt_models)),
        tuple(sorted(all_models)),
        trace_groups,
        False,
    )
    traces_contiguous = all(dataset.trace_completeness.values())
    return Dataset(
        dataset.observations,
        dataset.prompt_ids,
        dataset.models,
        dataset._trace_groups,
        all_prompts_traced and traces_contiguous,
    )


def workload_fingerprint(dataset: Dataset) -> str:
    """Hash workload identity and routing context while excluding all measurements."""

    prompts: list[JsonValue] = []
    prompt_heads: dict[str, Observation] = {}
    for row in dataset:
        prompt_heads.setdefault(row.prompt_id, row)
    for prompt_id in dataset.prompt_ids:
        row = prompt_heads[prompt_id]
        identity = row.prompt if row.prompt is not None else f"prompt_id:{prompt_id}"
        prompts.append(
            {
                "prompt_id": prompt_id,
                "prompt_identity_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                "task": row.task,
                "trace_id": row.trace_id,
                "sequence_index": row.sequence_index,
            }
        )
    manifest: dict[str, JsonValue] = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "models": list(dataset.models),
        "prompts": prompts,
    }
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = item
    return result


def _parse_json_line(line: str, line_number: int) -> Mapping[str, object]:
    if not line.strip():
        raise ValidationError(f"blank JSONL line at line {line_number}")
    try:
        value = json.loads(
            line,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid constant {constant}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValidationError(f"invalid JSON at line {line_number}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValidationError(f"line {line_number} must contain a JSON object")
    return value


def parse_jsonl(text: str, *, require_signal: bool = False) -> Dataset:
    records: list[Mapping[str, object]] = []
    if not text:
        raise ValidationError("the observation dataset is empty")
    try:
        encoded_size = len(text.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValidationError("JSONL text is not valid UTF-8") from error
    if encoded_size > MAX_JSONL_FILE_BYTES:
        raise ValidationError(
            f"JSONL input exceeds the {MAX_JSONL_FILE_BYTES}-byte file limit"
        )
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line_number > MAX_JSONL_ROWS:
            raise ValidationError(f"JSONL input exceeds the {MAX_JSONL_ROWS}-row limit")
        if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
            raise ValidationError(
                f"JSONL line {line_number} exceeds the {MAX_JSONL_LINE_BYTES}-byte line limit"
            )
        records.append(_parse_json_line(line, line_number))
    return validate_observations(records, require_signal=require_signal)


def load_jsonl(path: str | Path, *, require_signal: bool = False) -> Dataset:
    source = Path(path)
    try:
        size = source.stat().st_size
        if size > MAX_JSONL_FILE_BYTES:
            raise ValidationError(
                f"{source} exceeds the {MAX_JSONL_FILE_BYTES}-byte file limit"
            )
        records: list[Mapping[str, object]] = []
        total_bytes = 0
        with source.open("rb") as stream:
            line_number = 0
            while True:
                raw_line = stream.readline(MAX_JSONL_LINE_BYTES + 3)
                if not raw_line:
                    break
                line_number += 1
                if line_number > MAX_JSONL_ROWS:
                    raise ValidationError(
                        f"{source} exceeds the {MAX_JSONL_ROWS}-row limit"
                    )
                total_bytes += len(raw_line)
                if total_bytes > MAX_JSONL_FILE_BYTES:
                    raise ValidationError(
                        f"{source} exceeds the {MAX_JSONL_FILE_BYTES}-byte file limit"
                    )
                payload = raw_line.removesuffix(b"\n").removesuffix(b"\r")
                if len(payload) > MAX_JSONL_LINE_BYTES:
                    raise ValidationError(
                        f"JSONL line {line_number} exceeds the "
                        f"{MAX_JSONL_LINE_BYTES}-byte line limit"
                    )
                try:
                    line = payload.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValidationError(
                        f"{source} is not valid UTF-8 at line {line_number}"
                    ) from error
                records.append(_parse_json_line(line, line_number))
    except ValidationError:
        raise
    except OSError:
        raise
    if not records:
        raise ValidationError("the observation dataset is empty")
    return validate_observations(records, require_signal=require_signal)


def dump_jsonl(dataset: Dataset | Sequence[Observation], path: str | Path) -> None:
    observations = dataset.observations if isinstance(dataset, Dataset) else tuple(dataset)
    validated = validate_observations(observations)
    text = "".join(
        json.dumps(row.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
        for row in validated
    )
    Path(path).write_text(text, encoding="utf-8")
