from __future__ import annotations

import json

import pytest
import routefoundry.schema as schema_module
from routefoundry.schema import (
    Observation,
    ValidationError,
    load_jsonl,
    parse_jsonl,
    validate_observations,
    workload_fingerprint,
)
from routefoundry.split import split_observations, stable_split


def record(
    prompt_id: str = "p1",
    model: str = "small",
    **overrides: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "prompt_id": prompt_id,
        "prompt": "Summarize the support ticket.",
        "task": "summarization",
        "model": model,
        "quality": 0.9,
        "latency_ms": 120,
        "cost_usd": 0.01,
        "load_ms": 25,
        "metadata": {"source": "unit-test"},
    }
    value.update(overrides)
    return value


def matrix(prompt_ids: tuple[str, ...] = ("p1", "p2", "p3")) -> list[dict[str, object]]:
    return [record(prompt_id, model) for prompt_id in prompt_ids for model in ("small", "large")]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quality", -0.1),
        ("quality", 1.1),
        ("quality", float("nan")),
        ("latency_ms", float("inf")),
        ("latency_ms", -1),
        ("cost_usd", -0.01),
        ("load_ms", float("-inf")),
        ("quality", True),
        ("cost_usd", "0.1"),
    ],
)
def test_rejects_invalid_numeric_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        validate_observations([record(**{field: value})])


def test_rejects_missing_unknown_and_blank_fields() -> None:
    missing = record()
    del missing["model"]
    with pytest.raises(ValidationError, match=r"missing required.*model"):
        validate_observations([missing])

    with pytest.raises(ValidationError, match=r"unknown field.*secret"):
        validate_observations([record(secret="do-not-accept")])

    with pytest.raises(ValidationError, match="prompt_id must be a non-empty"):
        validate_observations([record(prompt_id="")])


def test_rejects_duplicate_pair_and_incomplete_matrix() -> None:
    with pytest.raises(ValidationError, match="duplicate observation"):
        validate_observations([record(), record()])

    incomplete = matrix(("p1", "p2"))
    incomplete.pop()
    with pytest.raises(ValidationError, match="incomplete prompt/model matrix"):
        validate_observations(incomplete)


@pytest.mark.parametrize(
    "changed",
    [
        {"prompt": "Different text"},
        {"task": "classification"},
        {"metadata": {"source": "different"}},
    ],
)
def test_rejects_inconsistent_prompt_metadata(changed: dict[str, object]) -> None:
    rows = [record(model="small"), record(model="large", **changed)]
    with pytest.raises(ValidationError, match="inconsistent prompt/task/trace/metadata"):
        validate_observations(rows)


def test_trace_fields_are_paired_consistent_and_unique() -> None:
    with pytest.raises(ValidationError, match="must be supplied together"):
        validate_observations([record(trace_id="trace-a")])
    with pytest.raises(ValidationError, match="non-negative integer"):
        validate_observations([record(trace_id="trace-a", sequence_index=True)])

    inconsistent = [
        record(model="small", trace_id="trace-a", sequence_index=0),
        record(model="large", trace_id="trace-a", sequence_index=1),
    ]
    with pytest.raises(ValidationError, match="inconsistent prompt/task/trace/metadata"):
        validate_observations(inconsistent)

    collision = [
        record("p1", "small", trace_id="trace-a", sequence_index=0),
        record("p1", "large", trace_id="trace-a", sequence_index=0),
        record("p2", "small", trace_id="trace-a", sequence_index=0),
        record("p2", "large", trace_id="trace-a", sequence_index=0),
    ]
    with pytest.raises(ValidationError, match="duplicate sequence_index=0"):
        validate_observations(collision)


def test_dataset_exposes_trace_order_and_completeness() -> None:
    rows = [
        record("p2", model, trace_id="trace-a", sequence_index=1)
        for model in ("small", "large")
    ] + [
        record("p1", model, trace_id="trace-a", sequence_index=0)
        for model in ("small", "large")
    ]
    complete = validate_observations(rows)
    assert complete.trace_complete
    assert complete.traces_complete
    assert complete.trace_coverage == 1.0
    assert complete.trace_completeness == {"trace-a": True}
    assert complete.trace_prompt_ids["trace-a"] == ("p1", "p2")
    assert complete.trace_ordered_prompt_ids == ("p1", "p2")

    partial_rows = rows + [record("p3", model) for model in ("small", "large")]
    partial = validate_observations(partial_rows)
    assert not partial.trace_complete
    assert partial.trace_coverage == pytest.approx(2 / 3)

    gapped = validate_observations(
        [
            record(prompt_id, model, trace_id="trace-a", sequence_index=sequence_index)
            for prompt_id, sequence_index in (("p1", 0), ("p2", 2))
            for model in ("small", "large")
        ]
    )
    assert not gapped.trace_complete
    assert gapped.trace_completeness == {"trace-a": False}


def test_compile_signal_requirement_is_explicit() -> None:
    rows = [record(prompt=None, task=None)]
    assert validate_observations(rows).prompt_count == 1
    with pytest.raises(ValidationError, match="neither prompt text nor an explicit task"):
        validate_observations(rows, require_signal=True)


def test_jsonl_is_utf8_strict_and_canonical(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = matrix()
    source = tmp_path / "results.jsonl"
    source.write_text("\n".join(json.dumps(item) for item in reversed(rows)), encoding="utf-8")
    dataset = load_jsonl(source)
    assert dataset.prompt_ids == ("p1", "p2", "p3")
    assert dataset.models == ("large", "small")
    assert [(row.prompt_id, row.model) for row in dataset] == sorted(
        (row["prompt_id"], row["model"]) for row in rows
    )

    with pytest.raises(ValidationError, match="blank JSONL line"):
        parse_jsonl(json.dumps(record()) + "\n\n" + json.dumps(record("p2")))
    with pytest.raises(ValidationError, match="invalid JSON"):
        parse_jsonl("{not-json}\n")
    with pytest.raises(ValidationError, match="duplicate JSON object key"):
        parse_jsonl(
            '{"prompt_id":"one","prompt_id":"two","model":"m","quality":1,'
            '"latency_ms":1,"cost_usd":0}'
        )


def test_metadata_must_be_json_and_finite() -> None:
    with pytest.raises(ValidationError, match="JSON-compatible"):
        validate_observations([record(metadata={"bad": object()})])
    with pytest.raises(ValidationError, match="non-finite"):
        validate_observations([record(metadata={"bad": float("nan")})])


def test_manual_observation_cannot_bypass_validation() -> None:
    unsafe = Observation("p1", "m1", float("nan"), 1.0, 0.0)
    with pytest.raises(ValidationError, match="quality must be finite"):
        validate_observations([unsafe])


def test_workload_fingerprint_excludes_measurements_but_covers_identity() -> None:
    rows = [
        record("p1", model, trace_id="trace-a", sequence_index=0)
        for model in ("small", "large")
    ]
    original = validate_observations(rows)
    changed_measurements = [
        {
            **row,
            "quality": 0.1,
            "latency_ms": 9999,
            "cost_usd": 99,
            "load_ms": 888,
        }
        for row in rows
    ]
    measured = validate_observations(changed_measurements)
    assert workload_fingerprint(original) == workload_fingerprint(measured)
    assert original.workload_fingerprint.startswith("sha256:")

    changed_prompt = [{**row, "prompt": "A different workload prompt."} for row in rows]
    assert (
        validate_observations(changed_prompt).workload_fingerprint
        != original.workload_fingerprint
    )

    extra_model = [
        *rows,
        record("p1", "third", trace_id="trace-a", sequence_index=0),
    ]
    assert validate_observations(extra_model).workload_fingerprint != original.workload_fingerprint

    changed_context = [
        {**row, "task": "different-task", "sequence_index": 5} for row in rows
    ]
    assert (
        validate_observations(changed_context).workload_fingerprint
        != original.workload_fingerprint
    )


def test_jsonl_resource_limits_fail_closed(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    line = json.dumps(record())

    monkeypatch.setattr(schema_module, "MAX_JSONL_ROWS", 1)
    with pytest.raises(ValidationError, match="row limit"):
        parse_jsonl(line + "\n" + json.dumps(record("p2")))

    monkeypatch.setattr(schema_module, "MAX_JSONL_ROWS", 100)
    monkeypatch.setattr(schema_module, "MAX_JSONL_LINE_BYTES", 10)
    with pytest.raises(ValidationError, match="line limit"):
        parse_jsonl(line)

    source = tmp_path / "oversized.jsonl"
    source.write_text(line, encoding="utf-8")
    monkeypatch.setattr(schema_module, "MAX_JSONL_LINE_BYTES", 1024 * 1024)
    monkeypatch.setattr(schema_module, "MAX_JSONL_FILE_BYTES", 10)
    with pytest.raises(ValidationError, match="file limit"):
        load_jsonl(source)

    monkeypatch.setattr(schema_module, "MAX_JSONL_FILE_BYTES", 1024 * 1024)
    monkeypatch.setattr(schema_module, "MAX_JSONL_LINE_BYTES", 10)
    with pytest.raises(ValidationError, match="line limit"):
        load_jsonl(source)

    two_rows = tmp_path / "two-rows.jsonl"
    two_rows.write_text(line + "\n" + json.dumps(record("p2")), encoding="utf-8")
    monkeypatch.setattr(schema_module, "MAX_JSONL_LINE_BYTES", 1024 * 1024)
    monkeypatch.setattr(schema_module, "MAX_JSONL_ROWS", 1)
    with pytest.raises(ValidationError, match="row limit"):
        load_jsonl(two_rows)


def test_split_is_prompt_level_order_independent_and_nonempty() -> None:
    dataset = validate_observations(matrix(tuple(f"p{i}" for i in range(12))))
    shuffled = validate_observations(reversed(dataset.observations))
    first = split_observations(dataset, seed=7)
    second = split_observations(shuffled, seed=7)
    assert first.to_dict() == second.to_dict()
    assert first.train.prompt_ids and first.dev.prompt_ids and first.test.prompt_ids

    partitions = [
        set(first.train.prompt_ids),
        set(first.dev.prompt_ids),
        set(first.test.prompt_ids),
    ]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert set.union(*partitions) == set(dataset.prompt_ids)
    assert stable_split("fixed-id", seed=99) == stable_split("fixed-id", seed=99)


def test_complete_traces_are_stable_indivisible_split_units() -> None:
    rows = [
        record(
            f"prompt-{trace_index}-{sequence_index}",
            model,
            trace_id=f"trace-{trace_index}",
            sequence_index=sequence_index,
        )
        for trace_index in range(9)
        for sequence_index in range(2)
        for model in ("small", "large")
    ]
    dataset = validate_observations(rows)
    first = split_observations(dataset, seed=19)
    second = split_observations(
        validate_observations(reversed(dataset.observations)), seed=19
    )
    assert first.split_unit == "trace"
    assert first.to_dict() == second.to_dict()

    partition_traces = [
        set(first.train.trace_prompt_ids),
        set(first.dev.trace_prompt_ids),
        set(first.test.trace_prompt_ids),
    ]
    assert not partition_traces[0] & partition_traces[1]
    assert not partition_traces[0] & partition_traces[2]
    assert not partition_traces[1] & partition_traces[2]
    assert set.union(*partition_traces) == set(dataset.trace_prompt_ids)
    serialized = first.to_dict()
    assert serialized["trace_counts"] == {
        "train": len(partition_traces[0]),
        "development": len(partition_traces[1]),
        "test": len(partition_traces[2]),
    }
    assert "trace-0" not in json.dumps(serialized, sort_keys=True)
