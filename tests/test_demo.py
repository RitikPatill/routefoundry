from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from routefoundry.demo import (
    DEMO_DATA_LABEL,
    DEMO_MODELS,
    audit_demo,
    demo_manifest,
    generate_demo_observations,
    make_demo_dataset,
    summarize_demo_decisions,
    write_demo_jsonl,
)
from routefoundry.split import split_observations


def test_demo_is_deterministic_complete_and_explicitly_non_evidence() -> None:
    first = generate_demo_observations()
    second = generate_demo_observations()

    assert first == second
    assert len(first) == 8 * 15 * 3
    assert len({row["task"] for row in first}) == 8
    assert {row["model"] for row in first} == set(DEMO_MODELS)
    assert all(row["metadata"]["data_origin"] == "synthetic" for row in first)
    assert all("non_evidence" in row["metadata"]["evidence_status"] for row in first)
    assert demo_manifest()["empirical_evidence"] is False
    assert "NON-EVIDENCE" in DEMO_DATA_LABEL
    assert all(isinstance(row["trace_id"], str) for row in first)
    assert all(isinstance(row["sequence_index"], int) for row in first)


def test_demo_has_enough_prompt_groups_for_stable_splits() -> None:
    dataset = make_demo_dataset()
    split = split_observations(dataset)
    records = generate_demo_observations()

    assert dataset.prompt_count == 120
    assert dataset.model_count == 3
    assert dataset.trace_complete
    assert len(dataset.trace_prompt_ids) == 15
    assert split.split_unit == "trace"
    task_traces = {
        row["task"]: {
            candidate["trace_id"]
            for candidate in records
            if candidate["task"] == row["task"]
        }
        for row in records
    }
    assert all(len(trace_ids) == 15 for trace_ids in task_traces.values())
    assert min(split.train.prompt_count, split.dev.prompt_count, split.test.prompt_count) > 10


def test_demo_budget_changes_at_least_one_explainable_decision() -> None:
    strict = summarize_demo_decisions(max_quality_loss=0.0, objective="cost")
    relaxed = summarize_demo_decisions(max_quality_loss=0.05, objective="cost")

    assert strict["provenance"]["synthetic"] is True
    assert strict["runtime_note"].startswith("Offline")
    assert any(
        strict["decisions"][task]["selected_model"] != relaxed["decisions"][task]["selected_model"]
        for task in strict["decisions"]
    )


def test_demo_production_audit_preserves_non_evidence_provenance() -> None:
    summary = audit_demo(max_quality_loss=0.02, objective="balanced")

    assert summary["schema_version"] == "routefoundry.audit.v2"
    assert summary["provenance"]["illustrative"] is True
    assert summary["provenance"]["empirical_evidence"] is False
    assert summary["baselines"]["oracle"]["deployable"] is False


def test_demo_jsonl_is_schema_shaped_and_labelled(tmp_path) -> None:
    path = write_demo_jsonl(tmp_path / "demo.jsonl", prompts_per_task=3)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 8 * 3 * 3
    assert all(row["metadata"]["evidence_status"] == "illustrative_non_evidence" for row in rows)
    assert all("trace_id" in row and "sequence_index" in row for row in rows)


def test_space_uses_core_offline_demo_and_report() -> None:
    path = Path(__file__).parents[1] / "space" / "app.py"
    spec = importlib.util.spec_from_file_location("routefoundry_space_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    markdown, report = module.run_demo(0.02, "balanced")
    assert DEMO_DATA_LABEL in markdown
    assert "No model or network service was called" in markdown
    assert "SYNTHETIC / ILLUSTRATIVE" in report
    assert "<script" not in report.lower()
