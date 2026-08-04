from __future__ import annotations

import pytest
from routefoundry.optimize import OptimizationError, audit, compile_policy, pareto_pool
from routefoundry.schema import validate_observations
from routefoundry.stats import bootstrap_ci


def workload(
    prompt_count: int = 60,
    *,
    with_trace: bool = False,
    trace_size: int = 5,
):  # type: ignore[no-untyped-def]
    rows: list[dict[str, object]] = []
    models = {
        "strong": (0.96, 110.0, 0.10, 160.0),
        "cheap": (0.95, 220.0, 0.00, 40.0),
        "fast": (0.945, 35.0, 0.04, 15.0),
        "dominated": (0.80, 500.0, 0.20, 300.0),
    }
    for index in range(prompt_count):
        task = "code" if index % 2 == 0 else "summarization"
        prompt = (
            f"Debug this python function race condition number {index}"
            if task == "code"
            else f"Summarize this report into concise bullets number {index}"
        )
        for model, (quality, latency, cost, load) in models.items():
            # Small deterministic variation ensures bootstrap intervals are meaningful.
            variation = (index % 5) * 0.0002
            row: dict[str, object] = {
                "prompt_id": f"prompt-{index:03d}",
                "prompt": prompt,
                "task": task,
                "model": model,
                "quality": quality - variation,
                "latency_ms": latency + index % 3,
                "cost_usd": cost,
                "load_ms": load,
            }
            if with_trace:
                row["trace_id"] = f"test-trace-{index // trace_size:03d}"
                row["sequence_index"] = index % trace_size
            rows.append(row)
    return validate_observations(rows, require_signal=True)


def test_pareto_pruning_removes_model_that_adds_no_coverage() -> None:
    pool = pareto_pool(workload())
    assert "strong" in pool
    assert "cheap" in pool
    assert "fast" in pool
    assert "dominated" not in pool


def test_cost_policy_obeys_development_quality_budget() -> None:
    policy = compile_policy(workload(), max_quality_loss=0.02, objective="cost")
    assert policy.fallback_model == "strong"
    assert {rule.model for rule in policy.routes.values()} == {"cheap"}
    assert all(rule.quality_loss <= 0.02 for rule in policy.routes.values())
    assert "dominated" not in policy.pool


def test_tight_quality_budget_forces_strong_model() -> None:
    policy = compile_policy(workload(), max_quality_loss=0.005, objective="cost")
    assert policy.routes
    assert {rule.model for rule in policy.routes.values()} == {"strong"}


def test_audit_has_all_required_baselines_and_uncertainty() -> None:
    result = audit(workload(), max_quality_loss=0.02, objective="balanced", seed=8)
    assert result.development_constraint_satisfied
    assert result.development_quality_loss <= 0.02
    assert set(result.baselines) == {
        "always-strongest",
        "always-cheapest",
        "always-fastest",
        "random",
        "task-only",
        "warm-only",
        "compiled",
        "oracle",
    }
    assert result.baselines["oracle"].deployable is False
    assert "non-deployable" in result.baselines["oracle"].description.lower()
    assert result.baselines["always-strongest"].quality_loss == 0.0
    assert (
        result.baselines["compiled"].confidence_intervals["quality"].low
        <= result.baselines["compiled"].quality
    )
    assert result.to_dict()["privacy"] == {"contains_raw_prompts": False}
    summary = result.to_dict()
    assert summary["schema_version"] == "routefoundry.audit.v2"
    assert summary["workload_fingerprint"].startswith("sha256:")
    held_out = summary["held_out_constraint"]
    assert held_out["quality_loss_ci_upper"] == result.baselines["compiled"].confidence_intervals[
        "quality_loss"
    ].high
    assert held_out["satisfied"] == (
        held_out["observed_quality_loss"] <= result.max_quality_loss
        and held_out["quality_loss_ci_upper"] <= result.max_quality_loss
    )
    assert any("user-supplied grader scores" in note for note in result.notes)
    assert any("do not guarantee" in note for note in result.notes)


def test_audit_is_deterministic_under_input_reordering() -> None:
    dataset = workload(60)
    reversed_dataset = validate_observations(reversed(dataset.observations), require_signal=True)
    first = audit(dataset, objective="latency", seed=123).to_dict()
    second = audit(reversed_dataset, objective="latency", seed=123).to_dict()
    assert first == second


def test_unordered_workload_has_no_residency_or_switch_claims() -> None:
    result = audit(workload(), objective="latency")
    assert not result.trace_complete
    assert all(metric.switch_count is None for metric in result.baselines.values())
    assert all(
        not metric.residency_penalties_applied for metric in result.baselines.values()
    )
    assert result.baselines["warm-only"].applicable is False
    assert result.baselines["compiled"].bootstrap_method == "prompt"
    assert result.to_dict()["uncertainty"]["bootstrap_method"] == "prompt"
    assert any("Residency and switching metrics are unavailable" in note for note in result.notes)


def test_explicit_trace_enables_ordered_load_and_switch_metrics() -> None:
    result = audit(workload(with_trace=True), objective="latency")
    assert result.trace_complete
    assert all(metric.residency_penalties_applied for metric in result.baselines.values())
    assert all(metric.switch_count is not None for metric in result.baselines.values())
    assert result.baselines["warm-only"].applicable
    assert result.baselines["compiled"].bootstrap_method == "trace_cluster"
    assert result.baselines["compiled"].bootstrap_unit_count >= 3
    repeated = audit(workload(with_trace=True), objective="latency")
    assert (
        result.baselines["compiled"].confidence_intervals
        == repeated.baselines["compiled"].confidence_intervals
    )
    assert any("Whole trace_id units" in note for note in result.notes)
    assert any("cluster bootstrap" in note for note in result.notes)


def test_audit_requires_minimum_workload_and_meaningful_splits() -> None:
    with pytest.raises(OptimizationError, match="at least 30 prompts"):
        audit(workload(29))
    with pytest.raises(OptimizationError, match=r"development=2 \(minimum 5\)"):
        audit(workload(30), seed=3)
    with pytest.raises(OptimizationError, match="at least 5 complete traces"):
        audit(workload(60, with_trace=True, trace_size=30))
    with pytest.raises(OptimizationError, match="at least 5 complete traces"):
        audit(workload(60, with_trace=True, trace_size=15))


def test_bootstrap_interval_is_seeded_and_handles_singleton() -> None:
    first = bootstrap_ci([1.0, 2.0, 3.0], resamples=100, seed=4)
    second = bootstrap_ci([1.0, 2.0, 3.0], resamples=100, seed=4)
    assert first == second
    singleton = bootstrap_ci([2.5], resamples=10)
    assert singleton.low == singleton.high == 2.5
