from __future__ import annotations

import json

import pytest

from routefoundry.classify import TaskClassifier
from routefoundry.policy import (
    POLICY_SCHEMA_VERSION,
    ExpectedMetrics,
    Policy,
    PolicyError,
    RouteRule,
    dump_policy,
    load_policy,
    route,
)


def make_policy(*, small_load_ms: float = 100.0) -> Policy:
    classifier = TaskClassifier(
        known_tasks=("code",),
        task_examples={"code": 3},
        token_counts={"code": {"python": 3, "debug": 2, "function": 1}},
        min_examples=2,
        min_evidence_score=0.55,
        min_margin=0.1,
    )
    metrics = {
        "code": {
            "small": ExpectedMetrics(0.94, 50.0, 0.01, small_load_ms, 3),
            "strong": ExpectedMetrics(0.95, 100.0, 0.10, 200.0, 3),
            "weak": ExpectedMetrics(0.70, 20.0, 0.0, 1.0, 3),
        },
        "__global__": {
            "small": ExpectedMetrics(0.94, 50.0, 0.01, small_load_ms, 3),
            "strong": ExpectedMetrics(0.95, 100.0, 0.10, 200.0, 3),
            "weak": ExpectedMetrics(0.70, 20.0, 0.0, 1.0, 3),
        },
    }
    return Policy(
        fallback_model="strong",
        objective="cost",
        max_quality_loss=0.02,
        pool=("small", "strong", "weak"),
        routes={
            "code": RouteRule("code", "small", 0.94, 0.01, "small is cheaper within quality budget")
        },
        model_metrics=metrics,
        classifier=classifier,
    )


def test_unknown_prompt_and_unknown_explicit_task_abstain() -> None:
    policy = make_policy()
    decision = route(policy, "Translate an unfamiliar poem")
    assert decision.model == "strong"
    assert decision.abstained
    assert "fallback" in decision.reason

    explicit = route(policy, "anything", task="never-seen")
    assert explicit.model == "strong"
    assert explicit.task == "unknown"
    assert explicit.abstained


def test_known_task_routes_with_explanation() -> None:
    decision = route(make_policy(), "Debug this Python function")
    assert decision.model == "small"
    assert not decision.abstained
    assert decision.expected_quality == 0.94
    assert "quality budget" in decision.reason


def test_warm_model_hysteresis_uses_load_and_switch_cost() -> None:
    retained = route(
        make_policy(small_load_ms=100.0),
        "Debug this Python function",
        warm_model="strong",
    )
    assert retained.model == "strong"
    assert retained.used_warm_model
    assert retained.switch_penalty_ms == 100.0
    assert "retained warm model" in retained.reason

    switched = route(
        make_policy(small_load_ms=1.0),
        "Debug this Python function",
        warm_model="strong",
    )
    assert switched.model == "small"
    assert not switched.used_warm_model
    assert "switched because" in switched.reason


def test_warm_model_outside_quality_budget_is_never_retained() -> None:
    decision = route(
        make_policy(small_load_ms=100.0),
        "Debug this Python function",
        warm_model="weak",
        switch_cost_ms=10_000,
    )
    assert decision.model == "small"
    assert not decision.used_warm_model
    assert "below allowed floor" in decision.reason


def test_policy_round_trip_is_stable_and_contains_no_prompt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    policy = make_policy()
    destination = tmp_path / "router.json"
    dump_policy(policy, destination)
    payload = destination.read_text(encoding="utf-8")
    assert '"schema_version": "routefoundry.policy.v1"' in payload
    assert '"contains_raw_prompts": false' in payload
    assert '"type": "hashed-feature-evidence-v1"' in payload
    assert '"feature_counts"' in payload
    assert '"token_counts"' not in payload
    assert '"min_evidence_score"' in payload
    assert '"min_confidence"' not in payload
    assert "python" not in payload.casefold()
    assert "debug" not in payload.casefold()
    assert "Debug this Python" not in payload
    assert load_policy(destination).to_dict() == policy.to_dict()

    mapping = json.loads(payload)
    assert load_policy(mapping).to_dict() == policy.to_dict()


def test_policy_rejects_unknown_schema_and_invalid_references() -> None:
    value = make_policy().to_dict()
    value["schema_version"] = "routefoundry.policy.v999"
    with pytest.raises(PolicyError, match="unsupported policy schema"):
        load_policy(value)

    with pytest.raises(PolicyError, match="fallback_model"):
        Policy(
            fallback_model="missing",
            objective="cost",
            max_quality_loss=0.02,
            pool=("small",),
            routes={},
            model_metrics={},
            classifier=TaskClassifier((), {}, {}),
            schema_version=POLICY_SCHEMA_VERSION,
        )

    with pytest.raises(PolicyError, match="quality"):
        ExpectedMetrics(float("nan"), 1.0, 0.0)

    invalid_classifier = make_policy().to_dict()
    classifier = invalid_classifier["classifier"]
    assert isinstance(classifier, dict)
    classifier["min_evidence_score"] = float("nan")
    with pytest.raises(ValueError, match="min_evidence_score"):
        load_policy(invalid_classifier)
