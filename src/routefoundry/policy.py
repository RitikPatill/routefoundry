"""Schema-versioned compiled policies and explainable runtime routing."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routefoundry.classify import UNKNOWN_TASK, Classification, TaskClassifier

POLICY_SCHEMA_VERSION = "routefoundry.policy.v1"
SUPPORTED_OBJECTIVES = frozenset({"balanced", "cost", "latency"})


class PolicyError(ValueError):
    """Raised when a policy is malformed or incompatible."""


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PolicyError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise PolicyError(f"{name} must be finite and >= {minimum:g}")
    return number


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ExpectedMetrics:
    quality: float
    latency_ms: float
    cost_usd: float
    load_ms: float = 0.0
    samples: int = 0

    def __post_init__(self) -> None:
        quality = _number(self.quality, "expected quality")
        if quality > 1.0:
            raise PolicyError("expected quality must be finite and in [0, 1]")
        for name, value in (
            ("latency_ms", self.latency_ms),
            ("cost_usd", self.cost_usd),
            ("load_ms", self.load_ms),
        ):
            _number(value, f"expected {name}")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 0:
            raise PolicyError("expected samples must be a non-negative integer")

    @classmethod
    def from_dict(cls, value: Mapping[str, object], name: str = "metrics") -> ExpectedMetrics:
        samples = value.get("samples", 0)
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
            raise PolicyError(f"{name}.samples must be a non-negative integer")
        quality = _number(value.get("quality"), f"{name}.quality")
        if quality > 1.0:
            raise PolicyError(f"{name}.quality must be <= 1")
        return cls(
            quality=quality,
            latency_ms=_number(value.get("latency_ms"), f"{name}.latency_ms"),
            cost_usd=_number(value.get("cost_usd"), f"{name}.cost_usd"),
            load_ms=_number(value.get("load_ms", 0.0), f"{name}.load_ms"),
            samples=samples,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "quality": self.quality,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "load_ms": self.load_ms,
            "samples": self.samples,
        }


@dataclass(frozen=True, slots=True)
class RouteRule:
    task: str
    model: str
    expected_quality: float
    quality_loss: float
    reason: str

    def __post_init__(self) -> None:
        route_strings = (self.task, self.model, self.reason)
        if not all(isinstance(value, str) and value for value in route_strings):
            raise PolicyError("route task, model, and reason must be non-empty")
        expected_quality = _number(self.expected_quality, "route expected_quality")
        if expected_quality > 1.0:
            raise PolicyError("route expected_quality must be finite and in [0, 1]")
        quality_loss = _number(self.quality_loss, "route quality_loss")
        if quality_loss > 1.0:
            raise PolicyError("route quality_loss must be finite and in [0, 1]")

    @classmethod
    def from_dict(cls, task: str, value: Mapping[str, object]) -> RouteRule:
        return cls(
            task=task,
            model=_string(value.get("model"), f"routes.{task}.model"),
            expected_quality=_number(
                value.get("expected_quality"), f"routes.{task}.expected_quality"
            ),
            quality_loss=_number(value.get("quality_loss"), f"routes.{task}.quality_loss"),
            reason=_string(value.get("reason"), f"routes.{task}.reason"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "expected_quality": self.expected_quality,
            "quality_loss": self.quality_loss,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Policy:
    fallback_model: str
    objective: str
    max_quality_loss: float
    pool: tuple[str, ...]
    routes: Mapping[str, RouteRule]
    model_metrics: Mapping[str, Mapping[str, ExpectedMetrics]]
    classifier: TaskClassifier
    seed: int = 42
    schema_version: str = POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_SCHEMA_VERSION:
            raise PolicyError(
                f"unsupported policy schema {self.schema_version!r}; "
                f"expected {POLICY_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.objective, str) or self.objective not in SUPPORTED_OBJECTIVES:
            raise PolicyError(f"unsupported objective {self.objective!r}")
        max_quality_loss = _number(self.max_quality_loss, "max_quality_loss")
        if max_quality_loss > 1.0:
            raise PolicyError("max_quality_loss must be in [0, 1]")
        if not isinstance(self.fallback_model, str) or self.fallback_model not in self.pool:
            raise PolicyError("fallback_model must be present in the model pool")
        if (
            not isinstance(self.pool, tuple)
            or not all(isinstance(model, str) and model for model in self.pool)
            or len(self.pool) != len(set(self.pool))
            or tuple(sorted(self.pool)) != self.pool
        ):
            raise PolicyError("pool must be unique and sorted")
        global_metrics = self.model_metrics.get("__global__")
        if global_metrics is None or any(model not in global_metrics for model in self.pool):
            raise PolicyError("model_metrics must contain global metrics for every pooled model")
        for task, metrics in self.model_metrics.items():
            if not metrics:
                raise PolicyError(f"model_metrics for {task!r} must not be empty")
            if any(model not in self.pool for model in metrics):
                raise PolicyError(f"model_metrics for {task!r} references a model outside the pool")
            if any(not isinstance(metric, ExpectedMetrics) for metric in metrics.values()):
                raise PolicyError(f"model_metrics for {task!r} contains invalid metrics")
        for task, rule in self.routes.items():
            if task != rule.task:
                raise PolicyError(f"route key {task!r} does not match rule task {rule.task!r}")
            if rule.model not in self.pool:
                raise PolicyError(f"route for {task!r} references model outside the pool")
            task_metrics = self.model_metrics.get(task)
            if task_metrics is None or rule.model not in task_metrics:
                raise PolicyError(f"route for {task!r} has no corresponding metrics")
        for task in self.classifier.known_tasks:
            if task not in self.routes:
                raise PolicyError(f"classifier task {task!r} has no route")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Policy:
        schema_version = value.get("schema_version")
        if schema_version != POLICY_SCHEMA_VERSION:
            raise PolicyError(
                f"unsupported policy schema {schema_version!r}; expected {POLICY_SCHEMA_VERSION!r}"
            )
        raw_pool = value.get("pool")
        raw_routes = value.get("routes")
        raw_metrics = value.get("model_metrics")
        raw_classifier = value.get("classifier")
        if not isinstance(raw_pool, list) or not all(isinstance(item, str) for item in raw_pool):
            raise PolicyError("pool must be a list of model strings")
        if not isinstance(raw_routes, Mapping) or not isinstance(raw_metrics, Mapping):
            raise PolicyError("routes and model_metrics must be objects")
        if not isinstance(raw_classifier, Mapping):
            raise PolicyError("classifier must be an object")

        routes: dict[str, RouteRule] = {}
        for task, rule in raw_routes.items():
            if not isinstance(task, str) or not isinstance(rule, Mapping):
                raise PolicyError("routes must map task strings to route objects")
            routes[task] = RouteRule.from_dict(task, rule)
        metrics: dict[str, dict[str, ExpectedMetrics]] = {}
        for task, model_map in raw_metrics.items():
            if not isinstance(task, str) or not isinstance(model_map, Mapping):
                raise PolicyError("model_metrics must map tasks to model metric objects")
            metrics[task] = {}
            for model, metric in model_map.items():
                if not isinstance(model, str) or not isinstance(metric, Mapping):
                    raise PolicyError("model_metrics entries must be objects")
                metrics[task][model] = ExpectedMetrics.from_dict(metric, f"{task}.{model}")

        seed = value.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PolicyError("seed must be an integer")
        return cls(
            schema_version=POLICY_SCHEMA_VERSION,
            fallback_model=_string(value.get("fallback_model"), "fallback_model"),
            objective=_string(value.get("objective"), "objective"),
            max_quality_loss=_number(value.get("max_quality_loss"), "max_quality_loss"),
            pool=tuple(raw_pool),
            routes=routes,
            model_metrics=metrics,
            classifier=TaskClassifier.from_dict(raw_classifier),
            seed=seed,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator": {"name": "routefoundry", "version": "0.1.0"},
            "objective": self.objective,
            "max_quality_loss": self.max_quality_loss,
            "fallback_model": self.fallback_model,
            "pool": list(self.pool),
            "routes": {task: self.routes[task].to_dict() for task in sorted(self.routes)},
            "model_metrics": {
                task: {
                    model: self.model_metrics[task][model].to_dict()
                    for model in sorted(self.model_metrics[task])
                }
                for task in sorted(self.model_metrics)
            },
            "classifier": self.classifier.to_dict(),
            "seed": self.seed,
            "privacy": {"contains_raw_prompts": False},
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    model: str
    task: str
    evidence_score: float
    abstained: bool
    used_warm_model: bool
    fallback_model: str
    reason: str
    expected_quality: float | None
    expected_latency_ms: float | None
    expected_cost_usd: float | None
    switch_penalty_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "task": self.task,
            "evidence_score": self.evidence_score,
            "abstained": self.abstained,
            "used_warm_model": self.used_warm_model,
            "fallback_model": self.fallback_model,
            "reason": self.reason,
            "expected_quality": self.expected_quality,
            "expected_latency_ms": self.expected_latency_ms,
            "expected_cost_usd": self.expected_cost_usd,
            "switch_penalty_ms": self.switch_penalty_ms,
        }


def _metrics_for(policy: Policy, task: str, model: str) -> ExpectedMetrics | None:
    metrics = policy.model_metrics.get(task, {}).get(model)
    if metrics is not None:
        return metrics
    return policy.model_metrics.get("__global__", {}).get(model)


def _decision(
    policy: Policy,
    *,
    model: str,
    classification: Classification,
    abstained: bool,
    used_warm_model: bool,
    reason: str,
    switch_penalty_ms: float = 0.0,
) -> RouteDecision:
    metrics = _metrics_for(policy, classification.task, model)
    return RouteDecision(
        model=model,
        task=classification.task,
        evidence_score=classification.evidence_score,
        abstained=abstained,
        used_warm_model=used_warm_model,
        fallback_model=policy.fallback_model,
        reason=reason,
        expected_quality=metrics.quality if metrics else None,
        expected_latency_ms=metrics.latency_ms if metrics else None,
        expected_cost_usd=metrics.cost_usd if metrics else None,
        switch_penalty_ms=switch_penalty_ms,
    )


def route(
    policy: Policy,
    prompt: str,
    *,
    task: str | None = None,
    warm_model: str | None = None,
    switch_cost_ms: float = 0.0,
) -> RouteDecision:
    """Choose a model, abstaining and applying warm-model hysteresis when warranted."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    switch_cost = _number(switch_cost_ms, "switch_cost_ms")
    classification = policy.classifier.classify(prompt, task=task)
    if classification.abstained:
        return _decision(
            policy,
            model=policy.fallback_model,
            classification=classification,
            abstained=True,
            used_warm_model=warm_model == policy.fallback_model,
            reason=f"abstained to strongest fallback: {classification.reason}",
        )

    rule = policy.routes.get(classification.task)
    if rule is None:
        unknown = Classification(UNKNOWN_TASK, classification.evidence_score, True, "no route")
        return _decision(
            policy,
            model=policy.fallback_model,
            classification=unknown,
            abstained=True,
            used_warm_model=warm_model == policy.fallback_model,
            reason="abstained to strongest fallback: classified task has no route",
        )

    selected = rule.model
    selected_metrics = _metrics_for(policy, classification.task, selected)
    if warm_model is None or warm_model == selected:
        return _decision(
            policy,
            model=selected,
            classification=classification,
            abstained=False,
            used_warm_model=warm_model == selected,
            reason=(
                f"kept already-warm routed model for {classification.task}"
                if warm_model == selected
                else rule.reason
            ),
        )
    if warm_model not in policy.pool:
        return _decision(
            policy,
            model=selected,
            classification=classification,
            abstained=False,
            used_warm_model=False,
            reason=f"{rule.reason}; ignored warm model outside compiled pool",
        )

    warm_metrics = _metrics_for(policy, classification.task, warm_model)
    fallback_metrics = _metrics_for(policy, classification.task, policy.fallback_model)
    if selected_metrics is None or warm_metrics is None or fallback_metrics is None:
        return _decision(
            policy,
            model=selected,
            classification=classification,
            abstained=False,
            used_warm_model=False,
            reason=f"{rule.reason}; insufficient metrics for warm-model comparison",
        )

    quality_floor = fallback_metrics.quality - policy.max_quality_loss
    if warm_metrics.quality + 1e-12 < quality_floor:
        return _decision(
            policy,
            model=selected,
            classification=classification,
            abstained=False,
            used_warm_model=False,
            reason=(
                f"{rule.reason}; warm model quality {warm_metrics.quality:.4f} is below "
                f"allowed floor {quality_floor:.4f}"
            ),
        )

    switch_penalty = selected_metrics.load_ms + switch_cost
    latency_gain = max(0.0, warm_metrics.latency_ms - selected_metrics.latency_ms)
    if switch_penalty + 1e-12 >= latency_gain:
        return _decision(
            policy,
            model=warm_model,
            classification=classification,
            abstained=False,
            used_warm_model=True,
            reason=(
                f"retained warm model: {switch_penalty:.2f} ms load/switch penalty "
                f"outweighs {latency_gain:.2f} ms expected latency gain; quality is in budget"
            ),
            switch_penalty_ms=switch_penalty,
        )
    return _decision(
        policy,
        model=selected,
        classification=classification,
        abstained=False,
        used_warm_model=False,
        reason=(
            f"{rule.reason}; switched because {latency_gain:.2f} ms expected latency gain "
            f"exceeds {switch_penalty:.2f} ms load/switch penalty"
        ),
        switch_penalty_ms=switch_penalty,
    )


def dump_policy(policy: Policy, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(policy.to_dict(), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_policy(source: str | Path | Mapping[str, object]) -> Policy:
    if isinstance(source, Mapping):
        return Policy.from_dict(source)
    path = Path(source)
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyError(f"could not load policy {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise PolicyError("policy JSON root must be an object")
    return Policy.from_dict(value)


# Friendly aliases for integrations that prefer verb-noun names.
route_prompt = route
save_policy = dump_policy


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "ExpectedMetrics",
    "Policy",
    "PolicyError",
    "RouteDecision",
    "RouteRule",
    "dump_policy",
    "load_policy",
    "route",
    "route_prompt",
    "save_policy",
]
