"""Deterministic model-pool pruning, policy compilation, and held-out audit."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from routefoundry.classify import TaskClassifier
from routefoundry.policy import (
    SUPPORTED_OBJECTIVES,
    ExpectedMetrics,
    Policy,
    RouteRule,
    route,
)
from routefoundry.schema import Dataset, Observation, validate_observations
from routefoundry.split import DEFAULT_SEED, DatasetSplit, split_observations
from routefoundry.stats import ConfidenceInterval, bootstrap_ci, mean, percentile

AUDIT_SCHEMA_VERSION = "routefoundry.audit.v2"
MIN_AUDIT_PROMPTS = 30
MIN_TRAIN_PROMPTS = 10
MIN_DEVELOPMENT_PROMPTS = 5
MIN_TEST_PROMPTS = 5
MIN_HELD_OUT_TRACE_CLUSTERS = 3


class OptimizationError(ValueError):
    """Raised when a dataset cannot support a defensible audit or policy."""


@dataclass(frozen=True, slots=True)
class BaselineMetrics:
    name: str
    deployable: bool
    description: str
    quality: float
    quality_loss: float
    cost_usd: float
    latency_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    coverage: float
    routing_regret: float
    switch_count: int | None
    residency_penalties_applied: bool
    applicable: bool
    bootstrap_method: str
    bootstrap_unit_count: int
    model_counts: Mapping[str, int]
    confidence_intervals: Mapping[str, ConfidenceInterval]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "deployable": self.deployable,
            "description": self.description,
            "quality": self.quality,
            "quality_loss": self.quality_loss,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "coverage": self.coverage,
            "routing_regret": self.routing_regret,
            "switch_count": self.switch_count,
            "residency_penalties_applied": self.residency_penalties_applied,
            "applicable": self.applicable,
            "bootstrap_method": self.bootstrap_method,
            "bootstrap_unit_count": self.bootstrap_unit_count,
            "model_counts": dict(sorted(self.model_counts.items())),
            "confidence_intervals": {
                name: interval.to_dict()
                for name, interval in sorted(self.confidence_intervals.items())
            },
        }


@dataclass(frozen=True, slots=True)
class AuditResult:
    objective: str
    max_quality_loss: float
    seed: int
    strongest_model: str
    recommended_pool: tuple[str, ...]
    models: tuple[str, ...]
    workload_fingerprint: str
    trace_complete: bool
    split: DatasetSplit
    development_quality_loss: float
    development_constraint_satisfied: bool
    held_out_quality_loss: float
    held_out_quality_loss_ci_upper: float
    held_out_constraint_satisfied: bool
    baselines: Mapping[str, BaselineMetrics]
    policy: Policy
    schema_version: str = AUDIT_SCHEMA_VERSION
    notes: tuple[str, ...] = ()
    # Baselines that made identical assignments, grouped. On an all-local pool several
    # named strategies collapse onto one another, and listing them as separate rows would
    # overstate how much independent comparison the audit actually performed.
    equivalent_baselines: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "objective": self.objective,
            "max_quality_loss": self.max_quality_loss,
            "seed": self.seed,
            "strongest_model": self.strongest_model,
            "recommended_pool": list(self.recommended_pool),
            "models": list(self.models),
            "workload_fingerprint": self.workload_fingerprint,
            "trace": {
                "complete": self.trace_complete,
                "residency_metrics_available": self.trace_complete,
            },
            "split": self.split.to_dict(),
            "counts": {
                "train_prompts": self.split.train.prompt_count,
                "development_prompts": self.split.dev.prompt_count,
                "test_prompts": self.split.test.prompt_count,
                "models": self.split.train.model_count,
            },
            "development_constraint": {
                "kind": "empirical_development_constraint",
                "score": "user_supplied_quality",
                "observed_quality_loss": self.development_quality_loss,
                "maximum_quality_loss": self.max_quality_loss,
                "satisfied": self.development_constraint_satisfied,
            },
            "held_out_constraint": {
                "kind": "held_out_bootstrap_check",
                "score": "user_supplied_quality",
                "observed_quality_loss": self.held_out_quality_loss,
                "quality_loss_ci_upper": self.held_out_quality_loss_ci_upper,
                "maximum_quality_loss": self.max_quality_loss,
                "satisfied": self.held_out_constraint_satisfied,
            },
            "uncertainty": {
                "bootstrap_method": self.baselines["compiled"].bootstrap_method,
                "held_out_unit_count": self.baselines["compiled"].bootstrap_unit_count,
            },
            "baselines": {name: self.baselines[name].to_dict() for name in sorted(self.baselines)},
            "equivalent_baselines": [list(group) for group in self.equivalent_baselines],
            "distinct_baseline_count": len(self.baselines)
            - sum(len(group) - 1 for group in self.equivalent_baselines),
            "policy": self.policy.to_dict(),
            "notes": list(self.notes),
            "privacy": {"contains_raw_prompts": False},
        }


def _ensure_dataset(
    value: Dataset | Iterable[Observation | Mapping[str, object]], *, require_signal: bool = False
) -> Dataset:
    if isinstance(value, Dataset):
        if require_signal:
            return validate_observations(value.observations, require_signal=True)
        return value
    return validate_observations(value, require_signal=require_signal)


def _validate_options(max_quality_loss: float, objective: str) -> None:
    if not math.isfinite(max_quality_loss) or not 0.0 <= max_quality_loss <= 1.0:
        raise OptimizationError("max_quality_loss must be finite and in [0, 1]")
    if objective not in SUPPORTED_OBJECTIVES:
        choices = ", ".join(sorted(SUPPORTED_OBJECTIVES))
        raise OptimizationError(f"objective must be one of: {choices}")


def _trace_cluster_bootstrap_ci(
    groups: Sequence[Sequence[float]],
    *,
    seed: int,
    confidence: float = 0.95,
    resamples: int = 1_000,
) -> ConfidenceInterval:
    """Resample whole traces and use the prompt-weighted mean in each replicate."""

    if len(groups) < MIN_HELD_OUT_TRACE_CLUSTERS:
        raise OptimizationError(
            "trace-cluster bootstrap requires at least "
            f"{MIN_HELD_OUT_TRACE_CLUSTERS} held-out traces"
        )
    if any(not group for group in groups):
        raise OptimizationError("trace-cluster bootstrap groups must not be empty")
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled: list[float] = []
        for _ in groups:
            sampled.extend(groups[generator.randrange(len(groups))])
        estimates.append(mean(sampled))
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        low=percentile(estimates, alpha),
        high=percentile(estimates, 1.0 - alpha),
        confidence=confidence,
    )


def _mean_metrics(rows: Sequence[Observation]) -> ExpectedMetrics:
    if not rows:
        raise OptimizationError("cannot aggregate an empty result group")
    return ExpectedMetrics(
        quality=mean([row.quality for row in rows]),
        latency_ms=mean([row.latency_ms for row in rows]),
        cost_usd=mean([row.cost_usd for row in rows]),
        load_ms=mean([row.load_ms for row in rows]),
        samples=len(rows),
    )


def aggregate_metrics(dataset: Dataset) -> dict[str, dict[str, ExpectedMetrics]]:
    """Aggregate task/model and global/model metrics without retaining prompts."""

    grouped: dict[str, dict[str, list[Observation]]] = defaultdict(lambda: defaultdict(list))
    for row in dataset:
        grouped["__global__"][row.model].append(row)
        if row.task is not None:
            grouped[row.task][row.model].append(row)
    return {
        task: {model: _mean_metrics(rows) for model, rows in sorted(models.items())}
        for task, models in sorted(grouped.items())
    }


def strongest_model(dataset: Dataset) -> str:
    metrics = aggregate_metrics(dataset)["__global__"]
    return min(
        metrics,
        key=lambda model: (
            -metrics[model].quality,
            metrics[model].cost_usd,
            metrics[model].latency_ms,
            model,
        ),
    )


def _dominates(left: ExpectedMetrics, right: ExpectedMetrics) -> bool:
    no_worse = (
        left.quality >= right.quality
        and left.cost_usd <= right.cost_usd
        and left.latency_ms <= right.latency_ms
        and left.load_ms <= right.load_ms
    )
    strictly_better = (
        left.quality > right.quality
        or left.cost_usd < right.cost_usd
        or left.latency_ms < right.latency_ms
        or left.load_ms < right.load_ms
    )
    return no_worse and strictly_better


def pareto_pool(
    value: Dataset | Iterable[Observation | Mapping[str, object]],
) -> tuple[str, ...]:
    """Keep models that add quality/cost/latency/load coverage for any task."""

    dataset = _ensure_dataset(value)
    aggregates = aggregate_metrics(dataset)
    useful: set[str] = set()
    for model_metrics in aggregates.values():
        for candidate, candidate_metrics in model_metrics.items():
            if not any(
                other != candidate and _dominates(other_metrics, candidate_metrics)
                for other, other_metrics in model_metrics.items()
            ):
                useful.add(candidate)
    # The strongest global model is a safety fallback even if an unusual load metric
    # makes it dominated in a task-specific view.
    useful.add(strongest_model(dataset))
    return tuple(sorted(useful))


def _normalize(value: float, values: Sequence[float]) -> float:
    lower, upper = min(values), max(values)
    if math.isclose(lower, upper):
        return 0.0
    return (value - lower) / (upper - lower)


def _objective_key(
    model: str,
    metrics: Mapping[str, ExpectedMetrics],
    objective: str,
) -> tuple[float, float, float, str]:
    item = metrics[model]
    if objective == "cost":
        return (item.cost_usd, item.latency_ms, -item.quality, model)
    if objective == "latency":
        return (item.latency_ms, item.cost_usd, -item.quality, model)
    costs = [metric.cost_usd for metric in metrics.values()]
    latencies = [metric.latency_ms for metric in metrics.values()]
    balanced = 0.5 * _normalize(item.cost_usd, costs) + 0.5 * _normalize(item.latency_ms, latencies)
    return (balanced, -item.quality, item.cost_usd + item.latency_ms, model)


def _choose_model(
    metrics: Mapping[str, ExpectedMetrics],
    *,
    fallback_model: str,
    max_quality_loss: float,
    objective: str,
    candidates: Iterable[str],
) -> str:
    fallback_quality = metrics[fallback_model].quality
    allowed = [
        model
        for model in candidates
        if model in metrics
        and metrics[model].quality + 1e-12 >= fallback_quality - max_quality_loss
    ]
    if not allowed:
        return fallback_model
    # Normalize balanced scores over only quality-feasible candidates.
    feasible_metrics = {model: metrics[model] for model in allowed}
    return min(allowed, key=lambda model: _objective_key(model, feasible_metrics, objective))


def _restricted_classifier(classifier: TaskClassifier, tasks: set[str]) -> TaskClassifier:
    supported = tuple(task for task in classifier.known_tasks if task in tasks)
    return TaskClassifier(
        known_tasks=supported,
        task_examples={task: classifier.task_examples[task] for task in supported},
        token_counts={task: classifier.token_counts[task] for task in supported},
        min_examples=classifier.min_examples,
        min_evidence_score=classifier.min_evidence_score,
        min_margin=classifier.min_margin,
    )


def _compiled_assignments(
    policy: Policy,
    dataset: Dataset,
    *,
    use_warm_hysteresis: bool,
    trace_order_available: bool | None = None,
) -> tuple[dict[str, str], dict[str, bool]]:
    assignments: dict[str, str] = {}
    covered: dict[str, bool] = {}
    residency_available = (
        dataset.trace_complete if trace_order_available is None else trace_order_available
    )
    ordered_groups = (
        tuple(dataset.trace_prompt_ids.values())
        if residency_available
        else tuple((prompt_id,) for prompt_id in dataset.prompt_ids)
    )
    for prompt_ids in ordered_groups:
        warm_model: str | None = None
        for prompt_id in prompt_ids:
            head = dataset.for_prompt(prompt_id)[0]
            # If no text exists, an explicit task is the only available routing signal.
            explicit_task = head.task if head.prompt is None else None
            decision = route(
                policy,
                head.prompt or "",
                task=explicit_task,
                warm_model=(warm_model if use_warm_hysteresis and residency_available else None),
            )
            assignments[prompt_id] = decision.model
            covered[prompt_id] = not decision.abstained
            warm_model = decision.model
    return assignments, covered


def _development_loss(policy: Policy, dataset: Dataset) -> float:
    assignments, _ = _compiled_assignments(policy, dataset, use_warm_hysteresis=False)
    losses = [
        dataset.row(prompt_id, policy.fallback_model).quality
        - dataset.row(prompt_id, assignments[prompt_id]).quality
        for prompt_id in dataset.prompt_ids
    ]
    return mean(losses)


def _build_policy(
    split: DatasetSplit,
    *,
    max_quality_loss: float,
    objective: str,
    seed: int,
    min_task_examples: int,
) -> tuple[Policy, float, bool]:
    if not split.dev.prompt_ids:
        raise OptimizationError("a non-empty development split is required")
    fallback = strongest_model(split.dev)
    aggregates = aggregate_metrics(split.dev)
    pool = tuple(sorted(set(pareto_pool(split.dev)) | {fallback}))

    fitted = TaskClassifier.fit(split.train, min_examples=min_task_examples)
    task_names = set(aggregates) - {"__global__"}
    classifier = _restricted_classifier(fitted, task_names)
    routes: dict[str, RouteRule] = {}
    for task in classifier.known_tasks:
        metrics = aggregates[task]
        selected = _choose_model(
            metrics,
            fallback_model=fallback,
            max_quality_loss=max_quality_loss,
            objective=objective,
            candidates=pool,
        )
        loss = max(0.0, metrics[fallback].quality - metrics[selected].quality)
        routes[task] = RouteRule(
            task=task,
            model=selected,
            expected_quality=metrics[selected].quality,
            quality_loss=loss,
            reason=(
                f"{selected} is the best {objective} candidate on development data for {task} "
                f"within the {max_quality_loss:.4f} empirical development quality-loss "
                "threshold"
            ),
        )

    policy_metrics = {
        task: {model: metric for model, metric in metrics.items() if model in pool}
        for task, metrics in aggregates.items()
    }
    policy = Policy(
        fallback_model=fallback,
        objective=objective,
        max_quality_loss=max_quality_loss,
        pool=pool,
        routes=routes,
        model_metrics=policy_metrics,
        classifier=classifier,
        seed=seed,
    )
    development_loss = _development_loss(policy, split.dev)

    # A classifier can occasionally confuse two known tasks.  Compilation is fail-safe:
    # if those mistakes violate the empirical development threshold, every known task
    # routes to the strongest fallback.  The held-out test remains untouched.
    if development_loss > max_quality_loss + 1e-12:
        safe_routes = {
            task: RouteRule(
                task=task,
                model=fallback,
                expected_quality=aggregates[task][fallback].quality,
                quality_loss=0.0,
                reason=(
                    "classifier routing exceeded the empirical development "
                    "quality-loss threshold; using safe fallback"
                ),
            )
            for task in classifier.known_tasks
        }
        policy = replace(policy, routes=safe_routes)
        development_loss = _development_loss(policy, split.dev)
    satisfied = development_loss <= max_quality_loss + 1e-12
    return policy, development_loss, satisfied


def compile_policy(
    value: Dataset | Iterable[Observation | Mapping[str, object]],
    *,
    max_quality_loss: float = 0.02,
    objective: str = "balanced",
    seed: int = DEFAULT_SEED,
    min_task_examples: int = 2,
) -> Policy:
    """Compile on train/development partitions without inspecting held-out test scores."""

    _validate_options(max_quality_loss, objective)
    dataset = _ensure_dataset(value, require_signal=True)
    if dataset.prompt_count < 3:
        raise OptimizationError("policy compilation requires at least three prompts")
    split = split_observations(dataset, seed=seed)
    policy, _, satisfied = _build_policy(
        split,
        max_quality_loss=max_quality_loss,
        objective=objective,
        seed=seed,
        min_task_examples=min_task_examples,
    )
    if not satisfied:  # Defensive: safe fallback should make this unreachable.
        raise OptimizationError(
            "could not satisfy the empirical development quality-loss constraint"
        )
    return policy


def _global_baseline_models(dataset: Dataset) -> tuple[str, str, str]:
    aggregates = aggregate_metrics(dataset)["__global__"]
    strongest = min(
        aggregates,
        key=lambda model: (
            -aggregates[model].quality,
            aggregates[model].cost_usd,
            aggregates[model].latency_ms,
            model,
        ),
    )
    cheapest = min(
        aggregates,
        key=lambda model: (
            aggregates[model].cost_usd,
            -aggregates[model].quality,
            aggregates[model].latency_ms,
            model,
        ),
    )
    fastest = min(
        aggregates,
        key=lambda model: (
            aggregates[model].latency_ms,
            -aggregates[model].quality,
            aggregates[model].cost_usd,
            model,
        ),
    )
    return strongest, cheapest, fastest


def _hash_choice(prompt_id: str, models: Sequence[str], seed: int) -> str:
    digest = hashlib.sha256(f"routefoundry-random:{seed}:{prompt_id}".encode()).digest()
    return models[int.from_bytes(digest[:8], "big") % len(models)]


def _oracle_assignments(
    dataset: Dataset,
    *,
    fallback_model: str,
    max_quality_loss: float,
    objective: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for prompt_id in dataset.prompt_ids:
        rows = dataset.for_prompt(prompt_id)
        metrics = {
            row.model: ExpectedMetrics(
                quality=row.quality,
                latency_ms=row.latency_ms,
                cost_usd=row.cost_usd,
                load_ms=row.load_ms,
                samples=1,
            )
            for row in rows
        }
        result[prompt_id] = _choose_model(
            metrics,
            fallback_model=fallback_model,
            max_quality_loss=max_quality_loss,
            objective=objective,
            candidates=metrics,
        )
    return result


def _evaluate_assignments(
    dataset: Dataset,
    assignments: Mapping[str, str],
    *,
    reference_model: str,
    name: str,
    deployable: bool,
    description: str,
    seed: int,
    covered: Mapping[str, bool] | None = None,
    applicable: bool = True,
    trace_order_available: bool | None = None,
) -> BaselineMetrics:
    qualities: list[float] = []
    losses: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    regrets: list[float] = []
    quality_groups: list[list[float]] = []
    loss_groups: list[list[float]] = []
    cost_groups: list[list[float]] = []
    latency_groups: list[list[float]] = []
    model_counts: dict[str, int] = defaultdict(int)
    switches = 0
    residency_available = (
        dataset.trace_complete if trace_order_available is None else trace_order_available
    )
    ordered_groups = (
        tuple(dataset.trace_prompt_ids.values()) if residency_available else (dataset.prompt_ids,)
    )

    for prompt_ids in ordered_groups:
        previous: str | None = None
        group_qualities: list[float] = []
        group_losses: list[float] = []
        group_costs: list[float] = []
        group_latencies: list[float] = []
        for prompt_id in prompt_ids:
            selected_model = assignments[prompt_id]
            selected = dataset.row(prompt_id, selected_model)
            reference = dataset.row(prompt_id, reference_model)
            rows = dataset.for_prompt(prompt_id)
            load_penalty = (
                selected.load_ms if residency_available and previous != selected_model else 0.0
            )
            if residency_available and previous is not None and previous != selected_model:
                switches += 1
            previous = selected_model
            qualities.append(selected.quality)
            losses.append(reference.quality - selected.quality)
            costs.append(selected.cost_usd)
            latencies.append(selected.latency_ms + load_penalty)
            group_qualities.append(selected.quality)
            group_losses.append(reference.quality - selected.quality)
            group_costs.append(selected.cost_usd)
            group_latencies.append(selected.latency_ms + load_penalty)
            regrets.append(max(row.quality for row in rows) - selected.quality)
            model_counts[selected_model] += 1
        quality_groups.append(group_qualities)
        loss_groups.append(group_losses)
        cost_groups.append(group_costs)
        latency_groups.append(group_latencies)

    coverage_values = (
        [1.0 if covered.get(prompt_id, False) else 0.0 for prompt_id in dataset.prompt_ids]
        if covered is not None
        else [1.0] * dataset.prompt_count
    )
    if residency_available:
        intervals = {
            "quality": _trace_cluster_bootstrap_ci(quality_groups, seed=seed),
            "quality_loss": _trace_cluster_bootstrap_ci(loss_groups, seed=seed + 1),
            "cost_usd": _trace_cluster_bootstrap_ci(cost_groups, seed=seed + 2),
            "latency_ms": _trace_cluster_bootstrap_ci(latency_groups, seed=seed + 3),
        }
        bootstrap_method = "trace_cluster"
        bootstrap_unit_count = len(quality_groups)
    else:
        intervals = {
            "quality": bootstrap_ci(qualities, seed=seed),
            "quality_loss": bootstrap_ci(losses, seed=seed + 1),
            "cost_usd": bootstrap_ci(costs, seed=seed + 2),
            "latency_ms": bootstrap_ci(latencies, seed=seed + 3),
        }
        bootstrap_method = "prompt"
        bootstrap_unit_count = len(qualities)
    return BaselineMetrics(
        name=name,
        deployable=deployable,
        description=description,
        quality=mean(qualities),
        quality_loss=mean(losses),
        cost_usd=mean(costs),
        latency_ms=mean(latencies),
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        coverage=mean(coverage_values),
        routing_regret=mean(regrets),
        switch_count=switches if residency_available else None,
        residency_penalties_applied=residency_available,
        applicable=applicable,
        bootstrap_method=bootstrap_method,
        bootstrap_unit_count=bootstrap_unit_count,
        model_counts=model_counts,
        confidence_intervals=intervals,
    )


def audit(
    value: Dataset | Iterable[Observation | Mapping[str, object]],
    *,
    max_quality_loss: float = 0.02,
    objective: str = "balanced",
    seed: int = DEFAULT_SEED,
    min_task_examples: int = 2,
) -> AuditResult:
    """Compile on development data and evaluate all baselines once on held-out test."""

    _validate_options(max_quality_loss, objective)
    dataset = _ensure_dataset(value, require_signal=True)
    if dataset.prompt_count < MIN_AUDIT_PROMPTS:
        raise OptimizationError(
            f"an audit requires at least {MIN_AUDIT_PROMPTS} prompts; "
            f"received {dataset.prompt_count}"
        )
    minimum_trace_units = MIN_HELD_OUT_TRACE_CLUSTERS + 2
    if dataset.trace_complete and len(dataset.trace_prompt_ids) < minimum_trace_units:
        raise OptimizationError(
            f"a trace-level audit requires at least {minimum_trace_units} complete traces: "
            "one train trace, one development trace, and three held-out trace clusters"
        )
    split = split_observations(dataset, seed=seed)
    split_minimums = {
        "train": (split.train.prompt_count, MIN_TRAIN_PROMPTS),
        "development": (split.dev.prompt_count, MIN_DEVELOPMENT_PROMPTS),
        "test": (split.test.prompt_count, MIN_TEST_PROMPTS),
    }
    undersized = [
        f"{name}={observed} (minimum {minimum})"
        for name, (observed, minimum) in split_minimums.items()
        if observed < minimum
    ]
    if undersized:
        raise OptimizationError(
            "audit split is too small for a meaningful held-out comparison: "
            + ", ".join(undersized)
        )
    if dataset.trace_complete and len(split.test.trace_prompt_ids) < MIN_HELD_OUT_TRACE_CLUSTERS:
        raise OptimizationError(
            "trace-level audit needs at least "
            f"{MIN_HELD_OUT_TRACE_CLUSTERS} held-out trace clusters for its bootstrap; "
            f"seed {seed} assigned {len(split.test.trace_prompt_ids)}"
        )

    policy, development_loss, satisfied = _build_policy(
        split,
        max_quality_loss=max_quality_loss,
        objective=objective,
        seed=seed,
        min_task_examples=min_task_examples,
    )
    train_strongest, cheapest, fastest = _global_baseline_models(split.dev)
    if train_strongest != policy.fallback_model:
        raise AssertionError("compiled fallback and strongest development baseline disagree")

    test = split.test
    models = tuple(sorted(test.models))
    all_strongest = {prompt_id: policy.fallback_model for prompt_id in test.prompt_ids}
    all_cheapest = {prompt_id: cheapest for prompt_id in test.prompt_ids}
    all_fastest = {prompt_id: fastest for prompt_id in test.prompt_ids}
    random_assignments = {
        prompt_id: _hash_choice(prompt_id, models, seed) for prompt_id in test.prompt_ids
    }
    task_only = {
        prompt_id: (
            policy.routes[head.task].model
            if head.task is not None and head.task in policy.routes
            else policy.fallback_model
        )
        for prompt_id in test.prompt_ids
        for head in (test.for_prompt(prompt_id)[0],)
    }
    warm_only = dict(all_strongest)
    trace_available = dataset.trace_complete
    compiled, compiled_coverage = _compiled_assignments(
        policy,
        test,
        use_warm_hysteresis=True,
        trace_order_available=trace_available,
    )
    oracle = _oracle_assignments(
        test,
        fallback_model=policy.fallback_model,
        max_quality_loss=max_quality_loss,
        objective=objective,
    )

    specs = [
        (
            "always-strongest",
            all_strongest,
            True,
            "Always use the strongest model selected on development data.",
            None,
            True,
        ),
        (
            "always-cheapest",
            all_cheapest,
            True,
            "Always use the lowest mean-cost model selected on development data.",
            None,
            True,
        ),
        (
            "always-fastest",
            all_fastest,
            True,
            "Always use the lowest mean-latency model selected on development data.",
            None,
            True,
        ),
        (
            "random",
            random_assignments,
            True,
            "Choose a model from a prompt-ID hash; deterministic but intentionally naive.",
            None,
            True,
        ),
        (
            "task-only",
            task_only,
            True,
            "Use the compiled per-task rule with explicit task metadata and no text classifier.",
            None,
            True,
        ),
        (
            "warm-only",
            warm_only,
            True,
            (
                "Keep the initial strongest model resident for each explicit trace."
                if trace_available
                else "Unavailable: input has no complete explicit trace order."
            ),
            None,
            trace_available,
        ),
        (
            "compiled",
            compiled,
            True,
            (
                "Compiled classifier, abstention, and trace-ordered warm-model hysteresis."
                if trace_available
                else "Compiled classifier and abstention; residency effects are unavailable."
            ),
            compiled_coverage,
            True,
        ),
        (
            "oracle",
            oracle,
            False,
            (
                "Non-deployable hindsight per-prompt choice; impossible at runtime and "
                "only an upper bound."
            ),
            None,
            True,
        ),
    ]
    baselines = {
        name: _evaluate_assignments(
            test,
            assignments,
            reference_model=policy.fallback_model,
            name=name,
            deployable=deployable,
            description=description,
            seed=seed + index * 10,
            covered=coverage,
            applicable=applicable,
            trace_order_available=trace_available,
        )
        for index, (
            name,
            assignments,
            deployable,
            description,
            coverage,
            applicable,
        ) in enumerate(specs)
    }

    trace_note = (
        (
            "Whole trace_id units were split without crossing partitions. Load penalties "
            "and switch counts use only supplied sequence_index order; uncertainty uses "
            "deterministic whole-trace cluster bootstrap."
        )
        if trace_available
        else (
            "Residency and switching metrics are unavailable: every prompt must include "
            "trace_id and sequence_index. No load or switch penalty was applied."
        )
    )
    notes = (
        "The hindsight oracle is non-deployable and must never be presented as a router result.",
        trace_note,
        (
            "Quality values are user-supplied grader scores. The empirical development "
            "constraint and held-out check do not guarantee real response quality or "
            "comparability across graders."
        ),
        (
            "Bootstrap intervals describe sampling variation only; they do not establish score "
            "calibration or include grader uncertainty. Untraced data resamples prompts."
        ),
    )
    held_out = baselines["compiled"]
    held_out_ci_upper = held_out.confidence_intervals["quality_loss"].high
    held_out_satisfied = (
        held_out.quality_loss <= max_quality_loss + 1e-12
        and held_out_ci_upper <= max_quality_loss + 1e-12
    )
    return AuditResult(
        equivalent_baselines=_equivalent_baselines(
            {
                "always-strongest": all_strongest,
                "always-cheapest": all_cheapest,
                "always-fastest": all_fastest,
                "random": random_assignments,
                "task-only": task_only,
                "warm-only": warm_only,
                "compiled": compiled,
            }
        ),
        objective=objective,
        max_quality_loss=max_quality_loss,
        seed=seed,
        strongest_model=policy.fallback_model,
        recommended_pool=policy.pool,
        models=dataset.models,
        workload_fingerprint=dataset.workload_fingerprint,
        trace_complete=dataset.trace_complete,
        split=split,
        development_quality_loss=development_loss,
        development_constraint_satisfied=satisfied,
        held_out_quality_loss=held_out.quality_loss,
        held_out_quality_loss_ci_upper=held_out_ci_upper,
        held_out_constraint_satisfied=held_out_satisfied,
        baselines=baselines,
        policy=policy,
        notes=notes,
    )


def _equivalent_baselines(
    assignments: Mapping[str, Mapping[str, str]],
) -> tuple[tuple[str, ...], ...]:
    """Group baselines whose per-prompt assignments are identical.

    ``warm-only`` is constructed from ``always-strongest``, and on an all-local pool every
    ``cost_usd`` is 0 so ``always-cheapest`` resolves to the same model too. Reporting such
    rows as separate strategies would imply more independent comparison than took place, so
    the audit states which names describe the same behaviour on this workload.
    """

    groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for name, mapping in assignments.items():
        signature = tuple(sorted(mapping.items()))
        groups.setdefault(signature, []).append(name)
    return tuple(tuple(sorted(names)) for names in groups.values() if len(names) > 1)


# Explicit aliases make the public surface easy to discover without duplicating logic.
audit_observations = audit
compile_router = compile_policy
prune_pareto_pool = pareto_pool


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "MIN_AUDIT_PROMPTS",
    "MIN_HELD_OUT_TRACE_CLUSTERS",
    "AuditResult",
    "BaselineMetrics",
    "OptimizationError",
    "aggregate_metrics",
    "audit",
    "audit_observations",
    "compile_policy",
    "compile_router",
    "pareto_pool",
    "prune_pareto_pool",
    "strongest_model",
]
