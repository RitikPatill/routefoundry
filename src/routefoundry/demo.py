"""Deterministic, keyless demonstration data and decision preview.

Nothing in this module invokes a model, opens a socket, or reads user data.  The numbers
are hand-designed to exercise routing behavior; they are illustrative and must never be
presented as benchmark evidence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from routefoundry.schema import Dataset, Observation, validate_observations

DEMO_DATA_LABEL = "SYNTHETIC / ILLUSTRATIVE / NON-EVIDENCE"
DEMO_VERSION = "routefoundry.synthetic-demo.v1"
DEMO_MODELS = ("local-small", "local-medium", "strong-reference")

_TASK_PROMPTS: Mapping[str, str] = {
    "summarization": "Summarize support ticket {index} in three concise bullets.",
    "extraction": "Extract the invoice number and due date from sample record {index}.",
    "classification": "Classify synthetic review {index} as positive, neutral, or negative.",
    "translation": "Translate example sentence {index} from German to English.",
    "coding": "Fix the deterministic Python function in exercise {index} and explain the bug.",
    "reasoning": "Solve multi-step scheduling puzzle {index} and show the constraints.",
    "tool_use": "Plan the tool calls needed to check fictional order {index} and issue a refund.",
    "multimodal": "Describe the chart attached to synthetic visual task {index}.",
}

# Average quality levels deliberately create easy, medium, and strong-only workloads.
# These are fabricated fixture values, not claims about any real model.
_QUALITY: Mapping[str, Mapping[str, float]] = {
    "summarization": {
        "local-small": 0.910,
        "local-medium": 0.942,
        "strong-reference": 0.955,
    },
    "extraction": {
        "local-small": 0.940,
        "local-medium": 0.948,
        "strong-reference": 0.955,
    },
    "classification": {
        "local-small": 0.938,
        "local-medium": 0.951,
        "strong-reference": 0.957,
    },
    "translation": {
        "local-small": 0.858,
        "local-medium": 0.932,
        "strong-reference": 0.963,
    },
    "coding": {
        "local-small": 0.802,
        "local-medium": 0.916,
        "strong-reference": 0.964,
    },
    "reasoning": {
        "local-small": 0.768,
        "local-medium": 0.891,
        "strong-reference": 0.971,
    },
    "tool_use": {
        "local-small": 0.824,
        "local-medium": 0.921,
        "strong-reference": 0.956,
    },
    "multimodal": {
        "local-small": 0.510,
        "local-medium": 0.867,
        "strong-reference": 0.949,
    },
}

_LATENCY_MS: Mapping[str, float] = {
    "local-small": 180.0,
    "local-medium": 470.0,
    "strong-reference": 920.0,
}
_COST_USD: Mapping[str, float] = {
    "local-small": 0.0000,
    "local-medium": 0.0003,
    "strong-reference": 0.0048,
}
_LOAD_MS: Mapping[str, float] = {
    "local-small": 95.0,
    "local-medium": 410.0,
    "strong-reference": 0.0,
}


def demo_manifest() -> dict[str, Any]:
    """Return provenance that should accompany every presentation of the demo."""

    return {
        "schema_version": DEMO_VERSION,
        "data_label": DEMO_DATA_LABEL,
        "synthetic": True,
        "illustrative": True,
        "empirical_evidence": False,
        "prompt_count": len(_TASK_PROMPTS) * 15,
        "model_count": len(DEMO_MODELS),
        "task_count": len(_TASK_PROMPTS),
        "trace_count": 15,
        "notice": (
            "Hand-designed fixture values for exercising RouteFoundry. "
            "They are not measurements of real models."
        ),
    }


def generate_demo_observations(*, prompts_per_task: int = 15) -> list[dict[str, Any]]:
    """Generate a complete deterministic observation matrix as schema-valid mappings."""

    if prompts_per_task < 3:
        raise ValueError("prompts_per_task must be at least 3 so data can be split")
    records: list[dict[str, Any]] = []
    for task_index, (task, prompt_template) in enumerate(sorted(_TASK_PROMPTS.items())):
        for prompt_index in range(1, prompts_per_task + 1):
            prompt_id = f"demo-{task}-{prompt_index:03d}"
            prompt = prompt_template.format(index=prompt_index)
            # The same metadata object content is repeated for each model because prompt
            # metadata is invariant across a complete observation matrix.
            metadata = {
                "data_origin": "synthetic",
                "evidence_status": "illustrative_non_evidence",
                "demo_version": DEMO_VERSION,
            }
            for model_index, model in enumerate(DEMO_MODELS):
                variation = ((prompt_index + task_index) % 5 - 2) * 0.0015
                records.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "task": task,
                        "trace_id": f"demo-trace-{prompt_index:03d}",
                        "sequence_index": task_index,
                        "model": model,
                        "quality": round(min(1.0, max(0.0, _QUALITY[task][model] + variation)), 6),
                        "latency_ms": round(
                            _LATENCY_MS[model]
                            + (prompt_index % 7) * 8.0
                            + task_index * 3.0
                            + model_index,
                            3,
                        ),
                        "cost_usd": round(_COST_USD[model] * (1.0 + (prompt_index % 4) * 0.04), 8),
                        "load_ms": _LOAD_MS[model],
                        "metadata": dict(metadata),
                    }
                )
    return records


def make_demo_dataset(*, prompts_per_task: int = 15) -> Dataset:
    """Return the validated synthetic demo matrix."""

    return validate_observations(
        generate_demo_observations(prompts_per_task=prompts_per_task),
        require_signal=True,
    )


def demo_observations(*, prompts_per_task: int = 15) -> tuple[Observation, ...]:
    """Return validated observations for library consumers."""

    return make_demo_dataset(prompts_per_task=prompts_per_task).observations


def write_demo_jsonl(path: str | Path, *, prompts_per_task: int = 15) -> Path:
    """Write the explicitly-labelled synthetic fixture as deterministic JSONL."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = generate_demo_observations(prompts_per_task=prompts_per_task)
    target.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return target


def _means(dataset: Dataset) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, list[Observation]]] = defaultdict(lambda: defaultdict(list))
    for row in dataset:
        assert row.task is not None
        grouped[row.task][row.model].append(row)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for task, models in grouped.items():
        result[task] = {}
        for model, rows in models.items():
            count = len(rows)
            result[task][model] = {
                "quality": sum(row.quality for row in rows) / count,
                "latency_ms": sum(row.latency_ms for row in rows) / count,
                "cost_usd": sum(row.cost_usd for row in rows) / count,
            }
    return result


def summarize_demo_decisions(
    *, max_quality_loss: float = 0.02, objective: str = "balanced"
) -> dict[str, Any]:
    """Create an offline decision preview for the Space.

    This is intentionally a small demonstration calculation, not the held-out audit
    engine and not evidence.  The Space uses it to explain how changing the empirical
    quality-loss threshold and objective changes selection without calling any model.
    """

    if not 0.0 <= max_quality_loss <= 1.0:
        raise ValueError("max_quality_loss must be between 0 and 1")
    if objective not in {"balanced", "cost", "latency"}:
        raise ValueError("objective must be one of: balanced, cost, latency")
    dataset = make_demo_dataset()
    means = _means(dataset)
    decisions: dict[str, dict[str, Any]] = {}
    for task in sorted(means):
        candidates = means[task]
        best_quality = max(values["quality"] for values in candidates.values())
        eligible = {
            model: values
            for model, values in candidates.items()
            if best_quality - values["quality"] <= max_quality_loss + 1e-12
        }

        def score(item: tuple[str, dict[str, float]]) -> tuple[float, str]:
            model, values = item
            if objective == "cost":
                value = values["cost_usd"]
            elif objective == "latency":
                value = values["latency_ms"]
            else:
                # Dimensionless, transparent demo trade-off.  Held-out auditing uses
                # the production objective implementation instead.
                value = values["cost_usd"] / 0.0048 + values["latency_ms"] / 920.0
            return value, model

        selected, values = min(eligible.items(), key=score)
        decisions[task] = {
            "selected_model": selected,
            "mean_quality": round(values["quality"], 6),
            "quality_loss": round(best_quality - values["quality"], 6),
            "mean_latency_ms": round(values["latency_ms"], 3),
            "mean_cost_usd": round(values["cost_usd"], 8),
            "eligible_models": sorted(eligible),
        }

    return {
        "schema_version": "routefoundry.demo-summary.v1",
        "provenance": demo_manifest(),
        "inputs": {
            "max_quality_loss": max_quality_loss,
            "objective": objective,
        },
        "observation_count": len(dataset),
        "decisions": decisions,
        "runtime_note": "Offline deterministic calculation; no network or model calls.",
    }


def audit_demo(*, max_quality_loss: float = 0.02, objective: str = "balanced") -> dict[str, Any]:
    """Run the production held-out audit over the labelled synthetic fixture."""

    # Local import keeps basic fixture generation lightweight and avoids a module cycle.
    from routefoundry.optimize import audit

    result = audit(
        make_demo_dataset(),
        max_quality_loss=max_quality_loss,
        objective=objective,
    ).to_dict()
    result["provenance"] = demo_manifest()
    result["runtime_note"] = "Offline deterministic audit; no network or model calls."
    return result


__all__ = [
    "DEMO_DATA_LABEL",
    "DEMO_MODELS",
    "DEMO_VERSION",
    "audit_demo",
    "demo_manifest",
    "demo_observations",
    "generate_demo_observations",
    "make_demo_dataset",
    "summarize_demo_decisions",
    "write_demo_jsonl",
]
