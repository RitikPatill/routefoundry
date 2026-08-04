"""Small deterministic statistical helpers used by audits and reports."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    low: float
    high: float
    confidence: float = 0.95

    def to_dict(self) -> dict[str, float]:
        return {"low": self.low, "high": self.high, "confidence": self.confidence}


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("values must be finite")
    return float(fmean(values))


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated percentile matching the common R-7 convention."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("values must be finite")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    confidence: float = 0.95,
    resamples: int = 1_000,
    seed: int = 42,
) -> ConfidenceInterval:
    """Percentile bootstrap over independent prompt-level values."""

    if not values:
        raise ValueError("bootstrap_ci requires at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    checked = [float(value) for value in values]
    if any(not math.isfinite(value) for value in checked):
        raise ValueError("values must be finite")

    generator = random.Random(seed)
    size = len(checked)
    estimates = [
        float(statistic([checked[generator.randrange(size)] for _ in range(size)]))
        for _ in range(resamples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        low=percentile(estimates, alpha),
        high=percentile(estimates, 1.0 - alpha),
        confidence=confidence,
    )
