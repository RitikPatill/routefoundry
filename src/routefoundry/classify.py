"""A deliberately small, deterministic task classifier with abstention."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from routefoundry.schema import Observation

UNKNOWN_TASK = "unknown"
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_FEATURE_ID_RE = re.compile(r"^bucket:([0-9]+)$")
DEFAULT_FEATURE_BUCKETS = 65_536
FEATURE_HASH_VERSION = "sha256-64-mod-v1"


def feature_bucket(token: str, *, bucket_count: int = DEFAULT_FEATURE_BUCKETS) -> str:
    """Map a token to a stable, bounded feature bucket.

    Feature hashing prevents compiled policies from embedding the original prompt
    vocabulary.  It is *not* anonymization: bucket membership and frequencies can
    still support dictionary or correlation attacks, so reports must not publish the
    feature-count table by default.
    """

    if not isinstance(token, str) or not token:
        raise ValueError("feature token must be a non-empty string")
    if isinstance(bucket_count, bool) or not isinstance(bucket_count, int) or bucket_count < 2:
        raise ValueError("feature bucket count must be an integer >= 2")
    payload = b"routefoundry:feature:v1\0" + token.casefold().encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    bucket = int.from_bytes(digest[:8], "big") % bucket_count
    return f"bucket:{bucket}"


def _is_feature_id(value: str, *, bucket_count: int) -> bool:
    match = _FEATURE_ID_RE.fullmatch(value)
    return match is not None and int(match.group(1)) < bucket_count


def _hash_legacy_counts(
    counts: Mapping[str, Mapping[str, int]], *, bucket_count: int
) -> dict[str, dict[str, int]]:
    """Convert the old in-memory token-count shape without retaining its strings."""

    hashed: dict[str, dict[str, int]] = {}
    for task, task_counts in counts.items():
        merged: Counter[str] = Counter()
        for token, count in task_counts.items():
            feature = (
                token
                if isinstance(token, str) and _is_feature_id(token, bucket_count=bucket_count)
                else feature_bucket(token, bucket_count=bucket_count)
            )
            merged[feature] += count
        hashed[task] = dict(merged)
    return hashed


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(text) if len(token) > 1)


@dataclass(frozen=True, slots=True)
class Classification:
    task: str
    evidence_score: float
    abstained: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            # This softmax-normalized classifier score is not calibrated probability.
            "evidence_score": self.evidence_score,
            "abstained": self.abstained,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True, init=False)
class TaskClassifier:
    """Multinomial hashed-feature evidence with support and margin thresholds."""

    known_tasks: tuple[str, ...]
    task_examples: Mapping[str, int]
    feature_counts: Mapping[str, Mapping[str, int]]
    min_examples: int = 2
    min_evidence_score: float = 0.55
    min_margin: float = 0.10
    feature_bucket_count: int = DEFAULT_FEATURE_BUCKETS

    def __init__(
        self,
        known_tasks: tuple[str, ...],
        task_examples: Mapping[str, int],
        feature_counts: Mapping[str, Mapping[str, int]] | None = None,
        min_examples: int = 2,
        min_evidence_score: float = 0.55,
        min_margin: float = 0.10,
        feature_bucket_count: int = DEFAULT_FEATURE_BUCKETS,
        *,
        token_counts: Mapping[str, Mapping[str, int]] | None = None,
    ) -> None:
        """Build a classifier, accepting ``token_counts`` only as an API shim.

        The compatibility keyword is immediately feature-hashed and is never retained
        or serialized.  New callers should pass ``feature_counts`` containing bucket
        identifiers.
        """

        if feature_counts is not None and token_counts is not None:
            raise ValueError("pass feature_counts, not both feature_counts and token_counts")
        if isinstance(feature_bucket_count, bool) or not isinstance(feature_bucket_count, int):
            raise ValueError("feature_bucket_count must be an integer >= 2")
        selected = feature_counts
        if token_counts is not None:
            selected = _hash_legacy_counts(token_counts, bucket_count=feature_bucket_count)
        if selected is None:
            raise ValueError("feature_counts must be provided")
        object.__setattr__(self, "known_tasks", known_tasks)
        object.__setattr__(self, "task_examples", task_examples)
        object.__setattr__(self, "feature_counts", selected)
        object.__setattr__(self, "min_examples", min_examples)
        object.__setattr__(self, "min_evidence_score", min_evidence_score)
        object.__setattr__(self, "min_margin", min_margin)
        object.__setattr__(self, "feature_bucket_count", feature_bucket_count)
        self.__post_init__()

    @property
    def token_counts(self) -> Mapping[str, Mapping[str, int]]:
        """Deprecated in-memory alias for optimizer compatibility.

        Despite the historical name, keys are feature-bucket identifiers, never raw
        tokens.  The serialized policy uses the honest ``feature_counts`` name.
        """

        return self.feature_counts

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.known_tasks))) != self.known_tasks:
            raise ValueError("known_tasks must be unique and sorted")
        if (
            isinstance(self.min_examples, bool)
            or not isinstance(self.min_examples, int)
            or self.min_examples < 1
        ):
            raise ValueError("min_examples must be positive")
        if (
            isinstance(self.min_evidence_score, bool)
            or not isinstance(self.min_evidence_score, int | float)
            or not math.isfinite(self.min_evidence_score)
            or not 0.0 <= self.min_evidence_score <= 1.0
        ):
            raise ValueError("min_evidence_score must be finite and in [0, 1]")
        if (
            isinstance(self.min_margin, bool)
            or not isinstance(self.min_margin, int | float)
            or not math.isfinite(self.min_margin)
            or not 0.0 <= self.min_margin <= 1.0
        ):
            raise ValueError("min_margin must be finite and in [0, 1]")
        if (
            isinstance(self.feature_bucket_count, bool)
            or not isinstance(self.feature_bucket_count, int)
            or self.feature_bucket_count < 2
        ):
            raise ValueError("feature_bucket_count must be an integer >= 2")
        for task in self.known_tasks:
            examples = self.task_examples.get(task)
            if (
                examples is None
                or isinstance(examples, bool)
                or not isinstance(examples, int)
                or examples < self.min_examples
            ):
                raise ValueError(f"task {task!r} has insufficient classifier examples")
            counts = self.feature_counts.get(task)
            if counts is None or not isinstance(counts, Mapping):
                raise ValueError(f"task {task!r} has no feature-count mapping")
            if any(
                not isinstance(feature, str)
                or not _is_feature_id(feature, bucket_count=self.feature_bucket_count)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for feature, count in counts.items()
            ):
                raise ValueError(f"task {task!r} has invalid feature counts")

    @classmethod
    def fit(
        cls,
        observations: Iterable[Observation],
        *,
        min_examples: int = 2,
        min_evidence_score: float = 0.55,
        min_margin: float = 0.10,
    ) -> TaskClassifier:
        if min_examples < 1:
            raise ValueError("min_examples must be positive")
        if not 0.0 <= min_evidence_score <= 1.0 or not 0.0 <= min_margin <= 1.0:
            raise ValueError("classification thresholds must be in [0, 1]")

        # Every model repeats prompt metadata, so count each prompt only once.
        seen_prompts: set[str] = set()
        examples: Counter[str] = Counter()
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in observations:
            if row.prompt_id in seen_prompts:
                continue
            seen_prompts.add(row.prompt_id)
            if row.task is None:
                continue
            examples[row.task] += 1
            if row.prompt is not None:
                counts[row.task].update(
                    feature_bucket(token, bucket_count=DEFAULT_FEATURE_BUCKETS)
                    for token in tokenize(row.prompt)
                )

        supported = tuple(sorted(task for task, count in examples.items() if count >= min_examples))
        return cls(
            known_tasks=supported,
            task_examples={task: examples[task] for task in supported},
            feature_counts={task: dict(counts[task]) for task in supported},
            min_examples=min_examples,
            min_evidence_score=min_evidence_score,
            min_margin=min_margin,
            feature_bucket_count=DEFAULT_FEATURE_BUCKETS,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TaskClassifier:
        classifier_type = value.get("type")
        legacy = classifier_type in (None, "token-evidence-v1")
        if not legacy and classifier_type != "hashed-feature-evidence-v1":
            raise ValueError(f"unsupported classifier type: {classifier_type!r}")
        known = value.get("known_tasks")
        examples = value.get("task_examples")
        counts = value.get("token_counts" if legacy else "feature_counts")
        if not isinstance(known, list) or not all(isinstance(item, str) for item in known):
            raise ValueError("classifier known_tasks must be a string list")
        if not isinstance(examples, Mapping) or not isinstance(counts, Mapping):
            raise ValueError("classifier counts must be mappings")
        parsed_examples: dict[str, int] = {}
        parsed_counts: dict[str, dict[str, int]] = {}
        for task in known:
            example_count = examples.get(task)
            feature_map = counts.get(task)
            if isinstance(example_count, bool) or not isinstance(example_count, int):
                raise ValueError("classifier example counts must be integers")
            if not isinstance(feature_map, Mapping):
                raise ValueError("classifier feature counts must be mappings")
            parsed_examples[task] = example_count
            parsed_counts[task] = {}
            for feature, count in feature_map.items():
                if (
                    not isinstance(feature, str)
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                ):
                    raise ValueError("classifier feature counts must be integer mappings")
                parsed_counts[task][feature] = count
        raw_min_examples = value.get("min_examples", 2)
        raw_min_evidence = value.get(
            "min_confidence" if legacy else "min_evidence_score",
            0.55,
        )
        raw_min_margin = value.get("min_margin", 0.10)
        raw_bucket_count = value.get("feature_bucket_count", DEFAULT_FEATURE_BUCKETS)
        if (
            isinstance(raw_min_examples, bool)
            or not isinstance(raw_min_examples, int)
            or raw_min_examples < 1
        ):
            raise ValueError("classifier min_examples must be a positive integer")
        if isinstance(raw_min_evidence, bool) or not isinstance(raw_min_evidence, int | float):
            raise ValueError("classifier min_evidence_score must be a number")
        if not math.isfinite(raw_min_evidence) or not 0.0 <= raw_min_evidence <= 1.0:
            raise ValueError("classifier min_evidence_score must be finite and in [0, 1]")
        if isinstance(raw_min_margin, bool) or not isinstance(raw_min_margin, int | float):
            raise ValueError("classifier min_margin must be a number")
        if (
            isinstance(raw_bucket_count, bool)
            or not isinstance(raw_bucket_count, int)
            or raw_bucket_count < 2
        ):
            raise ValueError("classifier feature_bucket_count must be an integer >= 2")
        common: dict[str, object] = {
            "known_tasks": tuple(known),
            "task_examples": parsed_examples,
            "min_examples": raw_min_examples,
            "min_evidence_score": float(raw_min_evidence),
            "min_margin": float(raw_min_margin),
            "feature_bucket_count": raw_bucket_count,
        }
        if legacy:
            return cls(token_counts=parsed_counts, **common)  # type: ignore[arg-type]
        if value.get("feature_hash_version") != FEATURE_HASH_VERSION:
            raise ValueError("unsupported classifier feature_hash_version")
        return cls(feature_counts=parsed_counts, **common)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "hashed-feature-evidence-v1",
            "feature_hash_version": FEATURE_HASH_VERSION,
            "feature_bucket_count": self.feature_bucket_count,
            "known_tasks": list(self.known_tasks),
            "task_examples": dict(self.task_examples),
            "feature_counts": {task: dict(self.feature_counts[task]) for task in self.known_tasks},
            "min_examples": self.min_examples,
            "min_evidence_score": self.min_evidence_score,
            "min_margin": self.min_margin,
        }

    def classify(self, prompt: str, *, task: str | None = None) -> Classification:
        if task is not None:
            if task in self.known_tasks:
                return Classification(task, 1.0, False, "explicit known task")
            return Classification(UNKNOWN_TASK, 0.0, True, f"explicit task {task!r} is unknown")
        if not self.known_tasks:
            return Classification(UNKNOWN_TASK, 0.0, True, "no supported tasks in training data")
        tokens = tokenize(prompt)
        if not tokens:
            return Classification(UNKNOWN_TASK, 0.0, True, "prompt has no classifiable tokens")
        features = tuple(
            feature_bucket(token, bucket_count=self.feature_bucket_count) for token in tokens
        )

        feature_space = set().union(
            *(set(self.feature_counts[known_task]) for known_task in self.known_tasks)
        )
        if not feature_space:
            return Classification(UNKNOWN_TASK, 0.0, True, "classifier feature space is empty")
        overlap = set(features) & feature_space
        if not overlap:
            return Classification(UNKNOWN_TASK, 0.0, True, "prompt has no known hashed evidence")

        # Laplace-smoothed multinomial log likelihood plus an empirical task prior.
        total_examples = sum(self.task_examples.values())
        scored: list[tuple[float, str]] = []
        for known_task in self.known_tasks:
            counts = self.feature_counts[known_task]
            denominator = sum(counts.values()) + len(feature_space)
            log_score = math.log(self.task_examples[known_task] / total_examples)
            for feature in features:
                log_score += math.log((counts.get(feature, 0) + 1) / denominator)
            scored.append((log_score, known_task))
        scored.sort(key=lambda item: (-item[0], item[1]))

        maximum = scored[0][0]
        weights = [math.exp(score - maximum) for score, _ in scored]
        normalizer = sum(weights)
        evidence_score = weights[0] / normalizer
        runner_up = weights[1] / normalizer if len(weights) > 1 else 0.0
        margin = evidence_score - runner_up
        winner = scored[0][1]
        if evidence_score < self.min_evidence_score:
            return Classification(
                UNKNOWN_TASK,
                evidence_score,
                True,
                f"uncalibrated evidence score {evidence_score:.3f} is below "
                f"{self.min_evidence_score:.3f}",
            )
        if margin < self.min_margin:
            return Classification(
                UNKNOWN_TASK,
                evidence_score,
                True,
                f"uncalibrated evidence margin {margin:.3f} is below {self.min_margin:.3f}",
            )
        return Classification(
            winner,
            evidence_score,
            False,
            "uncalibrated hashed-feature evidence exceeded thresholds",
        )
