"""Stable leakage-resistant train/development/test partitioning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from routefoundry.schema import Dataset

DEFAULT_SEED = 42


def _unit_hash(prompt_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"routefoundry:{seed}:{prompt_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def stable_split(
    prompt_id: str,
    *,
    seed: int = DEFAULT_SEED,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> str:
    """Assign an ID by hash; the result never depends on JSONL row order."""

    if not prompt_id:
        raise ValueError("prompt_id must not be empty")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 < dev_fraction < 1.0) or train_fraction + dev_fraction >= 1.0:
        raise ValueError("dev_fraction must be positive and leave room for test")
    value = _unit_hash(prompt_id, seed)
    if value < train_fraction:
        return "train"
    if value < train_fraction + dev_fraction:
        return "dev"
    return "test"


def _assignment_digest(
    identifiers: tuple[str, ...], *, seed: int, partition: str, split_unit: str
) -> str:
    """Fingerprint a complete assignment without serializing prompt identifiers.

    This digest supports deterministic equality/reproducibility checks, but it is not
    anonymity: low-entropy identifiers may still be guessable when an attacker already
    has a candidate dataset.
    """

    digest = hashlib.sha256()
    digest.update(f"routefoundry:split:v2:{split_unit}:{seed}:{partition}\0".encode())
    for identifier in sorted(identifiers):
        encoded = identifier.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: Dataset
    dev: Dataset
    test: Dataset
    seed: int
    split_unit: str = "prompt"

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "seed": self.seed,
            "split_unit": self.split_unit,
            "prompt_counts": {
                "train": self.train.prompt_count,
                "development": self.dev.prompt_count,
                "test": self.test.prompt_count,
            },
            "assignment_digests": {
                "train": _assignment_digest(
                    self.train.prompt_ids,
                    seed=self.seed,
                    partition="train",
                    split_unit="prompt",
                ),
                "development": _assignment_digest(
                    self.dev.prompt_ids,
                    seed=self.seed,
                    partition="development",
                    split_unit="prompt",
                ),
                "test": _assignment_digest(
                    self.test.prompt_ids,
                    seed=self.seed,
                    partition="test",
                    split_unit="prompt",
                ),
            },
        }
        if self.split_unit == "trace":
            traces = {
                "train": tuple(self.train.trace_prompt_ids),
                "development": tuple(self.dev.trace_prompt_ids),
                "test": tuple(self.test.trace_prompt_ids),
            }
            result["trace_counts"] = {
                partition: len(trace_ids) for partition, trace_ids in traces.items()
            }
            result["trace_assignment_digests"] = {
                partition: _assignment_digest(
                    trace_ids,
                    seed=self.seed,
                    partition=partition,
                    split_unit="trace",
                )
                for partition, trace_ids in traces.items()
            }
        return result


def _empty_like(dataset: Dataset) -> Dataset:
    return Dataset((), (), dataset.models)


def split_observations(
    dataset: Dataset,
    *,
    seed: int = DEFAULT_SEED,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> DatasetSplit:
    """Split trace units when complete, otherwise split prompt units.

    Complete traces are indivisible to prevent replay leakage and nonadjacent-event
    switching claims.  A pure hash split can leave a partition empty for small datasets;
    with at least three units, the nearest hash-ranked unit moves from the largest
    partition.  All behavior is deterministic and independent of source row order.
    """

    split_unit = "trace" if dataset.trace_complete else "prompt"
    units: dict[str, tuple[str, ...]] = (
        dict(dataset.trace_prompt_ids)
        if split_unit == "trace"
        else {prompt_id: (prompt_id,) for prompt_id in dataset.prompt_ids}
    )
    assignments: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for unit_id in sorted(units):
        assignments[
            stable_split(
                unit_id,
                seed=seed,
                train_fraction=train_fraction,
                dev_fraction=dev_fraction,
            )
        ].append(unit_id)

    if len(units) >= 3:
        minimum_units = {"train": 1, "dev": 1, "test": 1}
        # Three held-out clusters are the smallest supported trace bootstrap.  When five
        # trace units exist, repair the hash assignment to make that bound possible while
        # retaining at least one indivisible train and development trace.
        if split_unit == "trace" and len(units) >= 5:
            minimum_units["test"] = 3
        for target_name in ("train", "dev", "test"):
            while len(assignments[target_name]) < minimum_units[target_name]:
                donors = [
                    name
                    for name in assignments
                    if len(assignments[name]) > minimum_units[name]
                ]
                donor_name = max(
                    donors,
                    key=lambda name: (len(assignments[name]), name),
                )
                candidates = assignments[donor_name]
                moved = min(
                    candidates,
                    key=lambda unit_id: (_unit_hash(unit_id, seed), unit_id),
                )
                candidates.remove(moved)
                assignments[target_name].append(moved)

    def make(unit_ids: list[str]) -> Dataset:
        prompt_ids = [prompt_id for unit_id in unit_ids for prompt_id in units[unit_id]]
        return dataset.subset(prompt_ids) if prompt_ids else _empty_like(dataset)

    return DatasetSplit(
        train=make(assignments["train"]),
        dev=make(assignments["dev"]),
        test=make(assignments["test"]),
        seed=seed,
        split_unit=split_unit,
    )
