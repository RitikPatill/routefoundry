"""Minimal RouteFoundry decision adapter; it deliberately performs no model call.

Applications should call ``choose_model`` immediately before their own inference runtime.
Passing sensitive prompts as CLI arguments can expose them through shell history or process
listings, so prefer importing this module for sensitive workloads.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from routefoundry.policy import RouteDecision, load_policy, route


def choose_model(
    policy_path: str | Path,
    prompt: str,
    *,
    task: str | None = None,
    warm_model: str | None = None,
    switch_cost_ms: float = 0.0,
) -> RouteDecision:
    """Load a policy and return an explainable local decision.

    ``warm_model`` must come from the caller's runtime state. RouteFoundry neither detects
    residency here nor invokes the returned model.
    """

    policy = load_policy(policy_path)
    return route(
        policy,
        prompt,
        task=task,
        warm_model=warm_model,
        switch_cost_ms=switch_cost_ms,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Choose a model locally from a compiled RouteFoundry policy."
    )
    parser.add_argument("policy", type=Path, help="Path to router.json")
    parser.add_argument("prompt", help="Prompt to route; not persisted by this adapter")
    parser.add_argument("--task", help="Optional explicit task label")
    parser.add_argument("--warm-model", help="Model currently resident in your runtime")
    parser.add_argument(
        "--switch-cost-ms",
        type=float,
        default=0.0,
        help="Measured extra switch/eviction penalty in milliseconds",
    )
    args = parser.parse_args(argv)

    decision = choose_model(
        args.policy,
        args.prompt,
        task=args.task,
        warm_model=args.warm_model,
        switch_cost_ms=args.switch_cost_ms,
    )
    print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
