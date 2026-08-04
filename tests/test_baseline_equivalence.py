"""Baselines that behave identically must be reported as such.

`warm-only` is built from `always-strongest`, and on an all-local pool every cost is 0 so
`always-cheapest` resolves to the same model too. Presenting those as separate rows would
claim more independent comparison than the audit performed.
"""

from __future__ import annotations

from routefoundry.demo import audit_demo
from routefoundry.optimize import _equivalent_baselines


def test_identical_assignments_are_grouped() -> None:
    groups = _equivalent_baselines(
        {
            "a": {"p1": "small", "p2": "small"},
            "b": {"p1": "small", "p2": "small"},
            "c": {"p1": "large", "p2": "small"},
        }
    )
    assert groups == (("a", "b"),)


def test_distinct_assignments_produce_no_group() -> None:
    assert _equivalent_baselines({"a": {"p1": "x"}, "b": {"p1": "y"}}) == ()


def test_audit_reports_the_collapse_and_a_distinct_count() -> None:
    summary = audit_demo()
    grouped = {name for group in summary["equivalent_baselines"] for name in group}
    # warm-only is a copy of always-strongest by construction, so it can never be distinct.
    assert {"always-strongest", "warm-only"} <= grouped
    assert summary["distinct_baseline_count"] < len(summary["baselines"])
