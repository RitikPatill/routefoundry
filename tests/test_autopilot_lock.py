"""A second run must refuse to share an output with a live run.

Concurrent runs interleave writes and compete for the same CPU, so their latencies are
inflated. Producing plausible-but-invalid numbers is worse than failing loudly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from routefoundry.autopilot import ConcurrentRunError, _run_lock


def test_lock_is_created_and_released(tmp_path: Path) -> None:
    output = tmp_path / "obs.jsonl"
    lock = output.with_suffix(output.suffix + ".lock")
    with _run_lock(output):
        assert lock.read_text("utf-8") == str(os.getpid())
    assert not lock.exists()


def test_live_owner_blocks_a_second_run(tmp_path: Path) -> None:
    output = tmp_path / "obs.jsonl"
    # A different, definitely-alive pid: the parent of this interpreter is a safe stand-in.
    output.with_suffix(output.suffix + ".lock").write_text(str(os.getppid()), encoding="utf-8")
    with pytest.raises(ConcurrentRunError, match="corrupt timings"):
        with _run_lock(output):
            pass


def test_stale_lock_from_a_killed_run_is_reclaimed(tmp_path: Path) -> None:
    output = tmp_path / "obs.jsonl"
    lock = output.with_suffix(output.suffix + ".lock")
    lock.write_text("2147483646", encoding="utf-8")  # pid that cannot be running
    with _run_lock(output):
        assert lock.read_text("utf-8") == str(os.getpid())


def test_unreadable_lock_is_treated_as_stale(tmp_path: Path) -> None:
    output = tmp_path / "obs.jsonl"
    output.with_suffix(output.suffix + ".lock").write_text("not-a-pid", encoding="utf-8")
    with _run_lock(output):
        pass
