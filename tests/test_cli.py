from __future__ import annotations

import json
from pathlib import Path

import pytest
from routefoundry.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_version_option_works_without_a_subcommand() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip().startswith("routefoundry 0.1.0")


def test_demo_creates_reviewable_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    result = runner.invoke(app, ["demo", "--output", str(output)])
    assert result.exit_code == 0, result.output
    for name in (
        "report.html",
        "summary.json",
        "router.json",
        "policy.txt",
        "hf-chat-ui-routes.json",
        "demo-observations.jsonl",
    ):
        assert (output / name).is_file(), name
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["report_metadata"]["raw_prompts_included"] is False
    assert any("SYNTHETIC" in label for label in summary["report_metadata"]["data_labels"])


def test_validate_and_route_demo_policy(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    demo_result = runner.invoke(app, ["demo", "--output", str(output)])
    assert demo_result.exit_code == 0, demo_result.output

    valid = runner.invoke(app, ["validate", str(output / "demo-observations.jsonl")])
    assert valid.exit_code == 0
    assert "Valid" in valid.output

    decision = runner.invoke(
        app,
        [
            "route",
            str(output / "router.json"),
            "Write a Python function to sort these records",
            "--json",
        ],
    )
    assert decision.exit_code == 0, decision.output
    assert '"model"' in decision.output


def test_compile_and_hf_export(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    assert runner.invoke(app, ["demo", "-o", str(demo_dir)]).exit_code == 0
    policy = tmp_path / "compiled.json"
    compiled = runner.invoke(
        app,
        ["compile", str(demo_dir / "demo-observations.jsonl"), "-o", str(policy)],
    )
    assert compiled.exit_code == 0, compiled.output
    exported = tmp_path / "routes.json"
    result = runner.invoke(
        app,
        ["export", str(policy), "--format", "hf-chat-ui", "-o", str(exported)],
    )
    assert result.exit_code == 0, result.output
    routes = json.loads(exported.read_text(encoding="utf-8"))
    assert routes[0]["name"] == "default"


def test_ollama_profile_requires_explicit_residency_acknowledgement() -> None:
    result = runner.invoke(app, ["ollama-profile", "llama3.2:3b"])
    assert result.exit_code == 2
    assert "temporarily changes model residency" in result.output


def test_ollama_profile_accepts_documented_models_option() -> None:
    result = runner.invoke(app, ["ollama-profile", "--models", "one,two"])
    assert result.exit_code == 2
    assert "temporarily changes model residency" in result.output


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "routefoundry.audit.v999"),
        ("workload_fingerprint", "sha256:different"),
        ("models", ["different-model"]),
        ("seed", 999),
        ("objective", "cost"),
        ("max_quality_loss", 0.5),
    ],
)
def test_ci_rejects_incompatible_baseline_before_metrics(
    tmp_path: Path, field: str, replacement: object
) -> None:
    output = tmp_path / "demo"
    demo_result = runner.invoke(app, ["demo", "-o", str(output)])
    assert demo_result.exit_code == 0, demo_result.output
    baseline = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    baseline[field] = replacement
    baseline_path = tmp_path / "incompatible.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ci",
            str(output / "demo-observations.jsonl"),
            "--baseline",
            str(baseline_path),
        ],
    )
    assert result.exit_code == 2
    assert "incompatible baseline" in result.output
    assert field in result.output


def test_invalid_export_format_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    assert runner.invoke(app, ["demo", "-o", str(output)]).exit_code == 0
    result = runner.invoke(
        app,
        ["export", str(output / "router.json"), "--format", "unknown", "-o", str(tmp_path / "x")],
    )
    assert result.exit_code == 2
    assert "format must be" in result.output
