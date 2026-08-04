from __future__ import annotations

import json

from routefoundry.report import prepare_summary, render_report, write_report


def _audit_mapping() -> dict[str, object]:
    return {
        "synthetic": True,
        "objective": "<cost & quality>",
        "max_quality_loss": 0.02,
        "strongest_model": "<strong-model>",
        "recommended_pool": ["small-model", "<strong-model>"],
        "counts": {"train_prompts": 10, "development_prompts": 4, "test_prompts": 5, "models": 2},
        "prompt": "private <script>alert('prompt')</script>",
        "nested": {
            "response": "private response",
            "display": '<img src=x onerror="alert(1)">',
        },
        "baselines": {
            "<compiled>": {
                "deployable": True,
                "quality": 0.95,
                "latency_ms": 42.0,
                "cost_usd": 0.001,
                "coverage": 0.9,
                "switch_count": 2,
            },
            "hindsight_oracle": {"deployable": False, "quality": 0.99},
        },
        "policy": {
            "routes": {
                "<coding>": {
                    "model": "<script>bad()</script>",
                    "expected_quality": 0.95,
                    "quality_loss": 0.01,
                    "reason": "quality < budget",
                }
            }
        },
    }


def test_report_omits_raw_text_and_escapes_every_dynamic_value() -> None:
    document = render_report(_audit_mapping(), title="<Route & report>")

    assert "private <script>" not in document
    assert "private response" not in document
    assert "<img src=x" not in document
    assert "&lt;img src=x onerror=\\&quot;" in document
    assert "&lt;Route &amp; report&gt;" in document
    assert "&lt;cost &amp; quality&gt;" in document
    assert "&lt;compiled&gt;" in document
    assert "&lt;coding&gt;" in document
    assert "&lt;script&gt;bad()&lt;/script&gt;" in document
    assert "SYNTHETIC / ILLUSTRATIVE" in document
    assert "HINDSIGHT ORACLE" in document
    assert "<script" not in document.lower()
    assert "javascript:" not in document.lower()
    assert "Audit at a glance" in document
    assert "Baseline comparison" in document
    assert "Compiled policy routes" in document
    assert "Show full sanitized summary JSON" in document


def test_prompt_inclusion_is_explicit_and_still_html_escaped() -> None:
    document = render_report(_audit_mapping(), include_prompts=True)

    assert "private &lt;script&gt;alert" in document
    assert "private <script>" not in document
    assert (
        "Raw prompts, prompt identifiers, model responses, and messages are included " in document
    )
    assert "by explicit request." in document


def test_write_report_writes_matching_privacy_preserving_summary(tmp_path) -> None:
    artifacts = write_report(_audit_mapping(), tmp_path)

    assert artifacts.html_path == tmp_path / "report.html"
    assert artifacts.summary_path == tmp_path / "summary.json"
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert "prompt" not in summary
    assert "response" not in summary["nested"]
    assert summary["nested"]["display"].startswith("<img")
    assert summary["report_metadata"]["raw_prompts_included"] is False
    assert artifacts.html_path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_prepare_summary_accepts_mapping_and_does_not_mutate_it() -> None:
    source = {"metrics": {"quality": 0.9}}
    summary = prepare_summary(source)

    assert source == {"metrics": {"quality": 0.9}}
    assert summary["metrics"]["quality"] == 0.9
    assert "report_metadata" in summary


def test_report_accepts_production_audit_result_object() -> None:
    from routefoundry.demo import make_demo_dataset
    from routefoundry.optimize import audit

    result = audit(make_demo_dataset(prompts_per_task=6))
    document = render_report(result)

    assert result.schema_version in document
    assert "HINDSIGHT ORACLE" in document
    assert "Models evaluated" in document
    assert "Mean latency (ms)" in document
    assert "Selected model" in document
    assert "<script" not in document.lower()
