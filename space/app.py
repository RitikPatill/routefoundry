"""RouteFoundry's offline Hugging Face Space.

The app imports the same package used by the CLI.  It evaluates only the checked-in,
deterministic synthetic fixture and never initializes a model client or network adapter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# This fallback supports running ``python space/app.py`` from a source checkout.  A wheel
# installation takes precedence on Hugging Face or in a clean environment.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from routefoundry.demo import DEMO_DATA_LABEL, audit_demo  # noqa: E402
from routefoundry.report import render_report  # noqa: E402


def _decision_markdown(summary: dict[str, Any]) -> str:
    policy = summary["policy"]
    lines = [
        f"## {DEMO_DATA_LABEL}",
        "",
        "These hand-designed fixture values are **not benchmark evidence**.",
        "No model or network service was called.",
        "",
        f"Quality-loss budget: `{summary['max_quality_loss']:.3f}`  ",
        f"Objective: `{summary['objective']}`  ",
        f"Held-out test prompts: `{summary['counts']['test_prompts']}`",
        "",
        "| Task | Selected model | Illustrative expected quality | Quality loss |",
        "|---|---|---:|---:|",
    ]
    for task, decision in policy["routes"].items():
        lines.append(
            f"| {task} | {decision['model']} | {decision['expected_quality']:.4f} | "
            f"{decision['quality_loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The `oracle` row in the report is a hindsight-only, non-deployable upper bound.",
        ]
    )
    return "\n".join(lines)


def run_demo(max_quality_loss: float, objective: str) -> tuple[str, str]:
    """Return a decision summary and dependency-free report without external calls."""

    summary = audit_demo(
        max_quality_loss=float(max_quality_loss),
        objective=str(objective),
    )
    return _decision_markdown(summary), render_report(
        summary,
        title="RouteFoundry synthetic decision preview",
    )


def build_app():  # type: ignore[no-untyped-def]
    """Construct the optional Gradio UI without coupling core code to Gradio."""

    import gradio as gr

    with gr.Blocks(title="RouteFoundry offline demo", analytics_enabled=False) as app:
        gr.Markdown(
            "# RouteFoundry\n"
            "Explore an explainable routing decision on a deterministic synthetic "
            "workload. **Illustrative only — not empirical evidence.**"
        )
        with gr.Row():
            quality_loss = gr.Slider(
                minimum=0.0,
                maximum=0.15,
                value=0.02,
                step=0.005,
                label="Maximum quality loss",
            )
            objective = gr.Dropdown(
                choices=["balanced", "cost", "latency"],
                value="balanced",
                label="Objective",
            )
        run = gr.Button("Recompute illustrative policy", variant="primary")
        decisions = gr.Markdown(label="Decision summary")
        report = gr.HTML(label="Static report")
        run.click(run_demo, inputs=[quality_loss, objective], outputs=[decisions, report])
        app.load(run_demo, inputs=[quality_loss, objective], outputs=[decisions, report])
    return app


if __name__ == "__main__":
    build_app().launch()
