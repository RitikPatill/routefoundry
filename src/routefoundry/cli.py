"""Command-line interface for RouteFoundry's deterministic local workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

from routefoundry import __version__
from routefoundry.autopilot import (
    DEFAULT_TIMEOUT_SECONDS,
    AutopilotError,
    RunProgress,
    discover_models,
    estimate_duration_seconds,
    format_duration,
    run_autopilot,
)
from routefoundry.demo import audit_demo, write_demo_jsonl
from routefoundry.exporters import export_hf_chat_ui, export_human_policy
from routefoundry.ollama import DEFAULT_BASE_URL, OllamaProfileError, profile_ollama_models
from routefoundry.optimize import audit, compile_policy
from routefoundry.policy import SUPPORTED_OBJECTIVES, Policy, dump_policy, load_policy, route
from routefoundry.report import write_report
from routefoundry.schema import ValidationError, load_jsonl
from routefoundry.tasks import TaskSuiteError, load_tasks, select_tasks

app = typer.Typer(
    name="routefoundry",
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode="rich",
    help=(
        "Audit model results and compile an explainable router against an empirical "
        "development quality-loss threshold. Core commands run locally without an "
        "API key or model call."
    ),
)
console = Console()
error_console = Console(stderr=True)


def _objective(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_OBJECTIVES:
        choices = ", ".join(sorted(SUPPORTED_OBJECTIVES))
        raise typer.BadParameter(f"must be one of: {choices}")
    return normalized


def _print_audit_summary(result: dict[str, Any], output: Path) -> None:
    compiled = result.get("baselines", {}).get("compiled", {})
    table = Table(title="RouteFoundry audit")
    table.add_column("Measure", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Objective", str(result.get("objective", "unknown")))
    table.add_row(
        "Evidence-score loss threshold",
        f"{float(result.get('max_quality_loss', 0)):.4f}",
    )
    table.add_row("Strong fallback", str(result.get("strongest_model", "unknown")))
    table.add_row("Recommended pool", ", ".join(result.get("recommended_pool", [])))
    if compiled:
        table.add_row(
            "Held-out empirical quality score",
            f"{float(compiled.get('quality', 0)):.4f}",
        )
        table.add_row(
            "Held-out quality-score loss",
            f"{float(compiled.get('quality_loss', 0)):.4f}",
        )
        table.add_row("Held-out compiled latency", f"{float(compiled.get('latency_ms', 0)):.2f} ms")
        table.add_row("Held-out compiled coverage", f"{float(compiled.get('coverage', 0)):.1%}")
    table.add_row("Artifacts", str(output.resolve()))
    console.print(table)


def _fail(exc: Exception) -> None:
    error_console.print(f"[bold red]RouteFoundry stopped:[/bold red] {exc}")
    raise typer.Exit(code=2) from exc


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Print the installed version and exit."),
    ] = False,
) -> None:
    if version:
        console.print(f"routefoundry {__version__}")
        raise typer.Exit()


@app.command()
def demo(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory for the synthetic demo artifacts."),
    ] = Path("out/demo"),
    max_quality_loss: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Maximum empirical development quality-score loss versus fallback.",
        ),
    ] = 0.02,
    objective: Annotated[
        str,
        typer.Option(callback=_objective, help="Optimization objective: balanced, cost, latency."),
    ] = "balanced",
) -> None:
    """Run the keyless, offline, explicitly synthetic product demonstration."""

    try:
        output.mkdir(parents=True, exist_ok=True)
        result = audit_demo(max_quality_loss=max_quality_loss, objective=objective)
        write_report(result, output, title="RouteFoundry synthetic demonstration")
        policy_value = result.get("policy")
        if not isinstance(policy_value, dict):
            raise ValueError("demo audit did not produce a policy object")
        policy = Policy.from_dict(policy_value)
        dump_policy(policy, output / "router.json")
        export_hf_chat_ui(policy.to_dict(), output / "hf-chat-ui-routes.json")
        export_human_policy(policy, output / "policy.txt")
        write_demo_jsonl(output / "demo-observations.jsonl")
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        _fail(exc)
    _print_audit_summary(result, output)
    console.print(
        "[yellow]Synthetic/illustrative only:[/yellow] this demo proves the workflow, "
        "not a calibrated or guaranteed real-world quality or savings claim."
    )


@app.command("validate")
def validate_command(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    require_signal: Annotated[
        bool,
        typer.Option(help="Require prompt text or a task label so a policy can be compiled."),
    ] = False,
) -> None:
    """Validate JSONL shape, ranges, uniqueness, metadata, and model-matrix completeness."""

    try:
        dataset = load_jsonl(input_path, require_signal=require_signal)
    except (OSError, ValueError, ValidationError) as exc:
        _fail(exc)
    console.print(
        f"[green]Valid[/green]: {dataset.prompt_count} prompts x "
        f"{dataset.model_count} models = {len(dataset)} observations"
    )


@app.command("audit")
def audit_command(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory for report, summary, and router artifacts."),
    ] = Path("out/audit"),
    max_quality_loss: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Maximum empirical development quality-score loss versus fallback.",
        ),
    ] = 0.02,
    objective: Annotated[
        str,
        typer.Option(callback=_objective, help="Optimization objective: balanced, cost, latency."),
    ] = "balanced",
    include_prompts: Annotated[
        bool,
        typer.Option(help="Include raw prompts in artifacts. Off by default for privacy."),
    ] = False,
) -> None:
    """Compile on development data and evaluate baselines once on held-out data."""

    try:
        dataset = load_jsonl(input_path, require_signal=True)
        result = audit(
            dataset,
            max_quality_loss=max_quality_loss,
            objective=objective,
        )
        rendered = result.to_dict()
        output.mkdir(parents=True, exist_ok=True)
        write_report(rendered, output, include_prompts=include_prompts)
        dump_policy(result.policy, output / "router.json")
        export_human_policy(result.policy, output / "policy.txt")
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        _fail(exc)
    _print_audit_summary(rendered, output)


@app.command("compile")
def compile_command(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination router JSON."),
    ] = Path("router.json"),
    max_quality_loss: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Maximum empirical development quality-score loss versus fallback.",
        ),
    ] = 0.02,
    objective: Annotated[
        str,
        typer.Option(callback=_objective, help="Optimization objective: balanced, cost, latency."),
    ] = "balanced",
) -> None:
    """Compile a small, explainable policy without producing the full audit report."""

    try:
        dataset = load_jsonl(input_path, require_signal=True)
        policy = compile_policy(
            dataset,
            max_quality_loss=max_quality_loss,
            objective=objective,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        dump_policy(policy, output)
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        _fail(exc)
    console.print(f"[green]Compiled[/green] {output.resolve()}")
    console.print(
        f"Fallback: [bold]{policy.fallback_model}[/bold] · pool: {', '.join(policy.pool)}"
    )


@app.command("route")
def route_command(
    policy_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    prompt: Annotated[str, typer.Argument(help="Prompt to classify locally. Not persisted.")],
    task: Annotated[str | None, typer.Option(help="Optional explicit task label.")] = None,
    warm_model: Annotated[
        str | None,
        typer.Option(help="Currently resident model, used for switch-cost hysteresis."),
    ] = None,
    switch_cost_ms: Annotated[
        float,
        typer.Option(min=0.0, help="Additional measured switch/eviction penalty in milliseconds."),
    ] = 0.0,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Explain one local routing decision without calling a model or network."""

    try:
        policy = load_policy(policy_path)
        decision = route(
            policy,
            prompt,
            task=task,
            warm_model=warm_model,
            switch_cost_ms=switch_cost_ms,
        )
    except (OSError, TypeError, ValueError) as exc:
        _fail(exc)
    value = decision.to_dict()
    if as_json:
        console.print_json(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return
    console.print(f"[bold cyan]{decision.model}[/bold cyan] · task={decision.task}")
    console.print(decision.reason)
    if decision.abstained:
        console.print("[yellow]Abstained to the strong fallback.[/yellow]")


@app.command("export")
def export_command(
    policy_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    format_name: Annotated[
        str,
        typer.Option("--format", help="Export format: hf-chat-ui or human."),
    ] = "hf-chat-ui",
) -> None:
    """Export a compiled policy to a supported downstream or review format."""

    try:
        policy = load_policy(policy_path)
        normalized = format_name.strip().lower()
        if normalized == "hf-chat-ui":
            export_hf_chat_ui(policy.to_dict(), output)
        elif normalized == "human":
            export_human_policy(policy, output)
        else:
            raise ValueError("format must be 'hf-chat-ui' or 'human'")
    except (OSError, TypeError, ValueError) as exc:
        _fail(exc)
    console.print(f"[green]Exported[/green] {normalized} to {output.resolve()}")


def _compiled_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    baselines = summary.get("baselines")
    if not isinstance(baselines, dict) or not isinstance(baselines.get("compiled"), dict):
        raise ValueError("baseline summary has no baselines.compiled object")
    return cast(dict[str, Any], baselines["compiled"])


@app.command("ci")
def ci_command(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    baseline: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option(help="Optional current summary JSON path.")] = None,
    max_quality_loss: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Held-out empirical quality-score loss threshold.",
        ),
    ] = 0.02,
    max_latency_regression: Annotated[
        float,
        typer.Option(min=0.0, help="Allowed fractional latency regression vs baseline."),
    ] = 0.05,
    max_cost_regression: Annotated[
        float,
        typer.Option(min=0.0, help="Allowed fractional cost regression vs baseline."),
    ] = 0.05,
    objective: Annotated[
        str,
        typer.Option(callback=_objective, help="Optimization objective: balanced, cost, latency."),
    ] = "balanced",
) -> None:
    """Fail CI on incompatible evidence or a held-out score/cost/latency regression."""

    try:
        previous = json.loads(baseline.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            raise ValueError("baseline must contain a JSON object")
        current_result = audit(
            load_jsonl(input_path, require_signal=True),
            max_quality_loss=max_quality_loss,
            objective=objective,
        ).to_dict()
        compatibility_fields = (
            "schema_version",
            "workload_fingerprint",
            "models",
            "seed",
            "objective",
            "max_quality_loss",
        )
        incompatible = [
            field
            for field in compatibility_fields
            if previous.get(field) != current_result.get(field)
        ]
        if incompatible:
            raise ValueError(
                "incompatible baseline; refusing metric comparison because these fields "
                f"differ or are missing: {', '.join(incompatible)}"
            )
        old = _compiled_metrics(previous)
        new = _compiled_metrics(current_result)
        failures: list[str] = []
        held_out = current_result.get("held_out_constraint")
        if not isinstance(held_out, dict):
            raise ValueError("current audit has no held_out_constraint object")
        if held_out.get("satisfied") is not True:
            failures.append(
                "held-out empirical quality-score loss is not supported within the "
                f"{max_quality_loss:.6f} threshold; observed="
                f"{float(held_out['observed_quality_loss']):.6f}, bootstrap CI upper="
                f"{float(held_out['quality_loss_ci_upper']):.6f}"
            )
        for field, allowance, label in (
            ("latency_ms", max_latency_regression, "latency"),
            ("cost_usd", max_cost_regression, "cost"),
        ):
            old_value = float(old[field])
            new_value = float(new[field])
            ceiling = old_value * (1.0 + allowance)
            if new_value > ceiling + 1e-12:
                failures.append(f"{label} {new_value:.6g} exceeds baseline ceiling {ceiling:.6g}")
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(current_result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    except (OSError, KeyError, TypeError, ValueError, ValidationError) as exc:
        _fail(exc)
    if failures:
        for failure in failures:
            error_console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(code=1)
    console.print(
        "[green]Compatible audit is within the configured empirical CI thresholds.[/green]"
    )


@app.command("ollama-profile")
def ollama_profile_command(
    models: Annotated[
        list[str] | None,
        typer.Argument(
            help="Optional positional existing Ollama models; comma-separated values work."
        ),
    ] = None,
    models_option: Annotated[
        str | None,
        typer.Option(
            "--models",
            help="Documented comma-separated list of already-installed Ollama models.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination profiling manifest."),
    ] = Path("out/ollama-profile.json"),
    repeats: Annotated[
        int,
        typer.Option(min=1, max=20, help="Backend-non-resident measurements per model."),
    ] = 3,
    base_url: Annotated[
        str,
        typer.Option(help="Trusted Ollama endpoint; localhost is the safe default."),
    ] = DEFAULT_BASE_URL,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes", help="Acknowledge temporary unload/load and best-effort restoration."
        ),
    ] = False,
) -> None:
    """Measure installed Ollama models without pulling, deleting, or recording responses."""

    model_groups = list(models or [])
    if models_option is not None:
        model_groups.append(models_option)
    selected = [
        item.strip()
        for group in model_groups
        for item in group.split(",")
        if item.strip()
    ]
    if not selected:
        _fail(ValueError("provide at least one installed model via --models or positionally"))
    if not yes:
        error_console.print(
            "Profiling temporarily changes model residency. Re-run with --yes after reading "
            "docs/OLLAMA_METHODOLOGY.md."
        )
        raise typer.Exit(code=2)
    try:
        manifest = profile_ollama_models(
            selected,
            output_path=output,
            repeats=repeats,
            base_url=base_url,
        )
    except (OSError, TypeError, ValueError, OllamaProfileError) as exc:
        _fail(exc)
    restore = manifest.get("residency_restore", {})
    console.print(f"[green]Profile written[/green] to {output.resolve()}")
    console.print(
        "Condition: backend-non-resident; OS cache uncontrolled. "
        f"Residency restoration: {restore.get('status', 'unknown')}."
    )


@app.command("autopilot")
def autopilot_command(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the observation matrix.")
    ] = Path("out/observations.jsonl"),
    models: Annotated[
        str | None,
        typer.Option("--models", help="Comma-separated model names. Default: every installed model."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Use a balanced subset of the suite (faster, less evidence)."),
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds allowed per generation.")
    ] = DEFAULT_TIMEOUT_SECONDS,
    base_url: Annotated[str, typer.Option("--base-url", help="Ollama endpoint.")] = DEFAULT_BASE_URL,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Reuse completed trials from a previous run.")
    ] = True,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the duration estimate confirmation.")
    ] = False,
) -> None:
    """Measure your installed Ollama models and build an observation matrix.

    Runs a bundled auto-gradable suite against each model, grades every answer
    deterministically, and writes observations that ``audit`` accepts. Nothing is pulled or
    deleted; only models already installed are measured.
    """

    try:
        suite = load_tasks()
        selected_tasks = select_tasks(suite, limit=limit)
        installed = discover_models(base_url=base_url)
    except (OSError, TaskSuiteError, OllamaProfileError, ValueError) as exc:
        _fail(exc)

    if not installed:
        error_console.print("[red]No models installed[/red]: Ollama reports an empty model list.")
        raise typer.Exit(code=2)

    available = {info.name for info in installed}
    if models:
        chosen = [name.strip() for name in models.split(",") if name.strip()]
        missing = [name for name in chosen if name not in available]
        if missing:
            error_console.print(f"[red]Not installed[/red]: {', '.join(missing)}")
            raise typer.Exit(code=2)
    else:
        chosen = [info.name for info in installed]

    estimate = estimate_duration_seconds(len(chosen), len(selected_tasks))
    console.print(
        f"Measuring [bold]{len(chosen)}[/bold] models x [bold]{len(selected_tasks)}[/bold] tasks "
        f"= {len(chosen) * len(selected_tasks)} generations."
    )
    console.print(
        f"Rough estimate: [bold]{format_duration(estimate)}[/bold] "
        "(laptop-CPU constants; a GPU fleet is much faster). Interrupting is safe: "
        "completed work is reused on the next run."
    )
    if not yes and not typer.confirm("Start the run?", default=True):
        raise typer.Exit(code=1)

    def _progress(update: RunProgress) -> None:
        mark = "[red]err[/red]" if update.error else ("[green]ok [/green]" if update.correct else "-- ")
        console.print(
            f"  [{update.index}/{update.total}] {mark} {update.model} "
            f"{update.task_id} {update.latency_ms / 1000:.1f}s",
            highlight=False,
        )

    try:
        stats = run_autopilot(
            chosen,
            selected_tasks,
            output,
            base_url=base_url,
            timeout=timeout,
            resume=resume,
            on_progress=_progress,
        )
    except (OSError, AutopilotError, OllamaProfileError) as exc:
        _fail(exc)

    table = Table(title="Autopilot: verifiable short-answer accuracy")
    table.add_column("Model")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")
    for model in chosen:
        total = stats.per_model_total.get(model, 0)
        correct = stats.per_model_correct.get(model, 0)
        share = f"{correct / total:.0%}" if total else "n/a"
        table.add_row(model, f"{correct}/{total}", share)
    console.print(table)

    console.print(
        f"[green]Matrix written[/green] to {output.resolve()} "
        f"({stats.measured} measured, {stats.resumed} reused, {stats.wall_seconds:.0f}s)."
    )
    if stats.dropped_prompts:
        console.print(
            f"[yellow]{len(stats.dropped_prompts)} prompt(s) dropped[/yellow] because at least one "
            "model failed to answer; an audit needs the same prompts for every model."
        )
    console.print(
        "These numbers measure verifiable short-answer accuracy on this suite and this "
        "hardware. They say nothing about open-ended generation quality. Next: "
        f"[bold]routefoundry audit {output}[/bold]"
    )


if __name__ == "__main__":
    app()
