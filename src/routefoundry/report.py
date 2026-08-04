"""Privacy-preserving, dependency-free audit reports.

The renderer deliberately produces boring HTML.  A report should remain readable when
opened from disk, should not execute supplied data, and should not need a CDN.  All
dynamic values pass through :func:`html.escape`; no values are inserted into scripts or
attributes.
"""

from __future__ import annotations

import dataclasses
import html
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = "1.0"

# These values commonly contain user/model text or stable source identifiers.  They are
# removed recursively unless a caller makes the deliberate, visible choice to include
# prompts.  Match normalized field names, not values, so ordinary metric strings remain
# useful in third-party audit mappings.
_EXPLICIT_PROMPT_DATA_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "raw_prompt",
        "raw_prompts",
        "prompt_text",
        "response",
        "responses",
        "raw_response",
        "raw_responses",
        "completion",
        "completions",
        "message",
        "messages",
        "prompt_id",
        "prompt_ids",
        "query",
        "queries",
        "question",
        "questions",
        "instruction",
        "instructions",
        "input",
        "inputs",
        "model_input",
        "model_inputs",
        "output",
        "outputs",
        "answer",
        "answers",
        "generated_text",
        "generated_texts",
        "document",
        "documents",
    }
)

_CREDENTIAL_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "client_secret",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "password",
        "refresh_token",
        "secret",
        "set_cookie",
        "token",
    }
)

# Hashed classifier buckets are required by an executable policy, but they are not
# useful in a public report and hashes are not anonymous.  Keep them out even when raw
# prompts are explicitly requested.
_CLASSIFIER_INTERNAL_KEYS = frozenset(
    {
        "feature_counts",
        "token_counts",  # legacy policy shape
        "classifier_feature_counts",
        "classifier_token_counts",
        "classifier_vocabulary",
        "vocabulary",
    }
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths written by :func:`write_report`."""

    html_path: Path
    summary_path: Path


def _plain(value: Any) -> Any:
    """Convert common audit containers to JSON-compatible Python values."""

    # AuditResult and Policy deliberately expose stable public dictionaries.  Prefer
    # those over dataclasses.asdict(), which cannot deep-copy some read-only mappings.
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted((_plain(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, Path | Enum):
        return str(value.value if isinstance(value, Enum) else value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _is_prompt_data_key(normalized: str, value: Any) -> bool:
    identifier_container = normalized in {
        "per_prompt",
        "per_prompt_metrics",
        "prompt_assignments",
        "prompt_metrics",
        "assignments_by_prompt",
    }
    text_suffix = normalized.endswith(
        (
            "_prompt",
            "_prompt_text",
            "_response",
            "_responses",
            "_response_text",
            "_message",
            "_messages",
            "_message_text",
            "_completion",
            "_completions",
            "_completion_text",
        )
    )
    # ``train_prompts`` and similar fields are useful numeric aggregate counts.  Lists
    # or mappings under a plural prompt key are source data and remain opt-in.
    plural_prompts = normalized.endswith("_prompts") and not isinstance(value, int | float)
    return (
        normalized in _EXPLICIT_PROMPT_DATA_KEYS
        or normalized.endswith(("_prompt_id", "_prompt_ids"))
        or normalized.startswith(("raw_prompt_", "raw_response_", "raw_message_"))
        or normalized.endswith(("_raw_prompt", "_raw_response", "_raw_message"))
        or identifier_container
        or text_suffix
        or plural_prompts
    )


def _is_classifier_internal_key(normalized: str) -> bool:
    return normalized in _CLASSIFIER_INTERNAL_KEYS or normalized.endswith(
        ("_feature_counts", "_token_counts")
    )


def _is_credential_key(normalized: str) -> bool:
    """Recognize common credential field names without inspecting secret values."""

    return normalized in _CREDENTIAL_KEYS or normalized.endswith(
        (
            "_api_key",
            "_authorization",
            "_cookie",
            "_credentials",
            "_password",
            "_secret",
            "_token",
        )
    )


def _sanitize_public(value: Any, *, include_prompts: bool) -> Any:
    """Build the report's recursive public view.

    Prompt data is opt-in.  Classifier feature tables are always report-private because
    deterministic feature hashes reduce accidental disclosure but do not anonymize the
    source vocabulary.
    """

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if _is_credential_key(normalized):
                continue
            if _is_classifier_internal_key(normalized):
                continue
            if not include_prompts and _is_prompt_data_key(normalized, item):
                continue
            cleaned[str(key)] = _sanitize_public(item, include_prompts=include_prompts)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_public(item, include_prompts=include_prompts) for item in value]
    return value


def _walk(value: Any) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []

    def visit(item: Any, key: str = "") -> None:
        values.append((key, item))
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif isinstance(item, list):
            for child in item:
                visit(child, key)

    visit(value)
    return values


def _labels(value: Mapping[str, Any]) -> list[str]:
    synthetic = False
    oracle = False
    explicitly_non_evidence = False
    for key, item in _walk(value):
        key_text = key.lower().replace("-", "_")
        item_text = str(item).lower() if isinstance(item, str) else ""
        if "oracle" in key_text or "oracle" in item_text:
            oracle = True
        if "synthetic" in key_text or "illustrative" in key_text:
            synthetic = item is not False
        if "synthetic" in item_text or "illustrative" in item_text:
            synthetic = True
        if "non_evidence" in key_text or "non-evidence" in item_text:
            explicitly_non_evidence = item is not False

    labels: list[str] = []
    if synthetic:
        labels.append("SYNTHETIC / ILLUSTRATIVE — NOT EMPIRICAL EVIDENCE")
    if oracle:
        labels.append("HINDSIGHT ORACLE — NON-DEPLOYABLE UPPER BOUND")
    if explicitly_non_evidence and not synthetic:
        labels.append("NON-EVIDENCE DATA")
    if not labels:
        labels.append("USER-SUPPLIED EVALUATION DATA — PROVENANCE NOT VERIFIED")
    return labels


def prepare_summary(
    audit_result: Mapping[str, Any] | Any,
    *,
    include_prompts: bool = False,
) -> dict[str, Any]:
    """Return the JSON summary used by both report artifacts.

    ``audit_result`` may be any mapping, a dataclass, or a small object exposing
    ``to_dict``.  The mapping contract lets the report remain decoupled from the audit
    implementation and is especially convenient for third-party integrations.
    """

    plain = _plain(audit_result)
    if not isinstance(plain, Mapping):
        raise TypeError("audit_result must be a mapping or convert to one")
    cleaned = _sanitize_public(plain, include_prompts=include_prompts)
    assert isinstance(cleaned, dict)  # narrowed by the mapping check above
    metadata = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "raw_prompts_included": include_prompts,
        "data_labels": _labels(cleaned),
        "privacy_note": (
            (
                "Raw prompts, prompt identifiers, model responses, and messages are "
                "included by explicit request."
            )
            if include_prompts
            else (
                "Raw prompts, prompt identifiers, model responses, and messages are "
                "omitted by default. Recognized credential fields are always omitted; "
                "this is not a data-loss-prevention guarantee."
            )
        ),
        "classifier_features_included": False,
    }
    # A dedicated namespace prevents accidental collision with metrics produced by an
    # audit engine while keeping the engine's mapping intact.
    cleaned["report_metadata"] = metadata
    return cleaned


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _display(value: Any, *, digits: int | None = None) -> str:
    """Format, then escape, a value for visible HTML text."""

    if value is None:
        text = "—"
    elif isinstance(value, bool):
        text = "yes" if value else "no"
    elif digits is not None and isinstance(value, int | float) and not isinstance(value, bool):
        text = f"{float(value):.{digits}f}"
    elif isinstance(value, list):
        text = ", ".join(str(item) for item in value) if value else "(none)"
    else:
        text = str(value)
    return html.escape(text, quote=True)


def _render_kpis(summary: Mapping[str, Any]) -> str:
    counts = _mapping(summary.get("counts"))
    policy = _mapping(summary.get("policy"))
    pool = summary.get("recommended_pool")
    if pool is None:
        pool = policy.get("pool")
    strongest = summary.get("strongest_model", policy.get("fallback_model"))
    prompt_parts = []
    for key, label in (
        ("train_prompts", "train"),
        ("development_prompts", "dev"),
        ("test_prompts", "test"),
    ):
        if key in counts:
            prompt_parts.append(f"{counts[key]} {label}")
    prompt_text = ", ".join(prompt_parts) if prompt_parts else counts.get("prompts")
    cards = (
        ("Objective", summary.get("objective"), None),
        ("Quality-loss budget", summary.get("max_quality_loss"), 4),
        ("Prompt split", prompt_text, None),
        ("Models evaluated", counts.get("models"), None),
        ("Recommended pool", pool, None),
        ("Strongest / fallback model", strongest, None),
    )
    return "".join(
        '<article class="kpi">'
        f'<span class="kpi-label">{html.escape(label, quote=True)}</span>'
        f"<strong>{_display(value, digits=digits)}</strong>"
        "</article>"
        for label, value, digits in cards
    )


def _render_baselines(summary: Mapping[str, Any]) -> str:
    raw_baselines = summary.get("baselines")
    rows: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw_baselines, Mapping):
        rows = [(str(name), _mapping(raw_baselines[name])) for name in sorted(raw_baselines)]
    elif isinstance(raw_baselines, list):
        rows = [
            (str(_mapping(item).get("name", f"baseline-{index + 1}")), _mapping(item))
            for index, item in enumerate(raw_baselines)
        ]
    if not rows:
        return '<p class="empty">No baseline comparison was supplied.</p>'

    body: list[str] = []
    for name, metrics in rows:
        deployable = metrics.get("deployable")
        status = "deployable" if deployable is not False else "hindsight / non-deployable"
        if "oracle" in name.casefold():
            status = "hindsight oracle / non-deployable"
        cells = (
            _display(name),
            _display(status),
            _display(metrics.get("quality"), digits=4),
            _display(metrics.get("quality_loss"), digits=4),
            _display(metrics.get("latency_ms"), digits=2),
            _display(metrics.get("latency_p95_ms"), digits=2),
            _display(metrics.get("cost_usd"), digits=6),
            _display(metrics.get("coverage"), digits=4),
            _display(metrics.get("switch_count")),
        )
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return (
        """<div class="table-wrap"><table>
      <thead><tr>
        <th>Baseline</th><th>Status</th><th>Quality</th><th>Quality loss</th>
        <th>Mean latency (ms)</th><th>p95 latency (ms)</th><th>Cost (USD)</th>
        <th>Coverage</th><th>Switches</th>
      </tr></thead>
      <tbody>"""
        + "".join(body)
        + "</tbody></table></div>"
    )


def _render_routes(summary: Mapping[str, Any]) -> str:
    policy = _mapping(summary.get("policy"))
    raw_routes = policy.get("routes")
    rows: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw_routes, Mapping):
        rows = [(str(task), _mapping(raw_routes[task])) for task in sorted(raw_routes)]
    elif isinstance(raw_routes, list):
        rows = [
            (str(_mapping(item).get("task", f"route-{index + 1}")), _mapping(item))
            for index, item in enumerate(raw_routes)
        ]
    if not rows:
        return '<p class="empty">No task-specific routes were supplied; use the fallback.</p>'

    body = []
    for task, route in rows:
        cells = (
            _display(task),
            _display(route.get("model", route.get("primary_model"))),
            _display(route.get("expected_quality"), digits=4),
            _display(route.get("quality_loss"), digits=4),
            _display(route.get("reason")),
        )
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return (
        """<div class="table-wrap"><table>
      <thead><tr>
        <th>Task / signal</th><th>Selected model</th><th>Expected quality</th>
        <th>Quality loss</th><th>Why this route</th>
      </tr></thead>
      <tbody>"""
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_report(
    audit_result: Mapping[str, Any] | Any,
    *,
    include_prompts: bool = False,
    title: str = "RouteFoundry audit report",
) -> str:
    """Render a self-contained static HTML report.

    The resulting document contains no JavaScript, remote resources, forms, iframes, or
    unescaped dynamic markup.
    """

    summary = prepare_summary(audit_result, include_prompts=include_prompts)
    metadata = summary["report_metadata"]
    labels = "".join(
        f'<li class="label">{html.escape(str(label), quote=True)}</li>'
        for label in metadata["data_labels"]
    )
    payload = html.escape(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False),
        quote=True,
    )
    safe_title = html.escape(str(title), quote=True)
    privacy_note = html.escape(str(metadata["privacy_note"]), quote=True)
    kpis = _render_kpis(summary)
    baselines = _render_baselines(summary)
    routes = _render_routes(summary)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ max-width: 74rem; margin: 0 auto; padding: 2rem; line-height: 1.55; }}
    header, section {{ border: 1px solid #8886; border-radius: .75rem; padding: 1rem 1.25rem;
      margin-bottom: 1rem; }}
    h1 {{ margin: 0 0 .4rem; }}
    .label {{ font-weight: 700; color: #b45309; margin: .25rem 0; }}
    .privacy {{ font-weight: 600; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
      gap: .75rem; }}
    .kpi {{ border: 1px solid #8885; border-radius: .6rem; padding: .8rem;
      min-width: 0; }}
    .kpi-label {{ display: block; font-size: .75rem; text-transform: uppercase;
      letter-spacing: .04em; opacity: .75; margin-bottom: .3rem; }}
    .kpi strong {{ display: block; overflow-wrap: anywhere; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: .875rem; }}
    th, td {{ border-bottom: 1px solid #8885; padding: .55rem; text-align: left;
      vertical-align: top; }}
    th {{ white-space: nowrap; }}
    .empty {{ opacity: .75; font-style: italic; }}
    details summary {{ cursor: pointer; font-weight: 700; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: .875rem; }}
    footer {{ opacity: .75; font-size: .875rem; }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <p class="privacy">{privacy_note}</p>
    <ul>{labels}</ul>
  </header>
  <section aria-labelledby="overview-heading">
    <h2 id="overview-heading">Audit at a glance</h2>
    <div class="kpis">{kpis}</div>
  </section>
  <section aria-labelledby="baselines-heading">
    <h2 id="baselines-heading">Baseline comparison</h2>
    {baselines}
  </section>
  <section aria-labelledby="routes-heading">
    <h2 id="routes-heading">Compiled policy routes</h2>
    {routes}
  </section>
  <section aria-labelledby="audit-data-heading">
    <h2 id="audit-data-heading">Reproducibility payload</h2>
    <details>
      <summary>Show full sanitized summary JSON</summary>
      <pre>{payload}</pre>
    </details>
  </section>
  <footer>Generated locally by RouteFoundry. No external resources or scripts are used.</footer>
</body>
</html>
"""


def write_report(
    audit_result: Mapping[str, Any] | Any,
    output: str | Path,
    *,
    include_prompts: bool = False,
    title: str = "RouteFoundry audit report",
    html_name: str = "report.html",
    summary_name: str = "summary.json",
) -> ReportArtifacts:
    """Write ``report.html`` and ``summary.json`` and return their paths.

    ``output`` may be a directory or a direct ``.html`` filename.  In the latter case,
    ``summary.json`` is written next to it.  Parent directories are created as needed.
    """

    target = Path(output)
    if target.suffix.lower() == ".html":
        html_path = target
        summary_path = target.with_name(summary_name)
    else:
        html_path = target / html_name
        summary_path = target / summary_name
    html_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = prepare_summary(audit_result, include_prompts=include_prompts)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(
        render_report(
            audit_result,
            include_prompts=include_prompts,
            title=title,
        ),
        encoding="utf-8",
    )
    return ReportArtifacts(html_path=html_path, summary_path=summary_path)


# A discoverable alias for callers that naturally search for a generator function.
generate_report = write_report


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ReportArtifacts",
    "generate_report",
    "prepare_summary",
    "render_report",
    "write_report",
]
