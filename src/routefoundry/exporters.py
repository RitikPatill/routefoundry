"""Validated, deterministic policy exporters.

The Hugging Face exporter targets the documented Chat UI heuristic-router format.  Chat
UI does not perform arbitrary semantic task routing: it recognizes only ``default``,
``multimodal``, and ``agentic``.  This module refuses to claim otherwise.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HF_CHAT_UI_ROUTE_NAMES = ("default", "multimodal", "agentic")
_HF_ROUTE_FIELDS = frozenset({"name", "description", "primary_model", "fallback_models"})
_HF_REQUIRED_FIELDS = frozenset({"name", "description", "primary_model"})
_DEFAULT_DESCRIPTIONS = {
    "default": "General-purpose route for conversations without image or MCP inputs.",
    "multimodal": "Route selected by Hugging Face Chat UI for image inputs.",
    "agentic": "Route selected by Hugging Face Chat UI when MCP tools are active.",
}
_PRIVATE_KEYS = frozenset(
    {
        "prompt",
        "raw_prompt",
        "prompt_text",
        "response",
        "completion",
        "messages",
        "feature_counts",
        "token_counts",
        "train_prompt_ids",
        "dev_prompt_ids",
        "test_prompt_ids",
        "api_key",
        "token",
        "secret",
        "password",
    }
)


class ExportValidationError(ValueError):
    """Raised when an exporter input cannot satisfy its target schema."""


def _nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportValidationError(f"{field} must be a non-empty string")
    text = value.strip()
    if "\x00" in text:
        raise ExportValidationError(f"{field} must not contain NUL characters")
    return text


def validate_hf_chat_ui_routes(routes: Any) -> list[dict[str, Any]]:
    """Validate and normalize Hugging Face Chat UI's documented routes array."""

    if isinstance(routes, str | bytes) or not isinstance(routes, Sequence):
        raise ExportValidationError("Hugging Face Chat UI routes must be a JSON array")
    if not routes:
        raise ExportValidationError("at least one Hugging Face Chat UI route is required")

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, raw_route in enumerate(routes):
        if not isinstance(raw_route, Mapping):
            raise ExportValidationError(f"route {index} must be an object")
        fields = {str(key) for key in raw_route}
        missing = _HF_REQUIRED_FIELDS - fields
        extra = fields - _HF_ROUTE_FIELDS
        if missing:
            raise ExportValidationError(
                f"route {index} is missing required field(s): {', '.join(sorted(missing))}"
            )
        if extra:
            raise ExportValidationError(
                f"route {index} has unsupported field(s): {', '.join(sorted(extra))}"
            )

        name = _nonempty_text(raw_route["name"], field=f"route {index}.name")
        if name not in HF_CHAT_UI_ROUTE_NAMES:
            supported = ", ".join(HF_CHAT_UI_ROUTE_NAMES)
            raise ExportValidationError(
                f"unsupported Hugging Face Chat UI route {name!r}; use only {supported}"
            )
        if name in seen_names:
            raise ExportValidationError(f"duplicate Hugging Face Chat UI route: {name}")
        seen_names.add(name)

        description = _nonempty_text(raw_route["description"], field=f"route {index}.description")
        primary = _nonempty_text(raw_route["primary_model"], field=f"route {index}.primary_model")
        raw_fallbacks = raw_route.get("fallback_models", [])
        if isinstance(raw_fallbacks, str | bytes) or not isinstance(raw_fallbacks, Sequence):
            raise ExportValidationError(f"route {index}.fallback_models must be an array")
        fallbacks = [
            _nonempty_text(model, field=f"route {index}.fallback_models[{fallback_index}]")
            for fallback_index, model in enumerate(raw_fallbacks)
        ]
        if len(set(fallbacks)) != len(fallbacks):
            raise ExportValidationError(f"route {index}.fallback_models contains duplicates")
        if primary in fallbacks:
            raise ExportValidationError(
                f"route {index}.primary_model must not be repeated as a fallback"
            )
        normalized.append(
            {
                "name": name,
                "description": description,
                "primary_model": primary,
                "fallback_models": fallbacks,
            }
        )

    # Stable order makes diffs and reproducibility checks useful even if the input was a
    # mapping whose construction order came from another language.
    order = {name: index for index, name in enumerate(HF_CHAT_UI_ROUTE_NAMES)}
    normalized.sort(key=lambda route: order[str(route["name"])])
    return normalized


def _route_from_value(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "name": name,
            "description": _DEFAULT_DESCRIPTIONS[name],
            "primary_model": value,
            "fallback_models": [],
        }
    if not isinstance(value, Mapping):
        raise ExportValidationError(f"route {name!r} must be a model name or object")
    primary = value.get("primary_model", value.get("model", value.get("selected_model")))
    if primary is None:
        raise ExportValidationError(f"route {name!r} is missing primary_model")
    return {
        "name": name,
        "description": value.get("description", _DEFAULT_DESCRIPTIONS[name]),
        "primary_model": primary,
        "fallback_models": value.get("fallback_models", value.get("fallbacks", [])),
    }


def _routefoundry_chat_ui_routes(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    fallback = _nonempty_text(policy.get("fallback_model"), field="fallback_model")
    raw_pool = policy.get("pool", [])
    pool = (
        [str(model) for model in raw_pool if isinstance(model, str) and model != fallback]
        if isinstance(raw_pool, Sequence) and not isinstance(raw_pool, str | bytes)
        else []
    )
    # A compiled RouteFoundry policy currently measures observed task performance but
    # does not carry verified model capability declarations.  It would be unsafe to map
    # a high-scoring "multimodal" or "tool_use" aggregate to Chat UI's capability routes.
    # Emit only the safe default until a future schema proves those capabilities.
    routes: list[dict[str, Any]] = [
        {
            "name": "default",
            "description": (
                "Safe fallback from the RouteFoundry policy. Chat UI cannot represent "
                "arbitrary semantic task rules."
            ),
            "primary_model": fallback,
            "fallback_models": pool,
        }
    ]
    if not isinstance(policy.get("routes", {}), Mapping):
        raise ExportValidationError("RouteFoundry policy routes must be an object")
    return validate_hf_chat_ui_routes(routes)


def build_hf_chat_ui_routes(
    policy: Mapping[str, Any] | Sequence[Any] | Any,
) -> list[dict[str, Any]]:
    """Build a valid Chat UI routes array from explicit policy route information.

    Accepted inputs are an already-shaped routes array, a mapping with
    ``hf_chat_ui_routes``/``chat_ui_routes``, or a ``routes`` mapping keyed by the three
    supported names.  A generic compiled policy can export its fallback as a default
    route. Arbitrary semantic tasks are not relabelled as Chat UI routes, and compiled
    multimodal/tool routes are not exported until model capability metadata is verified.
    """

    to_dict = getattr(policy, "to_dict", None)
    if callable(to_dict):
        policy = to_dict()
    if not isinstance(policy, Mapping):
        return validate_hf_chat_ui_routes(policy)

    if policy.get("schema_version") == "routefoundry.policy.v1":
        return _routefoundry_chat_ui_routes(policy)

    for key in ("hf_chat_ui_routes", "chat_ui_routes"):
        if key in policy:
            return validate_hf_chat_ui_routes(policy[key])

    raw_routes = policy.get("routes")
    if isinstance(raw_routes, Sequence) and not isinstance(raw_routes, str | bytes):
        return validate_hf_chat_ui_routes(raw_routes)
    if isinstance(raw_routes, Mapping):
        unknown = {str(name) for name in raw_routes} - set(HF_CHAT_UI_ROUTE_NAMES)
        if unknown:
            raise ExportValidationError(
                "Hugging Face Chat UI cannot represent semantic route(s): "
                + ", ".join(sorted(unknown))
            )
        return validate_hf_chat_ui_routes(
            [
                _route_from_value(name, raw_routes[name])
                for name in HF_CHAT_UI_ROUTE_NAMES
                if name in raw_routes
            ]
        )

    # Compiled RouteFoundry policies normally contain task-specific rules.  Chat UI
    # cannot express them, but exporting the configured safe fallback as its default
    # route is truthful and useful.
    default_model = policy.get(
        "default_model",
        policy.get("fallback_model", policy.get("strong_model")),
    )
    if default_model is None:
        raise ExportValidationError(
            "policy needs explicit Chat UI routes or a default/fallback/strong model"
        )
    fallback_models = policy.get("fallback_models", [])
    return validate_hf_chat_ui_routes(
        [
            {
                "name": "default",
                "description": _DEFAULT_DESCRIPTIONS["default"],
                "primary_model": default_model,
                "fallback_models": fallback_models,
            }
        ]
    )


def export_hf_chat_ui(
    policy: Mapping[str, Any] | Sequence[Any],
    output: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Validate a policy, optionally write routes JSON, and return the normalized array."""

    routes = build_hf_chat_ui_routes(policy)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(routes, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    return routes


def write_hf_chat_ui_routes(policy: Mapping[str, Any] | Sequence[Any], output: str | Path) -> Path:
    """Write Chat UI routes JSON and return the output path."""

    path = Path(output)
    export_hf_chat_ui(policy, path)
    return path


def _public_policy(value: Any) -> Any:
    if isinstance(value, Mapping):
        public: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.strip().lower().replace("-", "_")
            credential_like = normalized.endswith(("_token", "_secret", "_password", "_key"))
            if normalized in _PRIVATE_KEYS or credential_like:
                continue
            public[key] = _public_policy(item)
        return public
    if isinstance(value, list | tuple):
        return [_public_policy(item) for item in value]
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExportValidationError("policy values must not contain NaN or infinity")
        return value
    return str(value)


def _render_tree(value: Any, *, indent: int = 0) -> list[str]:
    prefix = "  " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key in sorted(value):
            item = value[key]
            if isinstance(item, Mapping | list):
                lines.append(f"{prefix}{key}:")
                lines.extend(_render_tree(item, indent=indent + 1))
            else:
                lines.append(f"{prefix}{key}: {json.dumps(item, ensure_ascii=False)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}(none)"]
        lines = []
        for item in value:
            if isinstance(item, Mapping | list):
                lines.append(f"{prefix}-")
                lines.extend(_render_tree(item, indent=indent + 1))
            else:
                lines.append(f"{prefix}- {json.dumps(item, ensure_ascii=False)}")
        return lines
    return [f"{prefix}{json.dumps(value, ensure_ascii=False)}"]


def render_policy(policy: Mapping[str, Any] | Any) -> str:
    """Render a review view without prompts, credentials, IDs, or classifier features."""

    to_dict = getattr(policy, "to_dict", None)
    if callable(to_dict):
        policy = to_dict()
    if not isinstance(policy, Mapping):
        raise ExportValidationError("policy must be an object")
    public = _public_policy(policy)
    lines = [
        "RouteFoundry policy",
        "===================",
        "",
        "This is a human-readable representation, not an executable configuration.",
        "Raw prompts, model responses, messages, and credential-like fields are omitted.",
        "Unknown tasks abstain to the policy's configured fallback model.",
        "",
    ]
    lines.extend(_render_tree(public))
    return "\n".join(lines).rstrip() + "\n"


def export_human_policy(policy: Mapping[str, Any] | Any, output: str | Path | None = None) -> str:
    """Render a policy and optionally write the standalone text representation."""

    rendered = render_policy(policy)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    return rendered


def write_human_policy(policy: Mapping[str, Any] | Any, output: str | Path) -> Path:
    path = Path(output)
    export_human_policy(policy, path)
    return path


__all__ = [
    "HF_CHAT_UI_ROUTE_NAMES",
    "ExportValidationError",
    "build_hf_chat_ui_routes",
    "export_hf_chat_ui",
    "export_human_policy",
    "render_policy",
    "validate_hf_chat_ui_routes",
    "write_hf_chat_ui_routes",
    "write_human_policy",
]
