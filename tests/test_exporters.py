from __future__ import annotations

import json

import pytest
from routefoundry.exporters import (
    ExportValidationError,
    build_hf_chat_ui_routes,
    export_hf_chat_ui,
    render_policy,
    validate_hf_chat_ui_routes,
)


def _routes() -> list[dict[str, object]]:
    return [
        {
            "name": "agentic",
            "description": "Tool use",
            "primary_model": "tool-model",
            "fallback_models": ["strong-model"],
        },
        {
            "name": "default",
            "description": "Everything else",
            "primary_model": "small-model",
            "fallback_models": ["strong-model"],
        },
    ]


def test_hf_export_matches_documented_schema_and_order(tmp_path) -> None:
    target = tmp_path / "routes.json"
    result = export_hf_chat_ui(_routes(), target)

    assert [route["name"] for route in result] == ["default", "agentic"]
    assert json.loads(target.read_text(encoding="utf-8")) == result
    assert set(result[0]) == {
        "name",
        "description",
        "primary_model",
        "fallback_models",
    }


@pytest.mark.parametrize(
    "routes,match",
    [
        (
            [
                {
                    "name": "coding",
                    "description": "Unsupported semantic route",
                    "primary_model": "model",
                }
            ],
            "unsupported Hugging Face Chat UI route",
        ),
        (
            [
                {
                    "name": "default",
                    "description": "Description",
                    "primary_model": "model",
                    "unknown": True,
                }
            ],
            "unsupported field",
        ),
        (
            [
                {
                    "name": "default",
                    "description": "Description",
                    "primary_model": "model",
                    "fallback_models": ["model"],
                }
            ],
            "must not be repeated",
        ),
    ],
)
def test_hf_schema_rejects_unrepresentable_or_invalid_routes(
    routes: list[dict[str, object]], match: str
) -> None:
    with pytest.raises(ExportValidationError, match=match):
        validate_hf_chat_ui_routes(routes)


def test_hf_builder_uses_only_explicit_supported_mapping_routes() -> None:
    routes = build_hf_chat_ui_routes(
        {
            "routes": {
                "default": "small-model",
                "multimodal": {
                    "model": "vision-model",
                    "fallbacks": ["strong-model"],
                },
            }
        }
    )

    assert routes[0]["primary_model"] == "small-model"
    assert routes[1]["name"] == "multimodal"
    assert routes[1]["fallback_models"] == ["strong-model"]


def test_generic_policy_exports_only_truthful_default_fallback() -> None:
    routes = build_hf_chat_ui_routes(
        {
            "fallback_model": "strong-model",
            "task_routes": {"coding": "coder-model"},
        }
    )

    assert [route["name"] for route in routes] == ["default"]
    assert routes[0]["primary_model"] == "strong-model"


def test_human_policy_is_standalone_deterministic_and_private() -> None:
    policy = {
        "fallback_model": "strong-model",
        "prompt": "private prompt",
        "api_key": "hf_do_not_print",
        "classifier": {
            "feature_counts": {"coding": {"canary-feature-bucket": 99}},
            "token_counts": {"coding": {"legacy-canary-token": 99}},
        },
        "rules": {"coding": {"model": "coder-model", "max_loss": 0.02}},
    }

    first = render_policy(policy)
    second = render_policy(policy)
    assert first == second
    assert "not an executable configuration" in first
    assert 'fallback_model: "strong-model"' in first
    assert "private prompt" not in first
    assert "hf_do_not_print" not in first
    assert "canary-feature-bucket" not in first
    assert "legacy-canary-token" not in first


def test_exporters_accept_production_policy_object() -> None:
    from routefoundry.demo import make_demo_dataset
    from routefoundry.optimize import audit

    policy = audit(make_demo_dataset(prompts_per_task=6)).policy
    routes = build_hf_chat_ui_routes(policy)
    text = render_policy(policy)

    assert routes[0]["name"] == "default"
    assert [route["name"] for route in routes] == ["default"]
    assert "routefoundry.policy.v1" in text
