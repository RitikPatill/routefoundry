from __future__ import annotations

import json

from routefoundry.classify import TaskClassifier
from routefoundry.policy import ExpectedMetrics, Policy, RouteRule, dump_policy
from routefoundry.report import prepare_summary, render_report
from routefoundry.schema import Observation, validate_observations
from routefoundry.split import split_observations

CANARY_WORD = "ZyzzyvaPromptCanary8472"
CANARY_ID_PREFIX = "private-customer-ticket-9931"
CREDENTIAL_CANARY = "credential-value-must-never-reach-public-report"


def _classifier() -> TaskClassifier:
    observations = [
        Observation(
            prompt_id=f"{CANARY_ID_PREFIX}-{index}",
            model="small",
            quality=0.9,
            latency_ms=10.0,
            cost_usd=0.01,
            prompt=f"{CANARY_WORD} debug the function",
            task="code",
        )
        for index in range(2)
    ]
    return TaskClassifier.fit(observations)


def _policy(classifier: TaskClassifier) -> Policy:
    metrics = {
        model: ExpectedMetrics(quality, latency, cost, samples=2)
        for model, quality, latency, cost in (
            ("small", 0.94, 20.0, 0.01),
            ("strong", 0.95, 50.0, 0.10),
        )
    }
    return Policy(
        fallback_model="strong",
        objective="cost",
        max_quality_loss=0.02,
        pool=("small", "strong"),
        routes={"code": RouteRule("code", "small", 0.94, 0.01, "within quality budget")},
        model_metrics={"code": metrics, "__global__": metrics},
        classifier=classifier,
    )


def test_compiled_policy_hashes_prompt_vocabulary_and_still_routes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    classifier = _classifier()
    decision = classifier.classify(f"Please {CANARY_WORD} debug this function")
    assert decision.task == "code"
    assert not decision.abstained
    assert "uncalibrated" in decision.reason
    assert "evidence_score" in decision.to_dict()
    assert "confidence" not in decision.to_dict()

    destination = tmp_path / "policy.json"
    dump_policy(_policy(classifier), destination)
    serialized = destination.read_text(encoding="utf-8")
    lowered = serialized.casefold()
    assert CANARY_WORD.casefold() not in lowered
    assert CANARY_ID_PREFIX.casefold() not in lowered
    assert '"type": "hashed-feature-evidence-v1"' in serialized
    assert '"feature_counts"' in serialized
    assert '"token_counts"' not in serialized


def test_default_public_artifacts_omit_prompt_ids_text_and_classifier_features() -> None:
    classifier = _classifier()
    rows = [
        {
            "prompt_id": f"{CANARY_ID_PREFIX}-{index}",
            "prompt": f"{CANARY_WORD} private prompt {index}",
            "task": "code",
            "model": "small",
            "quality": 0.9,
            "latency_ms": 10.0,
            "cost_usd": 0.01,
        }
        for index in range(3)
    ]
    split = split_observations(validate_observations(rows), seed=9)
    split_json = json.dumps(split.to_dict(), sort_keys=True)
    assert CANARY_ID_PREFIX.casefold() not in split_json.casefold()
    assert split.to_dict()["prompt_counts"] == {"train": 1, "development": 1, "test": 1}

    source = {
        "split": split.to_dict(),
        "policy": {"classifier": classifier.to_dict()},
        "prompt": f"{CANARY_WORD} private prompt",
        "train_prompt_ids": [f"{CANARY_ID_PREFIX}-0"],
        "nested": {
            "responses": [f"{CANARY_WORD} private response"],
            "messages": [{"content": f"{CANARY_WORD} private message"}],
            "model_response": f"{CANARY_WORD} aliased response",
            "chat_messages": [f"{CANARY_WORD} aliased message"],
            "per_prompt_metrics": {f"{CANARY_ID_PREFIX}-0": {"quality": 0.9}},
            "query": f"{CANARY_WORD} aliased query",
            "instruction": f"{CANARY_WORD} aliased instruction",
            "api_key": CREDENTIAL_CANARY,
            "auth": {
                "access_token": CREDENTIAL_CANARY,
                "Authorization": CREDENTIAL_CANARY,
                "session_cookie": CREDENTIAL_CANARY,
            },
            "public_metric": 0.9,
        },
    }
    summary = prepare_summary(source)
    summary_json = json.dumps(summary, sort_keys=True)
    document = render_report(source)
    for artifact in (summary_json, document):
        lowered = artifact.casefold()
        assert CANARY_WORD.casefold() not in lowered
        assert CANARY_ID_PREFIX.casefold() not in lowered
        assert "feature_counts" not in lowered
        assert "token_counts" not in lowered
        assert CREDENTIAL_CANARY not in artifact
    assert summary["nested"] == {"auth": {}, "public_metric": 0.9}
    assert summary["report_metadata"]["classifier_features_included"] is False


def test_include_prompts_is_explicit_but_feature_tables_remain_private() -> None:
    source = {
        "prompt": CANARY_WORD,
        "prompt_id": CANARY_ID_PREFIX,
        "response": f"response for {CANARY_WORD}",
        "password": CREDENTIAL_CANARY,
        "classifier": _classifier().to_dict(),
    }
    summary = prepare_summary(source, include_prompts=True)
    serialized = json.dumps(summary, sort_keys=True)
    assert CANARY_WORD in serialized
    assert CANARY_ID_PREFIX in serialized
    assert "feature_counts" not in serialized
    assert CREDENTIAL_CANARY not in serialized
