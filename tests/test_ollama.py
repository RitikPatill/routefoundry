from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from routefoundry.ollama import (
    ModelNotFoundError,
    OllamaAPIError,
    OllamaProfiler,
    nanoseconds_to_milliseconds,
    profile_ollama_models,
)


class FakeOllama:
    def __init__(
        self,
        *,
        installed: tuple[str, ...] = ("tiny:latest",),
        resident: tuple[str, ...] = (),
        evict_on_profile: bool = False,
    ) -> None:
        self.installed = installed
        self.resident = set(resident)
        self.evict_on_profile = evict_on_profile
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []
        self.profile_bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))

        if request.method == "GET" and request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": name,
                            "model": name,
                            "digest": f"sha256:{index:064x}",
                            "size": index * 1_000_000,
                            "details": {
                                "format": "gguf",
                                "family": "test-family",
                                "parameter_size": "1B",
                                "quantization_level": "Q4_TEST",
                                "untrusted_extra": "MUST_NOT_BE_COPIED",
                            },
                        }
                        for index, name in enumerate(self.installed, start=1)
                    ]
                },
            )
        if request.method == "GET" and request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={"models": [{"name": name, "model": name} for name in self.resident]},
            )
        if request.method == "GET" and request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.test"})
        if request.method == "POST" and request.url.path == "/api/generate":
            assert body is not None
            model = body["model"]
            if body.get("keep_alive") == 0:
                self.resident.discard(model)
                return httpx.Response(200, json={"done": True})
            if "prompt" not in body:
                self.resident.add(model)
                return httpx.Response(200, json={"done": True})

            self.profile_bodies.append(body)
            if self.evict_on_profile:
                self.resident.clear()
            self.resident.add(model)
            return httpx.Response(
                200,
                json={
                    "response": "BACKEND_SECRET_MUST_NOT_BE_RECORDED",
                    "done": True,
                    "total_duration": 12_500_000,
                    "load_duration": 7_250_000,
                    "prompt_eval_duration": 3_000_000,
                    "eval_duration": 2_250_000,
                    "prompt_eval_count": 4,
                    "eval_count": 1,
                },
            )
        return httpx.Response(404, json={"error": "unexpected endpoint"})


class EventuallyConsistentUnloadOllama(FakeOllama):
    """Return one stale process-list read after acknowledging an unload."""

    def __init__(self) -> None:
        super().__init__()
        self._stale_model: str | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        if (
            request.method == "POST"
            and request.url.path == "/api/generate"
            and body is not None
            and body.get("keep_alive") == 0
            and body["model"] in self.resident
        ):
            self._stale_model = str(body["model"])
        if (
            request.method == "GET"
            and request.url.path == "/api/ps"
            and self._stale_model is not None
        ):
            stale = self._stale_model
            self._stale_model = None
            self.requests.append((request.method, request.url.path, body))
            self.resident.discard(stale)
            return httpx.Response(200, json={"models": [{"name": stale, "model": stale}]})
        return super().__call__(request)


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama.test")


def test_profile_records_backend_metrics_and_writes_manifest(tmp_path: Path) -> None:
    fake = FakeOllama()
    output = tmp_path / "profile.json"

    with make_client(fake) as client:
        manifest = profile_ollama_models(
            ["tiny:latest"], client=client, repeats=2, output_path=output
        )

    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert manifest["schema_version"] == "1.0"
    assert manifest["measurement"] == {
        "residency_condition": "backend-non-resident",
        "os_cache_condition": "OS-cache-uncontrolled",
        "repeats_per_model": 2,
    }
    observations = manifest["models"][0]["observations"]
    assert len(observations) == 2
    assert observations[0]["load_duration_ns"] == 7_250_000
    assert observations[0]["load_duration_ms"] == 7.25
    assert observations[0]["total_duration_ms"] == 12.5
    assert observations[0]["prompt_eval_count"] == 4
    assert observations[0]["eval_count"] == 1
    assert manifest["models"][0]["digest"].startswith("sha256:")
    assert manifest["models"][0]["installed_size_bytes"] == 1_000_000
    assert manifest["models"][0]["details"] == {
        "format": "gguf",
        "family": "test-family",
        "parameter_size": "1B",
        "quantization_level": "Q4_TEST",
    }
    assert manifest["ollama"]["version"] == "0.test"
    assert manifest["residency_restore"]["status"] == "restored"
    assert fake.resident == set()

    assert len(fake.profile_bodies) == 2
    assert len({body["prompt"] for body in fake.profile_bodies}) == 1
    assert all(body["stream"] is False for body in fake.profile_bodies)
    assert all(body["options"]["num_predict"] == 1 for body in fake.profile_bodies)
    assert all(body["options"]["num_ctx"] == 2048 for body in fake.profile_bodies)
    assert all(body["options"]["temperature"] == 0 for body in fake.profile_bodies)
    assert all(body["options"]["seed"] == 0 for body in fake.profile_bodies)

    called_paths = {path for _, path, _ in fake.requests}
    assert called_paths <= {"/api/tags", "/api/ps", "/api/version", "/api/generate"}
    assert not called_paths & {"/api/pull", "/api/delete", "/api/create", "/api/copy"}


def test_original_residency_is_restored_after_profile_eviction() -> None:
    fake = FakeOllama(
        installed=("tiny:latest", "original:latest"),
        resident=("original:latest",),
        evict_on_profile=True,
    )

    with make_client(fake) as client:
        manifest = OllamaProfiler(client=client).profile(["tiny:latest"], repeats=1)

    restore = manifest["residency_restore"]
    assert restore["status"] == "restored"
    assert restore["initial_models"] == ["original:latest"]
    assert restore["final_models"] == ["original:latest"]
    assert restore["expiry_deadlines_restored"] is False
    assert {tuple(action.values()) for action in restore["actions"]} == {
        ("unload", "tiny:latest"),
        ("preload", "original:latest"),
    }


def test_restoration_waits_for_eventually_consistent_unload() -> None:
    fake = EventuallyConsistentUnloadOllama()

    with make_client(fake) as client:
        manifest = OllamaProfiler(client=client).profile(["tiny:latest"], repeats=1)

    assert manifest["residency_restore"]["status"] == "restored"
    assert manifest["residency_restore"]["final_models"] == []
    assert fake.resident == set()


def test_repeated_profile_waits_for_eventually_consistent_unload() -> None:
    fake = EventuallyConsistentUnloadOllama()

    with make_client(fake) as client:
        manifest = OllamaProfiler(client=client).profile(["tiny:latest"], repeats=2)

    assert len(manifest["models"][0]["observations"]) == 2
    assert manifest["residency_restore"]["status"] == "restored"


def test_missing_model_fails_before_any_mutating_request() -> None:
    fake = FakeOllama()

    with make_client(fake) as client, pytest.raises(ModelNotFoundError):
        OllamaProfiler(client=client).profile(["not-installed:latest"])

    assert [(method, path) for method, path, _ in fake.requests] == [("GET", "/api/tags")]


def test_api_error_does_not_expose_response_body() -> None:
    secret = "SUPER_SECRET_FROM_BACKEND"

    def failing_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": secret})

    with make_client(failing_handler) as client, pytest.raises(OllamaAPIError) as caught:
        OllamaProfiler(client=client).profile(["tiny:latest"])

    assert caught.value.status_code == 503
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)


def test_generation_error_restores_residency_and_scrubs_error_body() -> None:
    fake = FakeOllama(resident=("tiny:latest",))
    secret = "PROMPT_AND_TOKEN_SHOULD_NOT_ESCAPE"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/generate":
            body = json.loads(request.content)
            if "prompt" in body:
                return httpx.Response(500, json={"error": secret, "prompt": body["prompt"]})
        return fake(request)

    with make_client(handler) as client, pytest.raises(OllamaAPIError) as caught:
        OllamaProfiler(client=client).profile(["tiny:latest"], repeats=1)

    assert secret not in str(caught.value)
    assert caught.value.residency_report is not None
    assert caught.value.residency_report["status"] == "restored"
    assert fake.resident == {"tiny:latest"}


def test_manifest_contains_neither_prompt_nor_generated_text() -> None:
    fake = FakeOllama()
    with make_client(fake) as client:
        manifest = OllamaProfiler(client=client).profile(["tiny:latest"], repeats=1)

    serialized = json.dumps(manifest)
    prompt = fake.profile_bodies[0]["prompt"]
    assert prompt not in serialized
    assert "BACKEND_SECRET_MUST_NOT_BE_RECORDED" not in serialized
    assert "cold" not in serialized.lower()
    assert manifest["protocol"]["raw_prompt_recorded"] is False
    assert manifest["protocol"]["generated_text_recorded"] is False
    assert len(manifest["protocol"]["prompt_sha256"]) == 64


@pytest.mark.parametrize(
    ("nanoseconds", "milliseconds"),
    [
        (0, 0.0),
        (1_000_000, 1.0),
        (1_234_567, 1.234567),
        (12_500_000, 12.5),
    ],
)
def test_duration_conversion(nanoseconds: int, milliseconds: float) -> None:
    assert nanoseconds_to_milliseconds(nanoseconds) == milliseconds


@pytest.mark.parametrize("invalid", [-1, True, float("inf"), float("-inf"), float("nan")])
def test_duration_conversion_rejects_invalid_values(invalid: int | float) -> None:
    with pytest.raises(ValueError):
        nanoseconds_to_milliseconds(invalid)
