"""Safe, reproducible profiling of models already installed in Ollama.

The profiler deliberately has no model-management features: it only reads ``tags``,
``ps``, and ``version``, and sends ``generate`` requests to models that were returned by
``tags``.  In particular, it never pulls or deletes a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import urlsplit

import httpx
import psutil  # type: ignore[import-untyped]

SCHEMA_VERSION: Final = "1.0"
ARTIFACT_TYPE: Final = "routefoundry.ollama-profile"
DEFAULT_BASE_URL: Final = "http://127.0.0.1:11434"
DEFAULT_REPEATS: Final = 3
MAX_REPEATS: Final = 20

# This is intentionally constant and is never copied into a result or exception.  A hash
# allows artifacts to be compared without making prompt logging a precedent.
_FIXED_PROMPT: Final = "Reply with OK."
_FIXED_PROMPT_SHA256: Final = hashlib.sha256(_FIXED_PROMPT.encode("utf-8")).hexdigest()
_PROFILE_KEEP_ALIVE: Final = "5m"
_RESTORE_KEEP_ALIVE: Final = "5m"
_PROFILE_CONTEXT_LENGTH: Final = 2048
_RESIDENCY_SETTLE_SECONDS: Final = 10.0
_RESIDENCY_POLL_SECONDS: Final = 0.1

_DURATION_FIELDS: Final = (
    "load_duration",
    "total_duration",
    "prompt_eval_duration",
    "eval_duration",
)
_COUNT_FIELDS: Final = ("prompt_eval_count", "eval_count")


class OllamaProfileError(RuntimeError):
    """Base error whose text never contains an Ollama response body."""

    residency_report: dict[str, Any] | None

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.residency_report = None


class OllamaAPIError(OllamaProfileError):
    """An Ollama request failed without exposing response or request contents."""

    operation: str
    status_code: int | None
    error_type: str

    def __init__(
        self,
        operation: str,
        *,
        status_code: int | None = None,
        error_type: str = "OllamaAPIError",
    ) -> None:
        self.operation = operation
        self.status_code = status_code
        self.error_type = error_type
        suffix = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"Ollama API operation {operation!r} failed{suffix}")


class ModelNotFoundError(OllamaProfileError):
    """A requested model was not present in ``/api/tags``."""


class InvalidOllamaResponseError(OllamaProfileError):
    """Ollama returned a success response without the required shape or metrics."""


class ResidencyVerificationError(OllamaProfileError):
    """A model remained resident after an explicit unload request."""


def nanoseconds_to_milliseconds(value: int | float) -> float:
    """Convert a non-negative backend nanosecond counter to milliseconds."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("duration must be a non-negative number")
    return round(float(value) / 1_000_000.0, 6)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_error(error: OllamaProfileError) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(error).__name__}
    if isinstance(error, OllamaAPIError):
        result["operation"] = error.operation
        result["status_code"] = error.status_code
        result["transport_error_type"] = error.error_type
    return result


def _identity_values(item: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("name", "model"):
        value = item.get(key)
        if isinstance(value, str) and value and value not in values:
            values.append(value)
    return tuple(values)


def _primary_identity(item: Mapping[str, Any]) -> str | None:
    identities = _identity_values(item)
    return identities[0] if identities else None


class OllamaProfiler:
    """Profile already-installed Ollama models without managing model files.

    A caller-supplied ``httpx.Client`` is useful for tests and remains owned by the caller.
    The default client ignores proxy environment variables so local requests cannot be
    redirected through a configured HTTP proxy.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")

        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            trust_env=False,
        )

    def __enter__(self) -> OllamaProfiler:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close an internally-created HTTP client."""

        if self._owns_client:
            self._client.close()

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise OllamaAPIError(
                operation,
                status_code=error.response.status_code,
                error_type=type(error).__name__,
            ) from None
        except httpx.HTTPError as error:
            # Do not include ``str(error)``: it may contain a credential-bearing URL.
            raise OllamaAPIError(operation, error_type=type(error).__name__) from None

        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise InvalidOllamaResponseError(
                f"Ollama operation {operation!r} returned invalid JSON"
            ) from None
        if not isinstance(payload, dict):
            raise InvalidOllamaResponseError(
                f"Ollama operation {operation!r} returned a non-object JSON value"
            )
        return cast(dict[str, Any], payload)

    def list_tags(self) -> list[dict[str, Any]]:
        """Return installed model tag records from Ollama."""

        payload = self._request_json("GET", "/api/tags", operation="list_tags")
        models = payload.get("models")
        if not isinstance(models, list):
            raise InvalidOllamaResponseError("Ollama tag response has no model list")
        if not all(isinstance(item, dict) for item in models):
            raise InvalidOllamaResponseError("Ollama tag response contains an invalid model")
        return [cast(dict[str, Any], item) for item in models]

    def list_resident_models(self) -> list[dict[str, Any]]:
        """Return model process records from Ollama."""

        payload = self._request_json("GET", "/api/ps", operation="list_resident_models")
        models = payload.get("models")
        if not isinstance(models, list):
            raise InvalidOllamaResponseError("Ollama process response has no model list")
        if not all(isinstance(item, dict) for item in models):
            raise InvalidOllamaResponseError("Ollama process response contains an invalid model")
        return [cast(dict[str, Any], item) for item in models]

    def _version(self) -> tuple[str | None, dict[str, Any] | None]:
        try:
            payload = self._request_json("GET", "/api/version", operation="get_version")
        except OllamaProfileError as error:
            return None, _safe_error(error)
        version = payload.get("version")
        if not isinstance(version, str) or not version:
            return None, {"type": "InvalidOllamaResponseError"}
        return version, None

    @staticmethod
    def _resolve_models(
        requested_models: Sequence[str], tags: Sequence[Mapping[str, Any]]
    ) -> list[tuple[str, str, Mapping[str, Any]]]:
        if not requested_models:
            raise ValueError("at least one model is required")

        lookup: dict[str, Mapping[str, Any]] = {}
        for tag in tags:
            for identity in _identity_values(tag):
                lookup.setdefault(identity, tag)

        resolved: list[tuple[str, str, Mapping[str, Any]]] = []
        seen: set[str] = set()
        for requested in requested_models:
            if not isinstance(requested, str) or not requested.strip():
                raise ValueError("model names must be non-empty strings")
            exact_name = requested.strip()
            resolved_tag = lookup.get(exact_name)
            if resolved_tag is None:
                raise ModelNotFoundError(
                    f"requested model {exact_name!r} is not installed according to Ollama tags"
                )
            canonical = _primary_identity(resolved_tag)
            if canonical is None:
                raise InvalidOllamaResponseError("an installed model has no name")
            if canonical not in seen:
                resolved.append((exact_name, canonical, resolved_tag))
                seen.add(canonical)
        return resolved

    @staticmethod
    def _resident_names(records: Sequence[Mapping[str, Any]]) -> set[str]:
        return {
            identity for record in records if (identity := _primary_identity(record)) is not None
        }

    def _unload(self, model: str, *, operation: str = "unload_model") -> None:
        self._request_json(
            "POST",
            "/api/generate",
            operation=operation,
            body={"model": model, "stream": False, "keep_alive": 0},
        )

    def _wait_until_absent(self, model: str) -> set[str]:
        """Wait for Ollama's eventually consistent process list to show an eviction."""

        deadline = time.monotonic() + _RESIDENCY_SETTLE_SECONDS
        while True:
            resident = self._resident_names(self.list_resident_models())
            if model not in resident:
                return resident
            if time.monotonic() >= deadline:
                return resident
            time.sleep(_RESIDENCY_POLL_SECONDS)

    def _preload(self, model: str) -> None:
        # Ollama treats a generate request without a prompt as a preload.  No generated
        # content is requested or retained.
        self._request_json(
            "POST",
            "/api/generate",
            operation="restore_resident_model",
            body={"model": model, "stream": False, "keep_alive": _RESTORE_KEEP_ALIVE},
        )

    def _profile_once(self, model: str) -> dict[str, Any]:
        payload = self._request_json(
            "POST",
            "/api/generate",
            operation="profile_generate",
            body={
                "model": model,
                "prompt": _FIXED_PROMPT,
                "stream": False,
                "keep_alive": _PROFILE_KEEP_ALIVE,
                "options": {
                    "num_ctx": _PROFILE_CONTEXT_LENGTH,
                    "num_predict": 1,
                    "temperature": 0,
                    "seed": 0,
                },
            },
        )

        metrics: dict[str, Any] = {"backend_reported": True}
        for field in _DURATION_FIELDS:
            value = payload.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise InvalidOllamaResponseError(
                    f"Ollama profile response has invalid {field!r} metric"
                )
            metrics[f"{field}_ns"] = value
            metrics[f"{field}_ms"] = nanoseconds_to_milliseconds(value)
        for field in _COUNT_FIELDS:
            value = payload.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidOllamaResponseError(
                    f"Ollama profile response has invalid {field!r} metric"
                )
            metrics[field] = value
        return metrics

    def _restore_residency(
        self,
        initial_names: set[str],
        profiled_names: set[str],
    ) -> dict[str, Any]:
        actions: list[dict[str, str]] = []
        errors: list[dict[str, Any]] = []

        try:
            current_names = self._resident_names(self.list_resident_models())
        except OllamaProfileError as error:
            return {
                "status": "unknown",
                "initial_models": sorted(initial_names),
                "final_models": None,
                "missing_models": None,
                "additional_models": None,
                "actions": actions,
                "errors": [_safe_error(error)],
                "expiry_deadlines_restored": False,
            }

        # Only remove models this invocation profiled.  An unrelated model that appeared
        # concurrently belongs to another process and is reported, never stopped.
        for model in sorted((current_names - initial_names) & profiled_names):
            try:
                self._unload(model, operation="restore_unload_profile_model")
                actions.append({"action": "unload", "model": model})
            except OllamaProfileError as error:
                errors.append({"model": model, **_safe_error(error)})

        try:
            current_names = self._resident_names(self.list_resident_models())
        except OllamaProfileError as error:
            errors.append(_safe_error(error))
            current_names = set()

        for model in sorted(initial_names - current_names):
            try:
                self._preload(model)
                actions.append({"action": "preload", "model": model})
            except OllamaProfileError as error:
                errors.append({"model": model, **_safe_error(error)})

        # Ollama may acknowledge ``keep_alive: 0`` before ``/api/ps`` reflects the
        # eviction.  Poll only for state changed by this invocation: all originally
        # resident models must be present, and profiled models that were not originally
        # resident must be gone.  Unrelated models are never stopped.
        deadline = time.monotonic() + _RESIDENCY_SETTLE_SECONDS
        while True:
            try:
                settled_names = self._resident_names(self.list_resident_models())
            except OllamaProfileError as error:
                errors.append(_safe_error(error))
                break
            originally_restored = initial_names <= settled_names
            profiled_extras_gone = not ((settled_names - initial_names) & profiled_names)
            if originally_restored and profiled_extras_gone:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(_RESIDENCY_POLL_SECONDS)

        try:
            final_names = self._resident_names(self.list_resident_models())
        except OllamaProfileError as error:
            errors.append(_safe_error(error))
            return {
                "status": "unknown",
                "initial_models": sorted(initial_names),
                "final_models": None,
                "missing_models": None,
                "additional_models": None,
                "actions": actions,
                "errors": errors,
                "expiry_deadlines_restored": False,
            }

        missing = sorted(initial_names - final_names)
        additional = sorted(final_names - initial_names)
        status = "restored" if not missing and not additional and not errors else "changed"
        return {
            "status": status,
            "initial_models": sorted(initial_names),
            "final_models": sorted(final_names),
            "missing_models": missing,
            "additional_models": additional,
            "actions": actions,
            "errors": errors,
            # Remaining lifetime is backend state that cannot be restored exactly via the
            # public API.  This stays explicit even when the resident set matches.
            "expiry_deadlines_restored": False,
        }

    def profile(
        self,
        models: Sequence[str],
        *,
        repeats: int = DEFAULT_REPEATS,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Profile ``models``, optionally write an atomic JSON manifest, and return it.

        Every requested name must exactly match a ``name`` or ``model`` value returned by
        ``/api/tags``.  The method snapshots residency before making any state-changing
        request.  Original residency is restored best-effort on both success and failure.
        """

        if isinstance(repeats, bool) or not isinstance(repeats, int):
            raise ValueError("repeats must be an integer")
        if repeats < 1 or repeats > MAX_REPEATS:
            raise ValueError(f"repeats must be between 1 and {MAX_REPEATS}")

        tags = self.list_tags()
        resolved = self._resolve_models(models, tags)
        initial_names = self._resident_names(self.list_resident_models())
        version, version_error = self._version()
        started_at = _utc_now()
        profiled_names = {canonical for _, canonical, _ in resolved}
        model_results: list[dict[str, Any]] = []

        try:
            for requested, canonical, tag in resolved:
                observations: list[dict[str, Any]] = []
                for repeat_index in range(1, repeats + 1):
                    self._unload(canonical)
                    after_unload = self._wait_until_absent(canonical)
                    if canonical in after_unload:
                        raise ResidencyVerificationError(
                            f"Ollama still reports model {canonical!r} as resident after unload"
                        )

                    observation_started_at = _utc_now()
                    metrics = self._profile_once(canonical)
                    observations.append(
                        {
                            "repeat": repeat_index,
                            "started_at": observation_started_at,
                            "completed_at": _utc_now(),
                            **metrics,
                        }
                    )

                digest = tag.get("digest")
                installed_size = tag.get("size")
                details = tag.get("details")
                public_details = (
                    {
                        key: details[key]
                        for key in (
                            "format",
                            "family",
                            "families",
                            "parameter_size",
                            "quantization_level",
                        )
                        if key in details and isinstance(details[key], str | list)
                    }
                    if isinstance(details, Mapping)
                    else {}
                )
                model_results.append(
                    {
                        "requested_name": requested,
                        "model": canonical,
                        "digest": digest if isinstance(digest, str) else None,
                        "installed_size_bytes": (
                            installed_size
                            if isinstance(installed_size, int)
                            and not isinstance(installed_size, bool)
                            else None
                        ),
                        "details": public_details,
                        "observations": observations,
                    }
                )
        except BaseException as error:
            residency = self._restore_residency(initial_names, profiled_names)
            if isinstance(error, OllamaProfileError):
                error.residency_report = residency
            raise

        residency = self._restore_residency(initial_names, profiled_names)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "measurement": {
                "residency_condition": "backend-non-resident",
                "os_cache_condition": "OS-cache-uncontrolled",
                "repeats_per_model": repeats,
            },
            "protocol": {
                "prompt_sha256": _FIXED_PROMPT_SHA256,
                "prompt_utf8_bytes": len(_FIXED_PROMPT.encode("utf-8")),
                "raw_prompt_recorded": False,
                "generated_text_recorded": False,
                "stream": False,
                "num_predict": 1,
                "context_length": _PROFILE_CONTEXT_LENGTH,
                "temperature": 0,
                "seed": 0,
                "unload_keep_alive": 0,
                "profile_keep_alive": _PROFILE_KEEP_ALIVE,
            },
            "ollama": {
                "version": version,
                "version_query_error": version_error,
            },
            "host": _host_metadata(),
            "models": model_results,
            "residency_restore": residency,
        }

        if output_path is not None:
            _write_manifest_atomic(Path(output_path), manifest)
        return manifest


def _host_metadata() -> dict[str, Any]:
    virtual_memory = psutil.virtual_memory()
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "processor": platform.processor() or None,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "memory": {"total_bytes": virtual_memory.total},
    }


def _write_manifest_atomic(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(manifest, temporary, indent=2, sort_keys=True, ensure_ascii=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            with suppress(OSError):
                Path(temporary_name).unlink(missing_ok=True)


def profile_ollama_models(
    models: Sequence[str],
    *,
    output_path: str | Path | None = None,
    repeats: int = DEFAULT_REPEATS,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`OllamaProfiler`."""

    with OllamaProfiler(base_url=base_url, timeout=timeout, client=client) as profiler:
        return profiler.profile(models, repeats=repeats, output_path=output_path)


# Short alias for integrations that prefer an action-oriented function name.
profile_ollama = profile_ollama_models


__all__ = [
    "ARTIFACT_TYPE",
    "DEFAULT_BASE_URL",
    "DEFAULT_REPEATS",
    "SCHEMA_VERSION",
    "InvalidOllamaResponseError",
    "ModelNotFoundError",
    "OllamaAPIError",
    "OllamaProfileError",
    "OllamaProfiler",
    "ResidencyVerificationError",
    "nanoseconds_to_milliseconds",
    "profile_ollama",
    "profile_ollama_models",
]
