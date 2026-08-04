from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path

import pytest

from scripts.release_gate import (
    REQUIRED_SDIST_PATHS,
    ReleaseGateError,
    validate_release_tag,
    validate_sdist,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PINNED_ACTION_RE = re.compile(
    r"^\s*uses:\s+[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?"
    r"@[0-9a-f]{40}\s+#\s+v[0-9][^\s]*\s*$"
)


def _write_source(
    root: Path, *, project_version: str = "0.1.0", module_version: str = "0.1.0"
) -> None:
    (root / "src" / "routefoundry").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "routefoundry"\nversion = "{project_version}"\n',
        encoding="utf-8",
    )
    (root / "src" / "routefoundry" / "__init__.py").write_text(
        f'__version__ = "{module_version}"\n',
        encoding="utf-8",
    )


def _write_sdist(path: Path, source_root: Path, *, missing: str | None = None) -> None:
    version = "0.1.0"
    archive_root = f"routefoundry-{version}"
    content = {
        relative: f"fixture for {relative}\n".encode()
        for relative in REQUIRED_SDIST_PATHS
        if relative != missing
    }
    content["pyproject.toml"] = (source_root / "pyproject.toml").read_bytes()
    content["src/routefoundry/__init__.py"] = (
        source_root / "src" / "routefoundry" / "__init__.py"
    ).read_bytes()
    with tarfile.open(path, mode="w:gz") as archive:
        for relative, payload in sorted(content.items()):
            member = tarfile.TarInfo(f"{archive_root}/{relative}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_release_tag_must_match_project_and_module_versions(tmp_path: Path) -> None:
    _write_source(tmp_path)
    versions = validate_release_tag("v0.1.0", tmp_path)
    assert versions.project == versions.module == "0.1.0"

    with pytest.raises(ReleaseGateError, match="does not match expected tag"):
        validate_release_tag("v0.1.1", tmp_path)


def test_release_gate_rejects_source_module_version_drift(tmp_path: Path) -> None:
    _write_source(tmp_path, module_version="0.1.1")
    with pytest.raises(ReleaseGateError, match="versions disagree"):
        validate_release_tag("v0.1.0", tmp_path)


def test_release_gate_does_not_read_version_from_another_toml_section(tmp_path: Path) -> None:
    (tmp_path / "src" / "routefoundry").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "routefoundry"\n\n[tool.example]\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "routefoundry" / "__init__.py").write_text(
        '__version__ = "0.1.0"\n', encoding="utf-8"
    )
    with pytest.raises(ReleaseGateError, match=r"static \[project] version"):
        validate_release_tag("v0.1.0", tmp_path)


def test_sdist_contract_checks_support_files_and_embedded_version(tmp_path: Path) -> None:
    _write_source(tmp_path)
    complete = tmp_path / "routefoundry-0.1.0.tar.gz"
    _write_sdist(complete, tmp_path)
    assert validate_sdist(complete, tmp_path).project == "0.1.0"

    incomplete = tmp_path / "routefoundry-incomplete.tar.gz"
    _write_sdist(incomplete, tmp_path, missing="space/requirements.txt")
    with pytest.raises(ReleaseGateError, match=r"space/requirements\.txt"):
        validate_sdist(incomplete, tmp_path)


def test_every_external_workflow_action_is_commit_pinned() -> None:
    uses_lines = [
        line
        for workflow in (REPOSITORY_ROOT / ".github" / "workflows").glob("*.yml")
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("uses:")
    ]
    assert uses_lines
    assert all(_PINNED_ACTION_RE.fullmatch(line) for line in uses_lines), uses_lines


def test_space_dependency_uses_public_versioned_release_without_credentials() -> None:
    requirements = (REPOSITORY_ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
    assert "routefoundry.git@v0.1.0" in requirements
    assert "gradio==6.22.0" in requirements
    assert "github.com:RitikPatill" not in requirements
    assert "https://@" not in requirements
    assert "${" not in requirements
