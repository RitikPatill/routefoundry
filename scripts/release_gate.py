"""Fail-closed checks for release tags and source-distribution contents.

The script deliberately parses only RouteFoundry's small, static version assignments. It
does not import or execute the package being released. Source archives are inspected in
place and are never extracted.
"""

from __future__ import annotations

import argparse
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_SECTION_RE = re.compile(r"^\[([^]]+)]\s*(?:#.*)?$")
_VERSION_ASSIGNMENT_RE = re.compile(r"^version\s*=\s*(['\"])([^'\"]+)\1\s*(?:#.*)?$")
_MODULE_VERSION_RE = re.compile(r"(?m)^__version__\s*=\s*(['\"])([^'\"]+)\1\s*$")
_SAFE_VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.]*$")

REQUIRED_SDIST_PATHS = frozenset(
    {
        "CHANGELOG.md",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "scripts/release_gate.py",
        "scripts/secret_scan.py",
        "space/README.md",
        "space/app.py",
        "space/requirements.txt",
        "src/routefoundry/__init__.py",
        "src/routefoundry/__main__.py",
        "src/routefoundry/cli.py",
        "tests/test_cli.py",
        "tests/test_release_gate.py",
    }
)


class ReleaseGateError(ValueError):
    """A release artifact or tag is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ReleaseVersions:
    project: str
    module: str

    @property
    def expected_tag(self) -> str:
        return f"v{self.project}"


def _version_from_text(text: str, pattern: re.Pattern[str], *, source: str) -> str:
    match = pattern.search(text)
    if match is None:
        raise ReleaseGateError(f"could not find a static version assignment in {source}")
    version = match.group(2)
    if not _SAFE_VERSION_RE.fullmatch(version):
        raise ReleaseGateError(f"unsafe or unsupported version {version!r} in {source}")
    return version


def _project_version_from_text(text: str, *, source: str) -> str:
    in_project = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        section = _SECTION_RE.fullmatch(line)
        if section is not None:
            if in_project:
                break
            in_project = section.group(1) == "project"
            continue
        if not in_project:
            continue
        assignment = _VERSION_ASSIGNMENT_RE.fullmatch(line)
        if assignment is not None:
            version = assignment.group(2)
            if not _SAFE_VERSION_RE.fullmatch(version):
                raise ReleaseGateError(f"unsafe or unsupported version {version!r} in {source}")
            return version
    raise ReleaseGateError(f"could not find a static [project] version in {source}")


def _release_versions(project_text: str, module_text: str, *, source: str) -> ReleaseVersions:
    versions = ReleaseVersions(
        project=_project_version_from_text(
            project_text,
            source=f"{source}/pyproject.toml",
        ),
        module=_version_from_text(
            module_text,
            _MODULE_VERSION_RE,
            source=f"{source}/src/routefoundry/__init__.py",
        ),
    )
    if versions.project != versions.module:
        raise ReleaseGateError(
            "project and module versions disagree: "
            f"{versions.project!r} != {versions.module!r} ({source})"
        )
    return versions


def read_source_versions(project_root: str | Path) -> ReleaseVersions:
    """Read and cross-check the source metadata without importing RouteFoundry."""

    root = Path(project_root)
    try:
        project_text = (root / "pyproject.toml").read_text(encoding="utf-8")
        module_text = (root / "src" / "routefoundry" / "__init__.py").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseGateError(f"could not read release metadata under {root}: {error}") from error
    return _release_versions(project_text, module_text, source=str(root))


def validate_release_tag(tag: str, project_root: str | Path) -> ReleaseVersions:
    """Require the pushed tag to exactly match both package version declarations."""

    versions = read_source_versions(project_root)
    if tag != versions.expected_tag:
        raise ReleaseGateError(
            f"release tag {tag!r} does not match expected tag {versions.expected_tag!r}"
        )
    return versions


def _safe_archive_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ReleaseGateError(f"source distribution contains unsafe path {name!r}")
    return path


def _read_archive_text(archive: tarfile.TarFile, member_name: str) -> str:
    member = archive.getmember(member_name)
    stream = archive.extractfile(member)
    if stream is None:
        raise ReleaseGateError(f"could not read {member_name!r} from source distribution")
    try:
        return stream.read().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseGateError(f"{member_name!r} is not UTF-8") from error


def validate_sdist(path: str | Path, project_root: str | Path) -> ReleaseVersions:
    """Check archive shape, required support files, and embedded version consistency."""

    source_versions = read_source_versions(project_root)
    archive_path = Path(path)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            safe_names = [_safe_archive_name(member.name) for member in members]
            roots = {name.parts[0] for name in safe_names}
            if len(roots) != 1:
                raise ReleaseGateError(
                    "source distribution must contain exactly one top-level directory"
                )
            root = next(iter(roots))
            expected_root = f"routefoundry-{source_versions.project}"
            if root != expected_root:
                raise ReleaseGateError(
                    f"source distribution root {root!r} is not {expected_root!r}"
                )

            relative_names = {
                PurePosixPath(*name.parts[1:]).as_posix()
                for name in safe_names
                if len(name.parts) > 1
            }
            missing = sorted(REQUIRED_SDIST_PATHS - relative_names)
            if missing:
                raise ReleaseGateError(
                    "source distribution is missing required path(s): " + ", ".join(missing)
                )

            archive_versions = _release_versions(
                _read_archive_text(archive, f"{root}/pyproject.toml"),
                _read_archive_text(archive, f"{root}/src/routefoundry/__init__.py"),
                source=str(archive_path),
            )
    except (OSError, tarfile.TarError, KeyError) as error:
        raise ReleaseGateError(
            f"could not inspect source distribution {archive_path}: {error}"
        ) from error

    if archive_versions != source_versions:
        raise ReleaseGateError(
            "source distribution version does not match the checked-out source: "
            f"{archive_versions.project!r} != {source_versions.project!r}"
        )
    return archive_versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate RouteFoundry release invariants")
    parser.add_argument("--tag", help="Git tag being released, for example v0.1.0")
    parser.add_argument("--sdist", type=Path, help="Built .tar.gz source distribution")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Checked-out RouteFoundry root",
    )
    args = parser.parse_args(argv)
    if args.tag is None and args.sdist is None:
        parser.error("at least one of --tag or --sdist is required")

    try:
        versions = read_source_versions(args.project_root)
        if args.tag is not None:
            versions = validate_release_tag(args.tag, args.project_root)
            print(f"Release tag {args.tag} matches RouteFoundry {versions.project}.")
        if args.sdist is not None:
            versions = validate_sdist(args.sdist, args.project_root)
            print(
                f"Source distribution contract passed for RouteFoundry {versions.project}: "
                f"{args.sdist}"
            )
    except ReleaseGateError as error:
        parser.exit(1, f"Release gate failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
