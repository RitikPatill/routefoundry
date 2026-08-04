"""Small fail-closed scanner for tracked/staged text files.

This is a repository guardrail, not a replacement for GitHub push protection or a
dedicated history scanner. It intentionally looks only for credential-shaped values and
private-key material so documentation can discuss environment-variable names safely.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 2_000_000
SKIP_SUFFIXES = {
    ".7z",
    ".bin",
    ".gif",
    ".gguf",
    ".ico",
    ".jpeg",
    ".jpg",
    ".onnx",
    ".parquet",
    ".pdf",
    ".png",
    ".safetensors",
    ".tar",
    ".zip",
}

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    (
        "GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    ),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("OpenAI-style token", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b")),
    ("Anthropic token", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "credential assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\b"
            r"\s*[:=]\s*['\"](?!example|replace|your-|<)[^'\"\s]{16,}['\"]"
        ),
    ),
)


def tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in completed.stdout.splitlines() if line.strip()]


def scan_file(path: Path) -> list[tuple[int, str]]:
    if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
        return []
    try:
        if path.stat().st_size > MAX_BYTES:
            # A scanner must not silently bless content it did not inspect. Large text
            # artifacts should be explicitly excluded, reduced, or reviewed with a
            # purpose-built history scanner before release.
            return [(0, f"unscanned text file larger than {MAX_BYTES} bytes")]
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    findings: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in PATTERNS:
            if pattern.search(line):
                findings.append((line_number, label))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan repository text for credential-shaped values"
    )
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    paths = args.paths or tracked_files()

    found = False
    for path in paths:
        for line_number, label in scan_file(path):
            found = True
            print(f"{path}:{line_number}: possible {label}", file=sys.stderr)
    if found:
        print("Secret scan failed. Remove the value and rotate it if it was real.", file=sys.stderr)
        return 1
    print(f"Secret scan passed ({len(paths)} files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
