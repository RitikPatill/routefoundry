from pathlib import Path

import scripts.secret_scan as scanner
from scripts.secret_scan import scan_file


def test_secret_scanner_flags_credential_shapes(tmp_path: Path) -> None:
    candidate = tmp_path / "bad.env"
    shaped_value = "gh" + "p_" + "a" * 36
    candidate.write_text(f"token='{shaped_value}'\n", encoding="utf-8")
    assert scan_file(candidate)


def test_secret_scanner_allows_examples_and_documentation(tmp_path: Path) -> None:
    candidate = tmp_path / "safe.md"
    candidate.write_text(
        "Use the API_KEY environment variable.\napi_key='your-key-here'\n",
        encoding="utf-8",
    )
    assert scan_file(candidate) == []


def test_secret_scanner_fails_closed_on_oversized_text(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    candidate = tmp_path / "large.txt"
    candidate.write_text("harmless but deliberately unscanned", encoding="utf-8")
    monkeypatch.setattr(scanner, "MAX_BYTES", 10)

    assert scan_file(candidate) == [(0, "unscanned text file larger than 10 bytes")]
