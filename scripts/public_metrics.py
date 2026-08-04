"""Read-only maintainer digest for public RouteFoundry launch signals.

The script has no write credential and cannot post, star, follow, or message anyone. It
turns public API counters into a small JSON snapshot and a prioritized maintenance list.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "routefoundry-maintainer/0.1"})
    with urllib.request.urlopen(request, timeout=15) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object from {url}")
    return value


def build_digest(owner: str, repo: str, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    github = fetch_json(f"https://api.github.com/repos/{owner}/{repo}")
    stars = int(github.get("stargazers_count", 0))
    previous_stars = int((previous or {}).get("github", {}).get("stars", stars))
    open_issues = int(github.get("open_issues_count", 0))

    actions: list[dict[str, str]] = []
    if bool(github.get("archived")):
        actions.append({"priority": "P0", "action": "Repository is archived; resolve status."})
    if open_issues:
        actions.append(
            {
                "priority": "P1",
                "action": (
                    f"Triage {open_issues} open issue/PR item(s) before adding roadmap scope."
                ),
            }
        )
    if stars == previous_stars:
        actions.append(
            {
                "priority": "P2",
                "action": "Interview a user and improve activation; do not manufacture promotion.",
            }
        )
    actions.append(
        {
            "priority": "P2",
            "action": (
                "Reply personally to technical questions and publish only reproducible claims."
            ),
        }
    )

    return {
        "schema_version": "1.0",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "github": {
            "repository": github.get("full_name", f"{owner}/{repo}"),
            "stars": stars,
            "star_delta": stars - previous_stars,
            "forks": int(github.get("forks_count", 0)),
            "open_issues_and_prs": open_issues,
            "subscribers": int(github.get("subscribers_count", 0)),
            "pushed_at": github.get("pushed_at"),
        },
        "actions": actions,
        "guardrail": (
            "Read-only public metrics; no automated posting, voting, starring, or messaging."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a read-only public launch digest")
    parser.add_argument("--repository", default="RitikPatill/routefoundry")
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    owner, separator, repo = args.repository.partition("/")
    if not separator or not owner or not repo:
        parser.error("--repository must be OWNER/NAME")

    previous = None
    if args.previous:
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
    try:
        digest = build_digest(owner, repo, previous)
    except (urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
        print(f"Unable to read public metrics: {exc}")
        return 2
    rendered = json.dumps(digest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
