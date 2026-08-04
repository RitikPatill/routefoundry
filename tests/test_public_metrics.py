from scripts.public_metrics import build_digest


def test_public_digest_is_read_only_and_prioritizes_issues(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.public_metrics.fetch_json",
        lambda _url: {
            "full_name": "RitikPatill/routefoundry",
            "stargazers_count": 12,
            "forks_count": 2,
            "open_issues_count": 3,
            "subscribers_count": 1,
            "pushed_at": "2026-08-04T00:00:00Z",
            "archived": False,
        },
    )
    digest = build_digest("RitikPatill", "routefoundry", {"github": {"stars": 10}})
    assert digest["github"]["star_delta"] == 2
    assert digest["actions"][0]["priority"] == "P1"
    assert "no automated posting" in digest["guardrail"].lower()
