# Ethical maintainer loop

`scripts/public_metrics.py` is deliberately read-only. It fetches unauthenticated public
GitHub repository metadata, writes an optional local JSON snapshot, and prioritizes
maintenance work. It has no credential and no ability to post, vote, star, follow, direct
message, or solicit engagement.

After the repository is public:

```bash
uv run python scripts/public_metrics.py \
  --repository RitikPatill/routefoundry \
  --output out/metrics/today.json

uv run python scripts/public_metrics.py \
  --repository RitikPatill/routefoundry \
  --previous out/metrics/yesterday.json \
  --output out/metrics/today.json
```

The useful weekly loop is:

1. fix broken CI, packaging, documentation links, or installation paths;
2. triage external issues/PRs and reproduce failures before proposing scope;
3. inspect where unaided users stop between install, first report, and integration;
4. interview users with consent when activation or repeat use stalls;
5. publish reproducible evidence and complete release notes;
6. participate personally in relevant communities under their current rules.

Public counters are weak proxies. Stars can reveal that someone noticed a repository; they
do not prove installation, repeat use, valid evidence, or production value. Track
independent installs, integrations, repeat audits, useful issues/PRs, and resolved user
friction as primary product signals.

## Guardrails

- Never add a credential or write-capable API to the metrics script.
- Never automate comments, replies, stars, follows, messages, cross-posting, or vote asks.
- Never turn contributor/user metadata into a prospecting list.
- Do not alert on an unchanged star count as though it were an incident.
- Keep snapshots out of the package and review them for unexpected public metadata before
  sharing.
- A maintainer personally reviews every external communication.

Automated persuasion would undermine community trust and the technical signal of the
project. See [the launch checklist](LAUNCH.md) and [beta checklist](BETA_CHECKLIST.md).
