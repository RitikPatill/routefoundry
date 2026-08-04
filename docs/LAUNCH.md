# Release and authentic launch checklist

A public launch is a reproducibility event, not a promise of attention. RouteFoundry should
be released only when an unfamiliar user can understand the evidence boundary, install the
package, run the synthetic demo, and inspect a real raw measurement without private help.

## 1. Repository release gate

- [ ] Repeat project-name, package-name, and trademark checks.
- [ ] Review `git status`, every staged file, every filename, and the complete Git history.
- [ ] Confirm unrelated project data, credentials, private prompts/responses, local paths,
      screenshots with personal data, and editor/AI-agent state are absent.
- [ ] Run formatting, lint, strict typing, tests with configured coverage, package builds,
      clean wheel/sdist smoke installs, metadata checks, and the secret scanner.
- [ ] Confirm CI uses immutable action revisions and passes on supported Python/platforms.
- [ ] Regenerate the keyless demo and inspect every artifact, including `router.json`.
- [ ] Confirm synthetic data says synthetic/non-evidence above the fold and in artifacts.
- [ ] Confirm real measurements link to raw manifests, exact commands, methodology, and
      limitations—never only a favorable screenshot.
- [ ] Verify [README](../README.md), [architecture](ARCHITECTURE.md),
      [schema](DATA_SCHEMA.md), and [threat model](THREAT_MODEL.md) agree with code.

## 2. Package and Hugging Face gate

- [ ] Publish PyPI through trusted publishing; store no repository API token.
- [ ] Install the released wheel into a clean environment and run `routefoundry demo`.
- [ ] Publish the Space from [the reviewed scaffold](../space/) and verify it uses the
      released package, requires no secret, and makes no outbound model/API call.
- [ ] Ensure the Space card discloses that all displayed observations are synthetic,
      illustrative, and non-evidence.
- [ ] If publishing the profile dataset, complete the
      [dataset-card scaffold](../huggingface/ollama-residency-profiles/README.md), copy only
      reviewed manifests, record the source revision, and test links in the dataset repo.
- [ ] State that compiled Hugging Face Chat UI export emits only the safe `default` route
      until verified model capability metadata exists.

## 3. External beta gate

- [ ] Five people who already compare at least two models install without author
      intervention.
- [ ] Two people integrate a compiled policy into a real evaluation or runtime workflow.
- [ ] Capture only consented, non-sensitive friction using the
      [beta interview checklist](BETA_CHECKLIST.md).
- [ ] Convert repeated friction into documentation, tests, or a scoped issue before adding
      broad roadmap features.
- [ ] Preserve negative findings: held-out failures, abstention, restoration errors, and
      timing outliers remain visible.
- [ ] Re-run security and privacy review after beta-driven changes.

These counts are evidence goals, not claims that beta success guarantees production
fitness, popularity, stars, or employment.

## 4. Public communication gate

- [ ] Prepare one short, reproducible walkthrough: problem, command, artifact, limitation.
- [ ] Link directly to source, raw evidence, and methodology.
- [ ] Check each community's current self-promotion, disclosure, and repost rules at posting
      time.
- [ ] Write and submit every post/comment personally; generated drafts receive human review.
- [ ] Keep maintainer capacity available to reproduce bug reports and ship fixes.
- [ ] Never coordinate votes, buy engagement, ask for stars as payment, mass-message users,
      automate replies, or repost an unchanged project under a new title.

Lead with a real user problem and a falsifiable artifact, not a star target. The
[read-only maintainer loop](MAINTAINER_LOOP.md) can prioritize maintenance after launch but
cannot post, vote, star, follow, or message anyone.

## If response is weak

Do not manufacture activity. Check in this order:

1. Can a clean machine install and run the demo?
2. Does the README explain what RouteFoundry is *not* before asking for data?
3. Can a user convert their evaluator output to the complete matrix?
4. Does the report answer a decision they actually have?
5. Can they integrate the policy without changing inference infrastructure?
6. Which privacy or evidence concern blocks repeat use?

Interview users, fix the largest observed activation problem, and announce a materially
different release only where community rules allow it. A useful small project with honest
limitations is a stronger signal than manipulated traction.
