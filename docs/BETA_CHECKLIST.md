# External beta interview checklist

Recruit people who already compare at least two models and have a real routing decision.
Do not recruit only friends who are likely to praise the idea. Ask permission before
recording; store no private prompts, responses, company names, or credentials in public
issues.

## Observe without coaching

1. Can they state RouteFoundry's input and output after reading the README opening?
2. Can they install and run the synthetic demo without author intervention?
3. Can they convert a small evaluator export into the complete prompt × model JSONL matrix?
4. Do schema failures explain what they need to repair?
5. Can they distinguish the empirical development constraint from the held-out check?
6. Do they understand that `evidence_score` is an uncalibrated classifier score?
7. Can they identify the fallback, one baseline, one abstention, and one route reason?
8. If they care about switching, can they supply real trace order and `load_ms` evidence?
9. Can they call `load_policy()`/`route()` before their existing inference runtime?
10. Which artifact or privacy concern prevents them from sharing or deploying the policy?
11. Would they repeat this audit after changing a model, grader, price, or runtime—and why?

## Capture evidence, not sentiment

For each session, record only consented, generalized notes:

- environment and installation path;
- time to first report and first integration decision;
- exact step where intervention was needed;
- misunderstood claim or field;
- observed error message;
- decision the report did or did not change;
- privacy/security objection;
- next action and whether it was completed independently.

Avoid leading satisfaction scores as the primary signal. A useful beta result is a
reproducible failure, successful unaided workflow, repeat run, or real integration.

## Exit criteria

The public launch gate asks for five unaided installs and two real integrations. These are
small evidence thresholds, not proof of production readiness or future popularity. Do not
count a synthetic demo run as an integration, a verbal promise as repeat use, or coordinated
stars as adoption.

Before broad launch, resolve or explicitly document every repeated P0/P1 activation or
privacy issue. Feed findings into tests/docs and review the
[release checklist](LAUNCH.md) again.
