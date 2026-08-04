# Security policy

## Supported versions

Until the first stable release, only the newest tagged version and `main` receive
security fixes.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not open a
public issue containing an exploit, credential, private prompt, model response, hostname,
or filesystem path. If private reporting is temporarily unavailable, open a minimal issue
asking the maintainer to enable a private channel without disclosing the vulnerability.

You should receive acknowledgement within five working days. A fix timeline depends on
severity and whether downstream users need coordinated disclosure.

## Security boundary

RouteFoundry v0.1 reads evaluation results, creates aggregate reports, and compiles local
policies. It does not need an API key. It does not send prompts or results to a cloud
service, does not enable telemetry, and omits raw prompts from reports by default.

An executable `router.json` contains model and task names, aggregate measurements, and
deterministic hashed feature counts. Reports omit those feature tables by default, but
policies, workload fingerprints, and split digests are pseudonymous rather than anonymous.
Review generated artifacts before sharing them; field-name filtering is defense in depth,
not a general data-loss-prevention system.

The optional Ollama profiler connects only to the configured Ollama HTTP endpoint and
never pulls or deletes a model. A remote Ollama endpoint has its own trust boundary; use
TLS and authentication outside localhost.

RouteFoundry's secret scanner is defense in depth. It does not replace GitHub push
protection, credential rotation, or a complete history scan.
