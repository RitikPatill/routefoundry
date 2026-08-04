# RouteFoundry product plan

Last reviewed: 2026-08-04

This is the public product and evidence plan. Internal launch notes, credentials, private
workloads, and machine-specific paths do not belong in the repository.

## Product promise

RouteFoundry turns a complete matrix of per-prompt model results into:

- an honest comparison against simple routing baselines;
- a small recommended model pool;
- an explainable policy tuned to an empirical development-set quality-loss constraint;
- a held-out report with uncertainty and abstention coverage;
- a local route decision that can account for measured model-switch cost;
- CI checks that reject incompatible workloads before comparing regressions.

It is an auditor and policy compiler, not another inference gateway. It complements
evaluation systems and serving runtimes.

## Why this boundary

Generic model routing is already served by projects such as vLLM Semantic Router,
RouteLLM, UIUC LLMRouter, Aurelio Semantic Router, LiteLLM, and SmarterRouter. Hugging Face
Chat UI also has a deliberately small default/multimodal/agentic policy format. The useful
gap is the workflow between model evaluation results and an inspectable policy:

```text
results -> validate -> split -> compare baselines -> constrain -> compile -> audit -> export
```

RouteFoundry will not claim a universally best router. Current unified evaluation research
shows that simple baselines remain competitive and model-pool recall can dominate algorithm
choice. See [docs/RESEARCH.md](docs/RESEARCH.md).

## v0.1 scope

### Included

- Strict, complete JSONL observation matrices.
- Stable prompt-level train/development/test isolation.
- Explicit trace ordering for residency/switch analysis.
- Always-strongest, always-cheapest, always-fastest, deterministic-random, task-only,
  warm-only, compiled, and non-deployable hindsight-oracle baselines.
- Cost, latency, quality, quality loss, coverage, regret, switch count, and bootstrap
  uncertainty where the input supports them.
- Deterministic policy with unknown-task abstention and warm-model hysteresis.
- Private-by-default static HTML/JSON reports.
- Keyless synthetic demo, clearly labelled non-evidence.
- Hugging Face Space using the same offline core.
- Safe Ollama backend-residency profiler that never pulls or deletes models.
- Read-only maintainer metrics; no automated posting, voting, starring, or messaging.

### Deliberately deferred

- An OpenAI-compatible gateway.
- Cloud provider keys, failover, and cost billing.
- llama.cpp or Kubernetes serving.
- Learned neural classifiers and model training.
- A public model artifact without a trained model.
- Universal benchmark claims.
- Automated community engagement.

## Evidence rules

- Synthetic data is never evidence.
- A hindsight oracle is always labelled non-deployable.
- Development-set constraint satisfaction is not a held-out guarantee.
- Held-out violations and confidence bounds are visible, not hidden.
- Backend-non-resident Ollama measurements are not called true cold starts because the OS
  filesystem cache is uncontrolled.
- Switching metrics require explicit trace and sequence metadata. Without it, RouteFoundry
  reports those metrics as unavailable.
- Input quality scores retain the semantics and limitations of their original grader.
- README numbers must regenerate from committed raw artifacts and methodology.

## Privacy rules

- The core makes no network or model call.
- Reports omit prompts, responses, prompt IDs, classifier features, and credentials by
  default.
- Compiled policies never serialize raw prompt vocabulary. Hashed feature aggregates are
  still not anonymous and must be reviewed before sharing.
- Prompt text is not logged.
- No telemetry is enabled.
- Local-only mode does not initialize a cloud client.
- Secret scans, tracked-file review, and clean-history review block publication.

## Release gates

### Gate 1: deterministic product

- `routefoundry demo` completes offline without an account, key, GPU, or model download.
- Malformed, incomplete, duplicated, non-finite, oversized, or ambiguous input fails closed.
- Audit, compile, route, export, and CI commands have contract tests.
- Default artifacts pass canary privacy-leak regression tests.
- Switching analysis follows explicit trace order.

### Gate 2: reproducible engineering

- Ruff, formatting, strict mypy, tests, configured coverage, package checks, secret scan,
  and clean wheel/sdist installs pass.
- CI covers supported Python versions plus Windows, Linux, and macOS.
- Workflow actions are immutable-pinned.
- Schema versions and workload fingerprints make incompatible CI comparisons fail closed.
- Threat model, security policy, methodology, limitations, and competitor boundary agree
  with the implementation.

### Gate 3: real evidence

- At least three existing local Ollama models have raw, machine-labelled backend-residency
  measurements with the fixed protocol and restoration report.
- Real quality claims wait for properly graded per-model observations.
- At least five external users install without author intervention.
- At least two users integrate a policy into a real workflow.
- Negative findings remain in the report.

### Gate 4: public launch

- PyPI uses trusted publishing; no repository token is stored.
- Hugging Face dataset/Space cards disclose source, license, synthetic content, and limits.
- Community posts are personally reviewed and submitted by a maintainer.
- No coordinated votes, mass messages, generated replies, purchased engagement, or repost
  of an unchanged project.
- Maintainer capacity exists to reproduce reports, answer questions, and ship fixes.

## Post-v0.1 roadmap

Features are added from user evidence, approximately in this order:

1. runtime adapters that preserve the compiled policy semantics;
2. importers for LightEval and established router benchmark formats;
3. drift reports and signed CI artifacts;
4. community hardware-profile dataset with explicit consent and provenance;
5. calibrated classifier experiments with held-out risk/coverage evidence;
6. additional serving-runtime exporters where their policy language is expressive enough.

Stars are not a release gate. Successful external installs, repeat use, independent
integrations, reproducible evidence, useful issues/PRs, and maintainer conversations are
the primary product signals.
