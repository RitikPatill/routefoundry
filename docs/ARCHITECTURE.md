# Architecture and evidence boundary

RouteFoundry separates evidence production, policy compilation, and model serving. It
accepts measurements that already exist; it does not generate responses or decide whether
an upstream grader was scientifically appropriate.

## Data flow

```text
complete prompt × model observations
                |
                v
        strict schema validation
                |
                v
 stable prompt/trace-unit train/dev/test split
          /             |              \
         v              v               v
  hashed-feature   development-only   simple baseline
    classifier     pool/route tuning     definitions
          \             |              /
           \            v             /
            +---- frozen policy ------+
                        |
                        v
              held-out evaluation once
                 /          |          \
                v           v           v
       sanitized report  router.json  CI/export
```

Individual model rows are never split apart from their prompt. With complete, gap-free
trace metadata, an entire trace is the indivisible split unit so related sequence events
cannot cross partitions; otherwise the split unit is one prompt. The deterministic hash
split uses a fixed seed, so row ordering cannot move units between partitions. The training
partition fits a small task classifier. Development observations choose the strongest
fallback, Pareto pool, and constrained route per supported task. The held-out partition is
then used for baseline and compiled-policy evaluation, not route selection.

The workload fingerprint covers prompt identity, task/trace context, schema, and candidate
model set while excluding measured quality, latency, and cost. This makes CI comparisons
fail closed when the evidence population changes. A fingerprint is deterministic and
pseudonymous, not anonymous.

## Components

| Component | Responsibility | Network behavior |
|---|---|---|
| `schema` | Strict JSONL parsing, resource limits, complete-matrix and trace validation | None |
| `split` | Stable prompt or whole-trace train/development/test assignments and digests | None |
| `classify` | Deterministic hashed-feature task classification with abstention | None |
| `optimize` | Baselines, Pareto pruning, constrained route selection, held-out audit | None |
| `policy` | Versioned policy parsing and an explainable route decision | None |
| `report` | Sanitized static HTML and JSON artifacts | None |
| `exporters` | Human review format and constrained Hugging Face Chat UI format | None |
| `ollama` | Optional profiling of already-installed models | Configured Ollama endpoint only |

The CLI composes these components but does not hide their boundaries. Library users can
load and route a policy directly; see [the runtime adapter](../examples/runtime_adapter.py).

## Policy design

A v0.1 policy contains:

- a development-selected strong fallback;
- a small, Pareto-filtered model pool;
- aggregate per-task and global model metrics;
- per-task route rules selected under the development quality-loss budget;
- a deterministic hashed-feature classifier;
- abstention and warm-model hysteresis settings encoded by the policy and route call.

Raw training vocabulary is not serialized. Tokens are mapped into stable SHA-256-derived
feature buckets, and the policy stores aggregate bucket counts. These counts are still
pseudonymous and may leak information through dictionary or correlation attacks. They are
needed to execute text classification, so `router.json` must be treated more cautiously
than the default sanitized report.

`route()` first honors a known explicit task or classifies prompt text. Its returned
`evidence_score` is the classifier's softmax-normalized score; it is **not calibrated** as a
probability of task correctness or answer quality. Unknown, unsupported, low-score, or
low-margin inputs abstain to the fallback.

When the caller supplies a currently resident `warm_model`, RouteFoundry may retain it if
its development aggregate remains inside the quality floor and the measured load plus
caller-supplied switching cost outweighs expected latency gain. The caller, not
RouteFoundry, owns the truth of current residency and the external model invocation.

## Evidence boundary

`quality` is a user-supplied empirical score in `[0, 1]`. RouteFoundry assumes scores use a
meaningfully comparable grader and normalization across every row. It validates type,
range, and matrix consistency; it cannot validate grader calibration, bias, leakage,
licensing, or relevance to users.

The configured budget constrains mean quality loss relative to the strongest fallback
chosen on development data. RouteFoundry reports two distinct facts:

1. **Development constraint:** the frozen compiled policy met the empirical budget on data
   used for route selection. This is a tuning result, not a generalization claim.
2. **Held-out check:** the observed mean quality loss and its bootstrap interval upper bound
   are both compared with the same budget on prompts not used for route selection.

The held-out check may fail even when development passed, and that failure remains visible.
A passing held-out check is not a guarantee for future traffic. Untraced audits use a
deterministic percentile bootstrap over prompts. Complete-trace audits use a deterministic
whole-trace cluster bootstrap and a prompt-weighted mean, preserving within-trace
dependence. Both cover sampling variation only; they exclude grader uncertainty,
distribution shift, runtime failures, and model updates. Trace-level bootstrap requires at
least three held-out trace clusters. An audit minimum of 30 prompts prevents tiny examples
from masquerading as an audit, but neither minimum guarantees useful statistical power.

The hindsight oracle inspects the answer scores for each held-out prompt before choosing a
model. It is non-deployable and serves only as an upper-bound diagnostic. It must not be
reported as the compiled router's performance.

## Trace and residency semantics

Model switching is a sequence property. RouteFoundry never reconstructs sequence from
prompt IDs. Residency metrics and trace-level splitting are available only when every
prompt supplies an explicit `trace_id` and `sequence_index` pair and every trace is
gap-free. Within each trace, sequence indices must be unique and consecutive; the first
index may be any non-negative integer.

With complete trace metadata, whole traces stay in one train/development/test partition and
evaluation replays each held-out trace in sequence order. A selected model's supplied
`load_ms` is charged when it becomes resident, including the first model in a trace.
`switch_count` counts changes between successive selected models; the initial selection is
not a switch. Without complete trace metadata, splitting falls back to prompt units, load
penalties and switch counts are unavailable, and no residency penalty is applied.

See [the observation schema](DATA_SCHEMA.md) and the optional
[Ollama methodology](OLLAMA_METHODOLOGY.md).

## Artifact and export boundaries

`audit` writes:

- `report.html`: self-contained, script-free static report;
- `summary.json`: sanitized machine-readable audit;
- `router.json`: executable policy, including pseudonymous feature buckets;
- `policy.txt`: sanitized human review format.

Default reports remove raw prompts, responses, prompt identifiers, classifier feature
counts, and credential-like fields recursively. This is a defense against accidental
disclosure, not a proof of anonymity. Model names, task labels, aggregate metrics,
fingerprints, and split digests can still be sensitive.

### Export boundaries

Hugging Face Chat UI's documented router format has only `default`, `multimodal`, and
`agentic` capability routes. A RouteFoundry semantic task such as `summarization` cannot be
represented truthfully in that format. A compiled v0.1 policy therefore exports only the
strong fallback as Chat UI's `default`, with other pooled models as fallbacks. It does not
promote task labels into capability declarations. Multimodal and agentic exports require a
future verified capability metadata contract.

## Determinism and compatibility

Given the same schema-valid rows, seed, objective, budget, and package version, validation,
splitting, compilation, routing, reporting, and export are deterministic. Bootstrap
resampling is seeded. JSON artifacts use explicit schema versions.

The `ci` command compares only like-for-like audits. It rejects a baseline when audit schema,
workload fingerprint, model set, seed, objective, or budget differs, then applies the
held-out, latency, and cost gates. Comparing different workloads requires a new baseline
and an explicit human interpretation, not bypassing the compatibility check.

## Deliberate non-goals

RouteFoundry v0.1 is not an inference proxy, response grader, model trainer, provider cost
ledger, retry/failover layer, or universal routing benchmark. The reasons for this boundary
and related projects are documented in [RESEARCH.md](RESEARCH.md).
