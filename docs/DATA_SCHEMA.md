# Observation schema

RouteFoundry accepts UTF-8 JSON Lines. Each non-blank line is one observation for one
prompt/model pair. The current schema identifier is `routefoundry.observation.v1`.

```json
{
  "prompt_id": "math-001",
  "prompt": "What is 17 × 23?",
  "task": "math",
  "trace_id": "worksheet-04",
  "sequence_index": 7,
  "model": "small-local",
  "quality": 1.0,
  "latency_ms": 420.0,
  "cost_usd": 0.0,
  "load_ms": 95.0,
  "metadata": {"grader": "exact-match-v1"}
}
```

## Fields

| Field | Required | Contract |
|---|---:|---|
| `prompt_id` | yes | Non-empty string with no leading/trailing whitespace; stable identity for one prompt |
| `model` | yes | Non-empty model identifier |
| `quality` | yes | Finite JSON number in `[0, 1]`; semantics come from your grader |
| `latency_ms` | yes | Finite, non-negative JSON number |
| `cost_usd` | yes | Finite, non-negative JSON number |
| `prompt` | no | Prompt text used by the compiled classifier |
| `task` | no | Explicit task/segment label |
| `trace_id` | paired | Non-empty sequence/group identifier; requires `sequence_index` |
| `sequence_index` | paired | Non-negative integer unique within a trace; requires `trace_id` |
| `load_ms` | no | Finite, non-negative model load/residency penalty; defaults to `0` |
| `metadata` | no | JSON object containing finite JSON-compatible values; defaults to `{}` |

Unknown fields are rejected. Duplicate JSON object keys, non-standard `NaN`/`Infinity`
values, blank lines, invalid UTF-8, booleans in numeric fields, and duplicate
prompt/model pairs are also rejected.

`prompt` and `task` may both be absent for validation-only data. Policy compilation and
audit require at least one of them for every prompt. An explicit task can route without
text classification if the task is known to the compiled policy; prompt text supports
classification when an explicit task is not supplied at runtime.

## Complete matrix invariant

Every prompt must have exactly one row for every candidate model in the dataset. RouteFoundry
fails on a missing pair instead of interpreting missing evidence as a low score.

For example, 100 prompts and 3 models require exactly 300 unique rows. Fields describing
the prompt—`prompt`, `task`, `trace_id`, `sequence_index`, and `metadata`—must be identical
across all model rows for that `prompt_id`.

Keep one normalized grading scale across the entire matrix. Do not mix exact-match,
preference, and judge scores as though their numbers were interchangeable. RouteFoundry
checks the range, not the scientific validity or calibration of your grader.

## Traces and switching

`trace_id` groups prompts that belong to one ordered runtime sequence. `sequence_index`
orders prompts inside that trace. Indices must be unique within a trace and need not begin
at zero. To count as a complete trace, they must be consecutive from that first value with
no gaps.

Both fields are optional as a pair. However, trace-aware splitting, residency analysis, and
cluster uncertainty are all-or-nothing: **every prompt** must have trace metadata and every
trace must be gap-free. Partial trace coverage or a sequence gap validates as metadata but
causes splitting to fall back to prompt units and residency metrics to be reported as
unavailable. RouteFoundry does not infer order from lexical prompt IDs or source-file row
order.

Within a complete trace, evaluation charges the selected row's `load_ms` whenever a model
becomes resident, including the first selection. A switch is counted only when successive
selections change model.

## Stable split and fingerprints

Without complete traces, splits happen at the prompt level using a deterministic hash of
`prompt_id` and seed. With complete, gap-free traces, the entire `trace_id` group is an
indivisible split unit so related events cannot leak across train/development/test. All
model rows for a prompt always remain together. The serialized split contains counts and
assignment digests, not raw prompt IDs; trace-level splits also include trace counts and
trace-assignment digests.

The audit workload fingerprint covers prompt identity, task/trace metadata, schema, and
candidate model set while excluding measurements. Both split digests and workload
fingerprints are deterministic and **pseudonymous, not anonymous**. Low-entropy prompt IDs
may be testable by an attacker with a candidate list.

## Resource limits

The parser fails before accepting data beyond these v0.1 limits:

- file or in-memory JSONL size: 64 MiB;
- one JSONL line: 1 MiB encoded as UTF-8;
- rows: 1,000,000.

File loading streams and validates lines subject to those limits. The validated complete
matrix is then held in memory for splitting and audit; plan capacity for the expanded
prompt × model row count.

## Minimal commands

```bash
routefoundry validate observations.jsonl
routefoundry validate observations.jsonl --require-signal
routefoundry audit observations.jsonl --output out/audit
```

`audit` requires at least 30 prompts and minimum deterministic split sizes of 10 training,
5 development, and 5 test prompts. Because a hash split can be imbalanced, a dataset with
exactly 30 prompts can still fail a partition minimum. A trace-level audit additionally
needs at least three held-out traces for cluster bootstrap, plus at least one training and
one development trace—at least five total before prompt-count minima are considered.
`compile` has a lower mechanical minimum of 3 prompts, but it still needs non-empty split
units and such a tiny policy should be treated only as a smoke test.

## Privacy guidance

Prompt text is not emitted in default reports, but it exists in the input file and in
memory during compilation. The executable policy stores hashed feature bucket counts so it
can classify new text. Feature hashing removes raw vocabulary from the artifact; it does
not anonymize the workload. Task names, model identifiers, metadata, prompt IDs, and
deterministic hashes may also be sensitive.

Use generalized labels, restrictive file permissions, and a separate publication review.
See [the threat model](THREAT_MODEL.md) and
[architecture artifact boundary](ARCHITECTURE.md#artifact-and-export-boundaries).
