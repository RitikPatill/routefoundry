# Threat model

This document describes RouteFoundry v0.1's security and privacy boundary. It is not a
claim that an artifact is anonymous or that a model-routing decision is safe for every
application.

## Assets

- raw prompts, task labels, prompt identifiers, and upstream model responses;
- grader outputs, costs, latency, load measurements, and workload distribution;
- model inventory, digests, runtime versions, and host metadata;
- upstream evaluator/provider credentials, which RouteFoundry does not need;
- integrity of compiled policies, CI comparisons, and published benchmark claims.

## Trust boundaries

### Local core

Validation, splitting, audit, compilation, reporting, export, and policy routing operate on
local files and create no network client. They enable no telemetry. The user chooses input
and output paths and is responsible for their permissions and backups.

### Executable policy

`router.json` crosses a more sensitive boundary than the default report. It contains model
and task names, aggregate metrics, and hashed classifier feature bucket counts. Feature
hashing prevents raw vocabulary serialization but is pseudonymization, not anonymization.
Dictionary and correlation attacks may still recover facts about a low-entropy workload.

Workload fingerprints and split assignment digests are also deterministic pseudonyms. A
party with candidate prompt IDs or prompt text may be able to test guesses.

### Optional Ollama adapter

The profiler connects only to the configured Ollama endpoint. Localhost is the safe
default. A remote endpoint is separately trusted to receive the fixed profiling request;
transport authentication and confidentiality are outside RouteFoundry's boundary. The
adapter temporarily changes model residency and then performs best-effort restoration.

### Distribution surfaces

Package indexes, GitHub, a Hugging Face Space/dataset, CI runners, browsers, shells, and
downstream inference clients are external systems with their own policies and logs. The
offline core guarantee does not mean installation is network-free or that a downstream
runtime keeps prompts local.

## Threats and controls

| Threat | Current control | Residual risk |
|---|---|---|
| Credential committed to Git | broad ignore rules, tracked/untracked secret scanner, push protection guidance, manual staged review | scanners miss novel or transformed secrets; Git history persists |
| Prompt or response leaked by a report | recursive denylist, raw content off by default, classifier internals always removed, escaped HTML, no scripts/remote assets | task/model names, metrics, hashes, or unknown key names may still disclose context |
| Vocabulary leaked by a policy | stable feature hashing; raw tokens are never serialized | bucket frequencies remain pseudonymous and may support guessing |
| Prompt leaked while routing | Python API does not log or persist prompt text | CLI arguments can appear in shell history, process listings, terminal capture, or crash tooling |
| Host exhausted by JSONL | strict UTF-8/JSON/type checks, 64 MiB file limit, 1 MiB line limit, 1,000,000-row limit | an accepted large complete matrix is held in memory |
| Missing evidence silently biases routing | exact prompt/model uniqueness and complete candidate matrix | upstream evaluator can still omit prompts before producing the file |
| Malicious object execution | plain JSON only; duplicate keys and non-finite values rejected; no pickle or dynamic import | metadata content is untrusted text and must stay escaped downstream |
| Policy tampering | strict schema, ranges, pool/route consistency, fail-closed loader | artifacts are not signed in v0.1; protect them with normal supply-chain controls |
| Misleading held-out claim | leakage-resistant prompt/whole-trace split, explicit development/held-out labels, unit-appropriate bootstrap interval, baselines, visible failures | grader bias, leakage before ingestion, distribution shift, and small samples remain |
| Oracle presented as deployable | report marks hindsight oracle non-deployable | downstream summaries can still remove the warning |
| Fabricated switching result | trace order must be explicit and gap-free for every prompt; otherwise switching is unavailable | supplied trace/load values may themselves be wrong |
| Ollama model mutation or response capture | only tags/process/version/generate routes; installed model exact-match; no pull/create/copy/push/delete; timing counters only | profiling changes residency and cannot restore original expiry deadlines |
| Credential-bearing Ollama URL or error leak | embedded credentials rejected; sanitized operation/status errors; proxy environment ignored by default local client | a separately configured remote service may log requests |
| Output overwrite | destination path is explicit and parent directories are created | report/policy files can replace an existing file; use a new directory or version control |
| Dependency or CI compromise | small runtime, lockfile, dependency review, CodeQL, immutable action pins | no dependency process eliminates ecosystem risk |

## Report privacy modes

Default report generation removes raw prompts, prompt IDs, responses/messages,
credential-like fields, and classifier feature tables. The static HTML contains escaped
summary JSON, no JavaScript, no forms, no iframes, and no remote resources.

An explicit `--include-prompts` mode weakens the intended boundary and should not be used
for publication without inspecting the resulting files. Even the default mode is a
denylist defense, not an information-flow proof. Before sharing, review both filenames and
contents of every artifact, including `router.json`, `summary.json`, human exports,
benchmark manifests, screenshots, and Git history.

## Benchmark integrity

Synthetic fixtures must be labelled non-evidence. Hardware measurements retain every
repeat, fixed protocol metadata, model digest, software version, and restoration result.
The Ollama profiler calls its condition backend-non-resident and OS-cache-uncontrolled;
it does not claim a true cold start or answer quality. See
[OLLAMA_METHODOLOGY.md](OLLAMA_METHODOLOGY.md).

Quality is user-supplied. A development-set constraint is a tuning result. A held-out
bootstrap check resamples prompts for untraced data or whole traces for complete traced
data; it covers sampling variation only, not grader uncertainty or future traffic. See
[the architecture evidence boundary](ARCHITECTURE.md#evidence-boundary).

## Non-goals

RouteFoundry does not prove that input data is licensed, consented, unbiased, or free of
personal data. It does not secure a remote inference runtime, validate upstream response
storage, sign artifacts, sandbox untrusted plugins, or guarantee a policy's real-world
quality. Security reports should follow [SECURITY.md](../SECURITY.md).
