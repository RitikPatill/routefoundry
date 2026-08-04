---
license: mit
pretty_name: RouteFoundry Ollama Residency Profiles
tags:
  - ollama
  - benchmarking
  - model-routing
  - systems
---

# RouteFoundry Ollama residency profiles

> **Dataset-card scaffold:** this file is maintained in the RouteFoundry source repository.
> It does not claim that a Hugging Face dataset has been published. Before publication,
> copy reviewed manifests into the dataset repository and replace this notice with its
> immutable source revision.

## Dataset summary

This dataset is intended to hold consented, machine-labelled manifests produced by
RouteFoundry's safe Ollama profiler. Each manifest records backend-reported model loading
and one-token generation counters for already-installed models, a fixed protocol
fingerprint, software/model versions, coarse host metadata, and residency restoration
status.

The current source repository contains one candidate manifest:

- [Windows laptop profile](../../benchmarks/windows-laptop/profile.json), with its
  [human-readable interpretation](../../benchmarks/windows-laptop/README.md).

Every observation is **backend-non-resident** and **OS-cache-uncontrolled**. The data is
not a cold-start benchmark, model-quality evaluation, or universal ranking.

## Intended uses

- reproduce and improve RouteFoundry's profiler protocol;
- inspect run-to-run load-duration variation on named hardware/software conditions;
- test parsers and visualizations for model residency data;
- contribute transparent negative results and restoration outcomes.

Do not use the dataset to rank answer quality, infer unrecorded hardware causes, promise
latency on another machine, or select a model without workload-specific quality evidence.

## Data generation

```bash
routefoundry ollama-profile \
  --models MODEL_A,MODEL_B,MODEL_C \
  --repeats 3 \
  --output data/MACHINE_SLUG/profile.json \
  --yes
```

The command profiles exact installed tag names only. It does not pull, create, copy, push,
or delete models and does not retain prompts or generated text. It does temporarily change
backend residency. Read the complete
[profiling methodology](../../docs/OLLAMA_METHODOLOGY.md).

## Manifest structure

Each JSON manifest includes:

- schema/artifact version and UTC timestamps;
- residency and cache condition labels;
- fixed generation protocol metadata and prompt digest;
- Ollama version and safe installed-model tag metadata;
- every repeat's backend-reported durations and token counts;
- coarse operating-system, Python, CPU-core, processor, and memory metadata;
- initial/final resident sets, restoration actions/errors, and restoration status.

It deliberately excludes hostname, username, base URL, request headers, raw prompt,
generated text, and response body. Machine and model metadata can still fingerprint a
contributor; contribution therefore requires informed review and consent.

## Limitations

- Operating-system, filesystem, driver, and disk caches are not flushed or measured.
- Backend counters exclude HTTP/client overhead.
- Three default repetitions do not establish a stable latency distribution.
- Tokenization differs across models, so prompt-evaluation duration is not a controlled
  cross-model throughput comparison.
- Power mode, thermals, storage, competing processes, and GPU are absent unless a future
  schema adds them explicitly.
- Exact original keep-alive expiry deadlines cannot be restored through the portable API.
- The profiler makes no answer-quality measurement.

## Licensing and provenance

The manifest schema, tooling, card, and repository-authored measurements are released under
the repository's MIT license. This does not license, redistribute, or grant rights to the
profiled model weights. Model names and digests identify locally installed artifacts only.

Before adding a community profile, record the RouteFoundry revision, preserve all repeats
and errors, confirm the contributor consented to the public host/model metadata, and review
the JSON for accidental identifiers. Do not accept hand-edited “best run” summaries without
the raw manifest.

## Publication checklist

- [ ] Copy each reviewed manifest to a stable `data/<machine-slug>/profile.json` path.
- [ ] Record the source RouteFoundry commit and profiler version.
- [ ] Confirm every contributor's public-data consent.
- [ ] Validate that raw prompts, generated text, usernames, hostnames, URLs, and credentials
      are absent.
- [ ] Preserve restoration failures and timing outliers.
- [ ] Update this card's inventory and limitations from the committed data.
- [ ] Test the card links in the actual Hugging Face dataset repository.
