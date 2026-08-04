# Ollama profiling methodology

RouteFoundry measures backend counters reported by an existing Ollama installation. It
does not install, download, copy, push, create, or delete models. The profiler is narrower
than a benchmark runner: it captures model residency/load evidence for models the user
already has and makes **no answer-quality claim**.

## Measurement label

Every observation is labelled **backend-non-resident** and
**OS-cache-uncontrolled**. An explicit Ollama unload removes the selected model from the
backend process, but it does not flush operating-system filesystem caches, driver caches,
or disk caches. These numbers must never be described as true cold starts.

The durations are Ollama's backend-reported nanosecond counters, preserved verbatim and
also converted to milliseconds:

- `load_duration`;
- `total_duration`;
- `prompt_eval_duration`;
- `eval_duration`;
- `prompt_eval_count`;
- `eval_count`.

RouteFoundry does not replace these counters with client wall-clock timing. HTTP, JSON, and
client scheduling overhead are outside the reported backend durations.

## Fixed protocol

For each requested model and repeat, RouteFoundry performs this sequence:

1. Read `GET /api/tags` and require an exact match against an installed `name` or `model`.
2. Read `GET /api/ps` and snapshot the initial resident model set before mutation.
3. Read `GET /api/version` when available.
4. Send `POST /api/generate` with `keep_alive: 0` to unload the selected model.
5. Poll `/api/ps` for up to 10 seconds and fail if the model remains resident.
6. Send one non-streaming generation with a constant harmless short prompt,
   `num_predict: 1`, context length `2048`, `temperature: 0`, `seed: 0`, and a five-minute
   profile keep-alive.
7. Retain only timing/token counters and timestamps. Do not retain generated text, context
   arrays, raw prompts, response bodies, or request headers.
8. Repeat steps 4–7 three times by default (configurable from one to twenty).
9. Restore the original resident set best-effort, poll for eventual consistency, and
   record the initial/final sets, actions, differences, and sanitized errors.

The fixed prompt is represented in artifacts only by its SHA-256 digest and UTF-8 byte
length. This permits protocol comparison without establishing raw prompt logging as an
artifact feature.

## Model and API safety

Only four Ollama routes are used:

- `GET /api/tags`;
- `GET /api/ps`;
- `GET /api/version`;
- `POST /api/generate`.

There are no calls to pull, push, create, copy, or delete endpoints. A model is never sent
to `generate` unless it appeared in the initial tag listing. A URL containing embedded
credentials is rejected. The default local HTTP client ignores proxy environment
variables so a configured proxy cannot silently redirect localhost profiling.

API exceptions retain only the operation, exception type, and HTTP status when available.
They exclude response bodies and raw transport exception strings, either of which could
contain prompts, tokens, or credential-bearing URLs.

## Residency restoration

Profiling necessarily changes backend residency. After profiling—or after a profiling
error—RouteFoundry:

1. unloads a profiled model if it was not resident initially;
2. preloads any initially resident model that disappeared, using an empty generation
   request;
3. polls the process list for up to 10 seconds for the state changed by this invocation;
4. reads `/api/ps` again and records the actual final set.

It never unloads an unrelated model that appeared concurrently because that model may
belong to another process. Ollama does not expose a portable way to restore a model's exact
original expiry deadline, so `expiry_deadlines_restored` is always `false`. Memory pressure
may evict another model during restoration; the manifest will then report `changed` rather
than claiming success.

## Manifest and reproducibility

The schema-versioned JSON manifest contains:

- UTC run and observation timestamps;
- Ollama version when obtainable;
- canonical model name, tag digest, installed byte size, and safe tag details;
- fixed protocol settings and prompt fingerprint;
- raw nanosecond counters, converted millisecond values, and token counts;
- operating system, architecture, Python, CPU-core, processor, and total-RAM metadata;
- an explicit residency restoration report.

It excludes hostname, username, base URL, headers, raw prompts, and generated text. Writes
use a temporary file in the destination directory followed by atomic replacement, so a
failed write does not leave a partially written manifest.

For comparison, hold Ollama version, model digest, machine, power mode, context length,
repeat count, and background workload constant. Even then, treat results as local empirical
observations, not universal rankings. First and later repetitions can differ sharply
because operating-system caches are uncontrolled. Report all repetitions and negative
results rather than selecting only a favorable run.

The committed example and its exact limitations are in
[benchmarks/windows-laptop/README.md](../benchmarks/windows-laptop/README.md). The broader
privacy boundary is in [THREAT_MODEL.md](THREAT_MODEL.md).
