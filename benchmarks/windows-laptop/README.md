# Windows laptop Ollama profile

This directory contains one machine-specific RouteFoundry profiler run:
[profile.json](profile.json). It is committed as protocol evidence, not as a model ranking.

## Conditions

| Item | Recorded value |
|---|---|
| Run time | 2026-08-03 22:47–22:48 UTC |
| Ollama | `0.5.9` |
| Operating system | Windows, release `10`, version `10.0.26200`, AMD64 |
| Python | `3.11.15` |
| CPU metadata | 12 physical / 16 logical cores; `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel` |
| Memory | 16,870,006,784 bytes (15.71 GiB) |
| GPU (separate observation) | NVIDIA GeForce RTX 3050 Laptop GPU; 4,096 MiB; driver `566.36` |
| Repeats | 3 per model |
| Context length | 2,048 tokens |
| Generation | non-streaming, deterministic settings, one predicted token |
| Residency label | **backend-non-resident** |
| Cache label | **OS-cache-uncontrolled** |

The GPU row was observed separately with
`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`; it
is not captured by or cryptographically bound to `profile.json`, and it does not establish
GPU causality for the timings. The manifest does not record power mode, storage medium,
thermals, or competing processes.

## Backend-reported timings

Values below are medians and full ranges over all three repeats, rounded to 0.1 ms from
the unrounded values in `profile.json`.

| Installed model | Ollama tag details | `load_duration_ms` median [range] | `total_duration_ms` median [range] |
|---|---|---:|---:|
| `llama3.2:3b` | 3.2B, Q4_K_M, GGUF | 5,124.7 [5,115.1–5,424.5] | 6,228.1 [6,182.2–6,393.5] |
| `deepseek-r1:1.5b` | 1.8B, Q4_K_M, GGUF | 1,350.4 [1,344.8–12,768.7] | 1,495.9 [1,485.9–12,909.9] |
| `gemma:2b` | 3B, Q4_0, GGUF | 4,158.1 [2,915.3–7,735.8] | 5,488.2 [4,368.7–8,753.8] |

Three observations are enough to expose run-to-run spread, not to estimate a stable
population distribution. In particular, the first recorded DeepSeek repetition is far
above the later two. The protocol did not measure the cause. The operating-system cache is
uncontrolled, so these values are **not true cold-start timings** and should not be used as
universal performance claims.

Prompt token counts differ by model (29, 7, and 26 respectively), reflecting tokenizer
behavior. Prompt-evaluation and total durations are therefore not controlled cross-model
throughput comparisons. The profile predicts only one token and records no generated
text; it contains **no answer-quality evaluation**.

## Residency result

The initial resident set was empty. After all measurements, the profiler unloaded the
remaining profiled model, polled Ollama's eventually consistent process list, and recorded:

```json
{
  "status": "restored",
  "initial_models": [],
  "final_models": [],
  "missing_models": [],
  "additional_models": [],
  "errors": [],
  "expiry_deadlines_restored": false
}
```

`expiry_deadlines_restored` remains false by design because Ollama does not expose a
portable way to restore the previous remaining keep-alive duration. A matching resident set
does not imply restoration of that hidden state.

## Reproduce

Run only against models already installed in the selected Ollama instance:

```bash
uv run routefoundry ollama-profile \
  --models llama3.2:3b,deepseek-r1:1.5b,gemma:2b \
  --repeats 3 \
  --output benchmarks/windows-laptop/profile.json \
  --yes
```

This command temporarily unloads and loads the named models. Do not run it while another
workload depends on their residency. A new run replaces machine/time-specific evidence; use
a new directory when preservation matters.

Read the full [profiling methodology](../../docs/OLLAMA_METHODOLOGY.md) before comparing
runs. Model digests, installed sizes, every raw counter, timestamps, and protocol fingerprint
remain in [profile.json](profile.json).
