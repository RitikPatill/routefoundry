# Four local models, suite v1, one Windows laptop

Measured 2026-08-04 with `routefoundry autopilot`. Raw rows are committed beside this file:
[`observations.jsonl`](observations.jsonl) (152 rows) and [`trials.jsonl`](trials.jsonl)
(per-model grader verdicts and timings).

## Conditions

| Item | Value |
|---|---|
| Command | `routefoundry autopilot --models deepseek-r1:1.5b,gemma:2b,llama3.2:3b,codellama:latest` |
| Suite | bundled `suite_v1` — 38 auto-gradable tasks, 5 task types |
| Generations | 152 (4 models x 38 tasks), 0 errors, 1,179 s wall |
| Decoding | `temperature 0`, `seed 0`, `num_predict 800`, `num_ctx` default |
| Host | Windows 11, Intel i5-1240P, 16 GB RAM, RTX 3050 4 GB, Ollama 0.5.9 |
| Grading | deterministic (exact number / exact string / JSON field / substring / regex) |
| Residency | one eviction per model before its first task; **OS cache uncontrolled** |

## Overall

| Model | Installed size | Correct | Accuracy | Median latency |
|---|---:|---:|---:|---:|
| deepseek-r1:1.5b | 1.12 GB | 27/38 | **71%** | 14.6 s |
| llama3.2:3b | 2.02 GB | 23/38 | 61% | 1.6 s |
| codellama:latest (7B) | 3.83 GB | 19/38 | 50% | 6.3 s |
| gemma:2b | 1.68 GB | 18/38 | 47% | 2.7 s |

The most accurate model here is also the smallest, and the largest is third. Size did not
predict accuracy on this suite.

## By task type — the actual finding

| Task type | deepseek-r1:1.5b | llama3.2:3b | gemma:2b | codellama |
|---|---:|---:|---:|---:|
| reasoning_math (12) | **10/12 (83%)** | 2/12 (17%) | 2/12 (17%) | 3/12 (25%) |
| code_gen (6) | **5/6 (83%)** | 3/6 (50%) | 1/6 (17%) | 4/6 (67%) |
| extraction_structured (9) | 4/9 (44%) | **9/9 (100%)** | 7/9 (78%) | 8/9 (89%) |
| classification (8) | 6/8 (75%) | **7/8 (88%)** | **7/8 (88%)** | 2/8 (25%) |
| code_fix (3) | 2/3 (67%) | 2/3 (67%) | 1/3 (33%) | 2/3 (67%) |

**No model won every category.** The 1.1 GB reasoning model was five times better at
arithmetic word problems than the 2 GB general model — and less than half as good at
structured extraction, where the 2 GB model scored 9/9. It is also nine times slower at the
median. A workload mixing these task types has no single best choice, which is the case for
routing; a workload made only of extraction should just use `llama3.2:3b`.

Reasoning output is stripped before grading (`<think>` blocks and answer preambles), so the
reasoning model is scored on its answer rather than penalised for its format.

## What these numbers do not support

- **Not general capability.** Every task is auto-gradable and short-answer, which excludes
  open-ended generation, long-context work, and instruction following beyond output format.
  A model that reasons well here may still write worse prose.
- **Not a hardware ranking.** One machine, one Ollama version, one quantisation per model.
  Backend-reported timings with an uncontrolled OS cache; the profile deliberately does not
  claim cold-start numbers.
- **Small samples per category.** 3 to 12 prompts per task type. A 1-item swing moves
  `code_fix` by 33 points. Treat the per-task table as a signal to verify on your own
  prompts, not as a leaderboard.
- **No routing claim.** Per-model accuracy is reproducible; the compiled router's speed
  advantage is not yet, for the reasons in [`../../docs/PRE_LAUNCH_FINDINGS.md`](../../docs/PRE_LAUNCH_FINDINGS.md).

## Reproduce

```bash
routefoundry autopilot \
  --models deepseek-r1:1.5b,gemma:2b,llama3.2:3b,codellama:latest \
  --output out/real.jsonl
```

Grading is deterministic and decoding is greedy, so answer correctness should reproduce on
the same model versions. Latency will not: it depends on your hardware, and on this laptop
it varies with thermal state and background load.
