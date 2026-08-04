# Pre-launch findings: why v0.1 does not publish a headline routing number

Last reviewed: 2026-08-04

An adversarial review executed `optimize.py::audit` against simulated autopilot output
matching this project's measured hardware, before any public claim was written. It found
that the obvious headline — "routing kept X% of quality at Y-times lower latency" — is not
defensible at the sample sizes and model pools autopilot actually produces. The findings
are recorded here rather than quietly fixed, because they constrain what this project may
claim and they are the reason the launch material contains no speed multiplier.

## 1. On a small local pool, the compiled router equals the trivial baseline

With three small local models and a 60-prompt suite, across 8 split seeds:

```
LATENCY_FACTOR = 1.00x   QUALITY_RETENTION = 100.0%   switch_count = 0
compiled == always-fastest == task-only == always-strongest
```

The router selects one model for every task. There is no routing decision to report, and
therefore no headline. A claim built on this configuration would be describing a tie.

## 2. Adding a large model produces a number that is not reproducible

Adding `deepseek-r1:8b` creates a real quality/latency spread and a headline appears. It
is unusable, because only the split seed varies between these runs:

| seed | latency factor | quality retention | held-out check |
|---|---|---|---|
| 42 | 1.00x | 100.0% | satisfied |
| 7 | 1.00x | 100.0% | satisfied |
| 1 | 2.34x | 100.0% | **failed** (CI 0.167 vs 0.05 budget) |
| 13 | 17.02x | 75.0% | **failed** (loss 0.222 = 4.4x budget) |
| 2026 | 0.78x (slower) | 100.0% | satisfied |

Same data, same models, same suite: 0.78x to 17.02x. Publishing any single value invites
the first person who reruns with another seed to contradict it publicly.

## 3. The quality-loss budget is inert at these sample sizes

`split_observations` splits by whole traces. At the default seed a 60-prompt suite yields a
development set of 6 prompts — **n = 1 per task**. With binary grading, per-(task, model)
quality is then 0.0 or 1.0, so a budget must exceed 1.0 to change any admission decision:

```
--max-quality-loss 0.0 | 0.02 | 0.05 | 0.20 | 0.90  ->  1 model, factor 1.00x  (all identical)
```

The flag does nothing across its entire range. At `--repeats 3` it becomes non-monotonic: a
*looser* budget can produce a faster router. Since the budget is this project's central
claim, that must be fixed before the claim is repeated in public.

## 4. Warm hysteresis disables the switch logic it exists to model

`policy.py` keeps the warm model whenever `load_ms + switch_cost >= latency_gain`. Measured
`load_ms` on this hardware is 1,350–5,125 ms, while plausible latency differences between
2–3B local models are a few hundred milliseconds. The penalty therefore always dominates and
the router never switches: `switch_count = 0` in every default-configuration run. The
residency analysis is real code with no effect at this scale.

## 5. Three of the eight baselines are the same row on an all-local pool

`warm_only = dict(all_strongest)` makes warm-only a byte-identical copy of always-strongest.
With every `cost_usd = 0`, always-cheapest also collapses onto always-strongest. On a local
run there are five distinct strategies plus a non-deployable oracle — not eight. Presenting
three identical rows under three different descriptions is trivially checkable and would be
read, correctly, as padding.

## 6. Runtime is minutes, not seconds

Measured prompt evaluation costs 20–56 ms/token on this laptop, so a realistic suite prompt
costs 3–10 s before a single token is generated. Roughly 5 models x 80 prompts is **35–75
minutes**. Any "60-second" framing would apply to the synthetic demo only, never to a real
measured run.

## What this changes

- v0.1 publishes **no speed multiplier and no quality-retention percentage**. The honest
  deliverable is the measurement and audit workflow itself, plus per-model verifiable
  accuracy, which is reproducible.
- Reporting must state the split seed and show sensitivity across seeds, not a single value.
- Baseline tables must collapse identical strategies on all-local pools instead of listing
  them separately.
- Before any routing claim is made, the budget's resolution problem (§3) and the hysteresis
  scale problem (§4) need fixes, and the suite needs enough prompts per task for a
  development set to carry more than one sample per task.
