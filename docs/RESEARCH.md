# Research and positioning notes

Last checked: 2026-08-04. Project capabilities change; verify primary documentation again
before making a release claim.

## Adjacent projects

- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) provides a
  programmable routing layer for heterogeneous model deployments.
- [RouteLLM](https://github.com/lm-sys/RouteLLM) covers learned routing, serving, and
  evaluation.
- [Aurelio Semantic Router](https://github.com/aurelio-labs/semantic-router) focuses on
  semantic intent and tool routing.
- [UIUC LLMRouter](https://github.com/ulab-uiuc/LLMRouter) collects routing algorithms and
  deployment components.
- [SmarterRouter](https://github.com/peva3/SmarterRouter) covers Ollama/llama.cpp discovery,
  profiling, VRAM management, and routing.
- [Hugging Face Chat UI router](https://huggingface.co/docs/chat-ui/configuration/llm-router)
  supports heuristic `default`, `multimodal`, and `agentic` routes.
- [Hugging Face LightEval](https://github.com/huggingface/lighteval) runs broad model
  evaluations.
- [RouterBench](https://github.com/withmartian/routerbench) provides multi-model response,
  cost, and correctness data for router research.
- The [LLMRouterBench paper](https://arxiv.org/abs/2601.07206) studies routers under a
  unified evaluation and emphasizes the importance of simple baselines and model-pool
  recall alongside routing method choice.

This list is positioning context, not an endorsement or exhaustive survey.

## RouteFoundry's boundary

RouteFoundry is not a gateway and does not claim a universally best routing algorithm. It
packages the workflow between existing evaluation results and an inspectable runtime
decision:

```text
validate -> isolate train/dev/test -> compare baselines -> constrain -> compile -> audit -> export
```

The useful artifact is not a hosted predictor. It is a small policy whose fallback, pool,
routes, aggregate evidence, abstention behavior, and warm-model decision can be inspected
and tested locally. The core deliberately avoids provider clients, model downloads, model
training, and response grading.

The implementation includes simple baselines because a complicated router must earn its
complexity on the user's workload. It includes a non-deployable hindsight oracle only as
an upper-bound diagnostic. It makes model-pool selection visible rather than treating the
pool as a fixed, neutral input.

## Design consequences

1. **Bring a complete matrix.** Missing prompt/model results fail closed instead of being
   interpreted as poor performance.
2. **Keep tuning and testing separate.** Classifier fitting uses training data; pool and
   routes use development data; the held-out partition is evaluated once.
3. **Make the budget empirical.** The development quality-loss constraint and held-out
   bootstrap check are reported separately.
4. **Abstain.** The classifier's evidence score is uncalibrated, so unsupported inputs use
   the strong fallback.
5. **Require sequence evidence.** Complete, gap-free traces stay in one split and drive
   cluster uncertainty; switching/load penalties are unavailable without that order.
6. **Respect exporter expressiveness.** A semantic task route is not relabelled as a Hugging
   Face Chat UI capability route.
7. **Treat privacy as an artifact property.** Hashed feature counts and deterministic
   digests are pseudonymous, not anonymous.

## Claims that require new evidence

Do not describe RouteFoundry as:

- the “first,” “only,” or universally best router/auditor;
- “zero quality loss” outside a precisely defined workload, grader, split, observed value,
  and uncertainty interval;
- a true “cold-start benchmark” when only backend residency was controlled;
- “private” when a chosen adapter or downstream runtime sends prompts remotely;
- “production ready” before independent installation, integration, security, and failure
  evidence exists;
- guaranteed to save cost, improve quality, attract a particular number of users/stars, or
  produce a hiring outcome.

Synthetic fixtures demonstrate software behavior only. Hardware profiles demonstrate the
measured timing counters only. Real routing claims require a documented workload, licensed
inputs, appropriate grading, preserved failures, and reproducible commands.

See [the architecture](ARCHITECTURE.md), [threat model](THREAT_MODEL.md), and
[product evidence rules](../PLAN.md#evidence-rules).
