# Launch drafts

Drafts only. Nothing here is posted automatically, and nothing should be posted that the
maintainer has not read, agrees with, and can defend in a comment thread. Every number below
comes from [`benchmarks/laptop-4model-suite-v1`](../benchmarks/laptop-4model-suite-v1/) and
must be re-checked against that directory before posting — if a rerun changes it, change the
post, not the data.

## Ground rules

- No coordinated voting, no asking friends to star, no reposting an unchanged project.
- Disclose authorship in the first comment.
- Answer criticism with the raw rows, or concede the point. Do not argue.
- If a claim cannot be reproduced from the committed artifacts, delete the claim.

---

## Show HN

**Title candidates** (under 80 characters; quantified; no hype adjectives):

1. `Show HN: My 1.1GB local model beat my 2GB one 10/12 on math, lost 4/9 to 9/9 on extraction`
   *(83 chars — trim to)* `Show HN: My 1.1GB model beat my 2GB model on math and lost badly on extraction`
   Leads with the surprise. Strongest hook, but reads as a blog post rather than a tool.
2. `Show HN: RouteFoundry – measure which of your local models is best at what`
   Clearest statement of what it does. Safest choice.
3. `Show HN: Benchmark your installed Ollama models in one command`
   Most concrete verb, weakest differentiation.
4. `Show HN: No model won every category across 152 measured local generations`
   Intriguing but vague about what it is.
5. `Show HN: RouteFoundry – auto-graded benchmarks for the models already on your machine`
   Descriptive, slightly long.

**Recommended: #2**, with the surprise in the first line of the body. A title that reads as
a finding invites "so what's the tool?"; a title that names the tool lets the finding land
in the post.

**Body / first comment:**

> Author here. I kept downloading local models without knowing which one to actually use for
> a given task, so I wrote something that measures them instead of guessing.
>
> `routefoundry autopilot` finds the models you already have in Ollama, runs a 38-task
> auto-gradable suite against each (arithmetic, structured extraction, classification, code
> output prediction), and grades every answer deterministically — exact number, exact string,
> JSON field, regex. No LLM judge, so there's no second model's bias to audit.
>
> On my laptop, across 152 generations, no model won every category. deepseek-r1:1.5b
> (1.1 GB) got 10/12 arithmetic word problems where llama3.2:3b (2.0 GB) got 2/12 — and then
> lost structured extraction 4/9 against 9/9, at nine times the median latency. The smallest
> model I have scored highest overall; the largest came third.
>
> Raw rows, conditions and limits are committed in the repo, so you can check the grading
> rather than trust the table.
>
> What it does **not** claim: this measures verifiable short-answer ability, not open-ended
> generation quality — a model that reasons well here may still write worse prose. Sample
> sizes per task type are 3–12 prompts, so treat the per-task table as a signal to verify on
> your own prompts, not a leaderboard. It's one machine, one quantisation per model, and the
> timings are backend-reported with an uncontrolled OS cache, so they're not cold-start
> numbers.
>
> There's a second half of the tool that compiles a routing policy from the matrix. I'm not
> publishing a speed claim for it, because when I tested it the number swung between 0.78x
> and 17x depending only on the random split seed. That, and why, is written up in
> docs/PRE_LAUNCH_FINDINGS.md.
>
> A real run takes minutes, not seconds: `routefoundry autopilot --limit 12` is about five.

---

## r/LocalLLaMA

This audience is hostile to marketing and rewards measurement. Lead with numbers, keep the
tool secondary, invite contradiction.

**Title:** `I measured 4 of my local models on 38 auto-graded tasks. No model won every category.`

**Body:**

> I had seven models sitting in Ollama and no idea which to use for what, so I measured
> instead of guessing. 152 generations on a 16 GB laptop, greedy decoding, deterministic
> grading (exact number / exact string / JSON field / regex — no LLM judge).
>
> | task type | deepseek-r1:1.5b | llama3.2:3b | gemma:2b | codellama 7b |
> |---|---|---|---|---|
> | arithmetic (12) | **10/12** | 2/12 | 2/12 | 3/12 |
> | extraction (9) | 4/9 | **9/9** | 7/9 | 8/9 |
> | classification (8) | 6/8 | **7/8** | **7/8** | 2/8 |
> | code output (6) | **5/6** | 3/6 | 1/6 | 4/6 |
> | median latency | 14.6 s | 1.6 s | 2.7 s | 6.3 s |
>
> The 1.1 GB reasoning model is 5x better at arithmetic than the 2 GB general model and less
> than half as good at extraction. Size did not predict accuracy — smallest scored highest
> overall, largest came third.
>
> Caveats up front: 3–12 prompts per category, so single items move percentages a lot. Only
> auto-gradable short-answer tasks, which excludes the open-ended writing that bigger models
> are usually better at. One machine, Q4 quants, backend-reported timings with an
> uncontrolled OS cache.
>
> `<think>` blocks are stripped before grading, so reasoning models are scored on the answer
> rather than punished for format — that turned out to matter a lot for deepseek.
>
> The tool is MIT and runs entirely locally: `routefoundry autopilot --limit 12` (~5 min).
> It never pulls or deletes models. Raw rows are in the repo.
>
> Genuinely interested in whether this holds on other fleets — if you run it, post your table,
> including the cases where it disagrees with mine.

---

## Objections, and honest answers

| Objection | Answer |
|---|---|
| "Your suite is too easy / too small." | Correct, and stated in the README: 3–12 prompts per task type. It's a signal to verify on your own prompts. Bring a harder suite — the format is one JSON line per task. |
| "38 prompts proves nothing." | It proves the models differ per category, which is all it claims. It does not establish effect sizes; that's why no confidence interval is quoted on the per-task table. |
| "Deterministic grading punishes verbose models." | It did, and that was a bug — `gemma:2b` answered "Negative." then explained, and scored zero. First-line and final-line answers are both accepted now, and `<think>` blocks are stripped. Tests cover both. |
| "This is just LiteLLM / RouteLLM / semantic-router." | Those route traffic. This measures your models and audits whether routing helps, then hands you a policy file. It has no gateway and doesn't want one. |
| "Why not just always use the biggest model?" | On this fleet the biggest model came third overall and lost arithmetic 3/12 to 10/12. That's the whole argument, and it's why the tool measures rather than assumes. |
| "Your latency numbers are meaningless." | Largely yes across machines — they're backend-reported with an uncontrolled OS cache and thermal state. They're comparable within one run on one machine, which is what routing decisions use. |
| "Where's the routing speed-up number?" | Deliberately absent. It swung 0.78x–17x on the split seed alone, so it isn't reportable yet. Written up in docs/PRE_LAUNCH_FINDINGS.md. |
| "You benchmarked models you happened to have." | Yes. It's a personal fleet, not a survey. The tool exists so you can run it on yours. |
