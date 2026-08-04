# Launch in 60 seconds

Everything is prepared. Two clicks and two pastes. Do it Tue–Thu, 12:00–17:00 UTC, on a day
you can glance at the thread for a couple of hours — that presence is what turns a post into
stars, and it is the only part nobody can do for you.

---

## Step 1 — Hacker News (30 seconds)

**Click:** [submit to Hacker News](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2FRitikPatill%2Froutefoundry&t=Show%20HN%3A%20RouteFoundry%20%E2%80%93%20measure%20which%20of%20your%20local%20models%20is%20best%20at%20what)

Title and URL arrive pre-filled. Press **submit**.

**Then immediately paste this as the first comment** (this is where the numbers and the
disclosure live — a Show HN without an author comment usually stalls):

```text
Author here. I kept downloading local models without knowing which one to actually use for a
given task, so I wrote something that measures them instead of guessing.

`routefoundry autopilot` finds the models you already have in Ollama, runs a 38-task
auto-gradable suite against each (arithmetic, structured extraction, classification, code
output prediction), and grades every answer deterministically - exact number, exact string,
JSON field, regex. No LLM judge, so there's no second model's bias to audit.

On my laptop, across 152 generations, no model won every category. deepseek-r1:1.5b (1.1 GB)
got 10/12 arithmetic word problems where llama3.2:3b (2.0 GB) got 2/12 - and then lost
structured extraction 4/9 against 9/9, at nine times the median latency. The smallest model I
have scored highest overall; the largest came third.

Raw rows, conditions and limits are committed in the repo, so you can check the grading rather
than trust the table.

What it does not claim: this measures verifiable short-answer ability, not open-ended
generation quality - a model that reasons well here may still write worse prose. Sample sizes
per task type are 3-12 prompts, so treat the per-task table as a signal to verify on your own
prompts, not a leaderboard. One machine, one quantisation per model, and the timings are
backend-reported with an uncontrolled OS cache, so they aren't cold-start numbers.

There's a second half of the tool that compiles a routing policy from the matrix. I'm not
publishing a speed claim for it, because when I tested it the number swung between 0.78x and
17x depending only on the random split seed. That, and why, is written up in
docs/PRE_LAUNCH_FINDINGS.md.

A real run takes minutes, not seconds: `routefoundry autopilot --limit 12` is about five.
```

---

## Step 2 — r/LocalLLaMA, the next day (30 seconds)

Stagger it. Two simultaneous posts look coordinated; a day apart looks like a person.

**Click:** [submit to r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/submit?title=I%20measured%204%20of%20my%20local%20models%20on%2038%20auto-graded%20tasks.%20No%20model%20won%20every%20category.)

Title arrives pre-filled. Paste this as the body:

```text
I had seven models sitting in Ollama and no idea which to use for what, so I measured instead
of guessing. 152 generations on a 16 GB laptop, greedy decoding, deterministic grading (exact
number / exact string / JSON field / regex - no LLM judge).

| task type | deepseek-r1:1.5b | llama3.2:3b | gemma:2b | codellama 7b |
|---|---|---|---|---|
| arithmetic (12) | **10/12** | 2/12 | 2/12 | 3/12 |
| extraction (9) | 4/9 | **9/9** | 7/9 | 8/9 |
| classification (8) | 6/8 | **7/8** | **7/8** | 2/8 |
| code output (6) | **5/6** | 3/6 | 1/6 | 4/6 |
| median latency | 14.6 s | 1.6 s | 2.7 s | 6.3 s |

The 1.1 GB reasoning model is 5x better at arithmetic than the 2 GB general model and less
than half as good at extraction. Size did not predict accuracy - smallest scored highest
overall, largest came third.

Caveats up front: 3-12 prompts per category, so single items move percentages a lot. Only
auto-gradable short-answer tasks, which excludes the open-ended writing bigger models are
usually better at. One machine, Q4 quants, backend-reported timings with an uncontrolled OS
cache.

`<think>` blocks are stripped before grading, so reasoning models are scored on the answer
rather than punished for format - that turned out to matter a lot for deepseek.

Tool is MIT and runs entirely locally: `routefoundry autopilot --limit 12` (~5 min). It never
pulls or deletes models. Raw rows are in the repo:
https://github.com/RitikPatill/routefoundry

Genuinely interested in whether this holds on other fleets - if you run it, post your table,
including the cases where it disagrees with mine.
```

---

## Step 3 — be in the thread

The only part that matters after posting. Check every 20–30 minutes for the first two hours.
Prepared answers to the ten most likely objections are in
[LAUNCH_DRAFTS.md](LAUNCH_DRAFTS.md#objections-and-honest-answers). Two rules: answer with the
raw rows, or concede the point. Never argue.

## What not to do

Do not ask anyone to upvote or star, do not post from a second account, do not repost the
same project unchanged if the first attempt stalls. A stalled launch costs nothing; a flagged
account costs the repo, and the repo is the part that gets you interviews.
