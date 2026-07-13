---
title: "Loop Design Playbook — Real-World Cases to Your Own Autonomous Loops"
description: "Case studies of production autonomous Claude Code loops distilled into a concrete design and operations playbook."
---

> This guide surveys how practitioners actually run autonomous Claude Code loops in production —
> the Ralph Wiggum loop, Bun's 750k-line Rust port, Anthropic's internal automations — and distills
> them into a concrete design and operations playbook. It is the practical companion to
> [loop_engineering.md](loop_engineering.md) (the component-level theory); see
> [loop_design_playbook_demo.ipynb](loop_design_playbook_demo.ipynb) for quantitative simulations
> of loop convergence, verifier quality, and cost.

A single engineer prompting an agent interactively is bounded by their own attention: one task at
a time, review after every step, work stops when they sleep. The teams producing outputs that one
person demonstrably could not — a language runtime ported in eleven days, six repositories shipped
overnight — are not prompting better. They are running loops: systems that generate the prompts,
verify the outputs, and escalate only what they cannot resolve. This document is about designing
and operating such loops deliberately, rather than discovering their failure modes by accident.

---

## Table of Contents

1. [Why Loops Outperform a Single Session](#1-why-loops-outperform-a-single-session)
2. [Case Studies from the Wild](#2-case-studies-from-the-wild)
3. [The Loop Taxonomy — Five Mechanisms](#3-the-loop-taxonomy--five-mechanisms)
4. [The Five Design Decisions](#4-the-five-design-decisions)
5. [The Verifier Is the Product](#5-the-verifier-is-the-product)
6. [Worked Example — A Research-to-Output Loop Portfolio](#6-worked-example--a-research-to-output-loop-portfolio)
7. [Operations Runbook](#7-operations-runbook)
8. [Metrics and the Weekly Loop Review](#8-metrics-and-the-weekly-loop-review)
9. [Common Pitfalls](#9-common-pitfalls)
10. [References](#10-references)

---

## 1. Why Loops Outperform a Single Session

### The attention-bounded ceiling

In interactive use, the human is a synchronous dependency in every cycle. If a task needs $N$
agent-iterations and each requires human review latency $t_h$ plus agent time $t_a$, wall-clock
time is:

$$
T_{\text{interactive}} = N \,(t_a + t_h)
$$

A loop removes $t_h$ from all iterations except the final review, and runs $W$ workers in
parallel:

$$
T_{\text{loop}} \approx \frac{N \, t_a}{W} + t_h^{\text{final}}
$$

With overnight scheduling, even $t_a$ stops costing attention: the loop converts *calendar time
you were not using* (sleep, meetings, commute) into iterations.

### The compounding mechanism

The deeper reason loops beat sessions is not speed but *retry structure*. Model a loop iteration
as succeeding with probability $p$ against a verifier. The number of iterations to first success
is geometric:

$$
\mathbb{E}[N] = \frac{1}{p}, \qquad P(\text{success within } k) = 1 - (1-p)^k
$$

A task an agent one-shots only 20% of the time ($p = 0.2$) succeeds within 10 verified iterations
with probability $1 - 0.8^{10} \approx 89\%$. A human running one attempt per sitting gives up
long before iteration 10; a loop does not. This only works when two conditions hold — and both
are design obligations, not defaults:

1. **State accumulates outside the context window** (files, git history, progress logs), so
   iteration $k+1$ builds on $k$ instead of repeating it.
2. **The verifier is trustworthy**, so "success" means success. Section 5 treats this as the
   central problem it is.

### Key insight

> **A loop is a retry distribution plus a verifier plus persistent state.** Everything else —
> scheduling, worktrees, sub-agents — is infrastructure for running that retry distribution
> safely at scale. If you cannot write down the verifier, you do not have a loop; you have an
> unattended random walk.

**Why this matters**: "Output one person cannot produce" comes from three multipliers stacking —
parallelism ($W$), unattended calendar time, and verified retries ($1/p$). Design decisions that
sacrifice any of the three (no isolation, no scheduler, no verifier) collapse the loop back to an
expensive interactive session.

---

## 2. Case Studies from the Wild

### 2.1 The Ralph Wiggum loop — brute-force iteration with fresh context

Geoffrey Huntley's technique, now an official Anthropic Claude Code plugin, is the minimal viable
loop: feed the agent the *same prompt file* repeatedly until a completion condition fires. In its
purest form it is a bash `while true`; the plugin version implements it with a Stop hook that
blocks the agent's exit and re-injects the prompt inside one session.

```bash
# Purest form (Huntley): state lives in files + git, context is fresh each pass
while true; do
  cat PROMPT.md | claude -p --dangerously-skip-permissions
done
```

```bash
# Plugin form: completion promise + iteration cap
/ralph-loop "Build a REST API for todos: CRUD, validation, tests.
Output <promise>COMPLETE</promise> when done." \
  --completion-promise "COMPLETE" --max-iterations 50
```

Design properties worth stealing:

- **Fresh context per iteration**: the agent re-reads the repo each pass, so context-window rot
  never accumulates. Progress persists in code, tests, and a progress file — not in conversation.
- **Explicit completion promise**: an exact string the agent must emit, checked by string match.
  Crude, but unambiguous.
- **`--max-iterations` as the safety net**: the loop's budget is declared up front, because a
  loop on an impossible task will otherwise run forever with full confidence.
- Reported results: Huntley ran a months-long loop that built a complete programming language;
  YC hackathon teams shipped 6+ repositories overnight for roughly $297 in API cost. The
  philosophy — "it's better to fail predictably than succeed unpredictably" — is an operations
  statement: a loop whose failure mode is known (hits iteration cap, leaves a progress file) is
  deployable; one whose failure mode is unknown is not.

### 2.2 Bun's Zig-to-Rust port — orchestrated parallel workflows

Jarred Sumner used Claude Code dynamic workflows to port the Bun runtime from Zig to Rust:
roughly **750,000 lines of Rust in 11 days** from first commit to merge, with **99.8% of the
existing test suite passing**. Structure, per Anthropic's write-up:

1. **Mapping phase** — analyze Rust lifetimes for every struct before any porting began.
2. **Parallel porting phase** — hundreds of files ported concurrently, each with **two
   independent reviewer agents**.
3. **Fix loop phase** — automated iterate-until-build-succeeds loops.
4. **Overnight optimization workflow** — after the merge, a scheduled workflow hunted unnecessary
   data copies and opened one PR per finding for human review in the morning.

Design properties worth stealing:

- **The existing test suite was the verifier.** The 99.8% figure was only possible because the
  port target had thousands of pre-existing tests. The loop did not need a clever judge; it
  needed a large deterministic one.
- **Plan/execute separation**: orchestration state lives outside any single agent conversation,
  so the plan survives context exhaustion and interruption (checkpoint-resume, not restart).
- **One PR per finding** in the overnight workflow keeps each human review decision small —
  the loop scales output while keeping review granularity constant.

### 2.3 Anthropic internal teams — loops without engineers

Anthropic's own non-engineering teams run production loops: Growth Marketing built an agentic
workflow that ingests a CSV of hundreds of ads, identifies underperformers, and generates new
variants within strict character limits using two specialized sub-agents — hours of work
compressed to minutes. Finance teams describe processes in plain English and let the agent
execute and produce Excel reports. The recurring pattern across teams is *file-centric*: read
files, transform, write files, on a trigger.

Design property worth stealing: **the loop interface is a file format**, not a conversation.
A CSV in, a CSV out, with schema constraints acting as a cheap structural verifier.

### 2.4 `/goal`, routines, and scheduled agents — loops as a managed service

Claude Code's `/goal` runs an agent until a verifiable end state ("sort every file in Downloads
by type; stop after 30 turns"); routines/scheduled agents run cloud-side on cron triggers with
connected tools. The published guidance converges on the same rules: start read-only, declare
the done-condition precisely, cap turns, and put explicit "do not" clauses in the prompt.

### Case study comparison

| Case | Loop mechanism | Parallelism | Verifier | Human's role |
|---|---|---|---|---|
| Ralph Wiggum | Same prompt re-injected, fresh context | 1 (per loop) | Completion promise + tests | Sets prompt + cap; reviews at end |
| Bun port | Orchestrated dynamic workflow | Hundreds of subagents | Existing test suite + dual reviewers | Reviews merged result + per-PR |
| Anthropic marketing | Trigger → sub-agent pipeline | 2 specialized sub-agents | Schema/character-limit constraints | Approves output batch |
| Overnight PR workflow | Scheduled scan → PR per finding | Per-finding | Build + benchmarks | Reviews each small PR |
| Routines | Cloud cron + connectors | 1 | Prompt-declared done-state | Reads morning output |

**Why this matters**: none of these cases relies on a smarter model than you have access to.
They differ from interactive use purely in loop structure: pre-existing verifiers, state outside
the context window, declared budgets, and review granularity kept constant while volume scales.

---

## 3. The Loop Taxonomy — Five Mechanisms

Claude Code exposes five distinct loop mechanisms. Choosing the wrong one is the most common
first mistake, so classify before building:

| Mechanism | Trigger | Runs where | Terminates when | Best for |
|---|---|---|---|---|
| Ralph loop (`/ralph-loop`, bash `while`) | Manual start | Your machine, one session | Completion promise or max-iterations | One well-specified build task, walk away |
| `/goal` | Manual start | Your machine | Goal judged complete or turn cap | Bounded cleanup/refactor with checkable end state |
| `/loop <interval>` | Recurring timer | Your machine (session open) | You stop it | Poll-and-react while you work on something else |
| Routines / scheduled agents | Cron, cloud | Claude cloud | Per-run completion | Daily/weekly jobs that must run with your laptop closed |
| cron + `claude -p` (headless CLI) | System cron | Any machine you own | Script exit | Self-hosted nightly pipelines, full control of environment |

Two composition rules:

1. **Timer loops trigger goal loops.** A routine or cron job should do triage and then *launch* a
   bounded goal-style run for each actionable item — never do open-ended work directly inside the
   scheduled invocation.
2. **One loop, one repository responsibility.** A loop that reads repo A and writes repo B is a
   pipeline stage; a loop that writes to the same files another loop writes to is a race
   condition (use worktrees or separate branches).

**Why this matters**: the mechanisms differ in where they run and what stops them — which is
exactly what determines their failure modes. A `/loop` dies with your terminal; a routine does
not. A Ralph loop without a max-iteration cap has no failure mode except your API budget.

---

## 4. The Five Design Decisions

Every loop, regardless of mechanism, is specified by five decisions. Write them down before
writing the prompt — this table *is* the design document:

| Decision | Question | Bad answer (loop will fail) | Good answer |
|---|---|---|---|
| Trigger | When does an iteration start? | "Whenever I remember to run it" | Cron expression, git event, or upstream file appearing |
| Unit of work | What does one iteration produce? | "Progress on the project" | One brief, one PR, one executed notebook, one classified failure |
| Verifier | How does the loop know the unit is good? | "The agent checks its work" | Tests pass, schema validates, independent reviewer approves, metric ≥ threshold |
| Budget | What bounds an iteration and the loop? | None | Max iterations, turn cap, token/cost ceiling, wall-clock timeout |
| Escalation | What happens when verification fails repeatedly? | Silent retry forever / silent skip | After $k$ failures: write state file, notify human, stop |

### The unit-of-work rule

The single highest-leverage choice is making the unit of work **small, complete, and reviewable
in under five minutes**. The Bun overnight workflow's "one PR per data-copy finding" is the
canonical example. Compare:

```
Bad unit:   "Improve the test coverage of the repo"        (unbounded, unverifiable)
Good unit:  "Raise coverage of src/auth/ to ≥85%; one PR;  (bounded, verified by coverage
             coverage report attached; stop at 20 turns"     tool; reviewable in minutes)
```

### Budget arithmetic

Set budgets from the retry math, not vibes. If you estimate one-shot success $p$ and want the
loop to succeed with probability $q$, the iteration cap is:

$$
k \geq \frac{\ln(1-q)}{\ln(1-p)}
$$

For $p = 0.25$, $q = 0.95$: $k \geq \ln(0.05)/\ln(0.75) \approx 10.4$ — cap at 11–12 iterations.
If the loop hits the cap, the task was mis-specified or too large; the fix is decomposition, not
a larger cap. The companion notebook simulates this and the cost curves that follow from it.

**Why this matters**: loops fail at their unspecified margins. Every production incident in a
loop traces back to one of these five cells being left implicit — usually verifier or
escalation.

---

## 5. The Verifier Is the Product

The published guidance and the case studies agree on one point so consistently it deserves its
own section: the hard part of loop engineering is not the worker, it is the checker. "AI
leverage = your skill × your clarity" — and clarity means a machine-checkable definition of done.

### The verifier hierarchy

Prefer verifiers higher in this table; fall through only when the level above does not exist:

| Level | Verifier | Failure it cannot catch | Cost |
|---|---|---|---|
| 1 | Deterministic: compiler, type checker, schema | Semantically wrong but well-formed output | Near zero |
| 2 | Test suite (pre-existing, human-written) | Behavior outside test coverage | Low |
| 3 | Metric threshold (benchmark, coverage %, PDMS score) | Metric gaming, overfitting to the metric | Medium |
| 4 | Independent reviewer agent (no shared reasoning trace) | Shared blind spots of the model family | Medium-high |
| 5 | Human review | Reviewer fatigue at volume | Highest, does not scale |

The Bun port sat at level 2 with thousands of tests — which is why it scaled to hundreds of
parallel workers. A loop whose only verifier is level 4 should run at low parallelism; a loop
with only level 5 is not a loop, it is a queue for a human.

### Verifier error model

A verifier has a false-pass rate $\beta$ (accepts a bad unit) and false-block rate $\alpha$
(rejects a good one). With per-iteration true success probability $p$, the probability that a
unit the loop ships is actually defective is:

$$
P(\text{defective} \mid \text{shipped}) = \frac{(1-p)\,\beta}{p\,(1-\alpha) + (1-p)\,\beta}
$$

Two practical consequences, both simulated in the notebook:

- **False passes compound with volume.** A loop shipping 30 units/week at $\beta = 0.1$,
  $p = 0.5$ ships roughly 3 defects/week *silently*. Volume is exactly what loops create, so a
  verifier tolerable at human pace becomes intolerable at loop pace.
- **False blocks are the cheap failure.** They cost iterations (money, time) but not trust.
  When tuning a reviewer-agent prompt, bias it strict: raising $\alpha$ raises cost linearly;
  raising $\beta$ erodes the entire premise of unattended operation.

### Key insight

> **Build the verifier before the loop.** If the task has no pre-existing test suite, metric, or
> schema, the first loop to build is the one that creates the verifier (write tests, define the
> benchmark), not the one that does the work. This inverts the natural impulse and is the single
> most reliable predictor of whether a loop survives contact with reality.

**Why this matters**: every multiplier from Section 1 — parallelism, calendar time, retries —
multiplies *whatever the verifier accepts*. A strong verifier turns those multipliers into
output; a weak one turns them into a defect factory running while you sleep.

---

## 6. Worked Example — A Research-to-Output Loop Portfolio

This section designs a concrete loop portfolio for a working setup that already exists around
this repository: a nightly research radar (`research_loop/` — arXiv fetch → Claude triage →
morning digest of 3 briefs, plus weekly/monthly reviews on cron) and this `ML_report` repository
(topic reports + executed notebooks, added via an `/add-report` skill). The current loops are
**read-and-report** loops; the gap between them and "output one person could not produce" is a
set of **act loops** with verifiers. The portfolio below closes that gap incrementally.

### Loop portfolio

| # | Loop | Mechanism | Unit of work | Verifier | Budget | Escalation |
|---|---|---|---|---|---|---|
| L1 | Research radar (existing) | cron 03:00 + `claude -p` | Daily digest, ≤3 briefs | Schema of DIGEST.md; human reads | 1 run/night | Log + skip day |
| L2 | Brief → draft report | Weekly cron, headless | One draft report md + notebook for the week's top brief, on a branch | Notebook executes clean (nbconvert exit 0); reviewer agent checks conventions vs `/add-report` skill | 15 turns; 1 report/week | Draft branch left + summary in weekly review |
| L3 | Experiment sweep | Overnight Ralph loop, on demand | One config change + eval run + logged score | Benchmark metric (e.g., NAVSIM PDMS) must be computed and logged; improvement not required | max-iterations 12; wall-clock till 07:00 | Sweep log committed; best config reported |
| L4 | Notebook health | Weekly routine | Re-execute all repo notebooks; one issue per new failure | nbconvert exit codes | 1 run/week | Issue list, never auto-fix |
| L5 | Comprehension guard | Weekly routine | 3-question quiz on what L2/L3 changed this week | Human answers; loop grades | 10 min/week | N/A — this loop guards the human |

### The pipeline these form

```
research_loop (L1, nightly)
      |  briefs/ + weekly experiment backlog
      v
L2: draft report loop (weekly)  ----->  ML_report branch: draft md + executed ipynb
      |                                        |
      |                                        v
      |                                Reviewer agent (conventions, no shared trace)
      |                                        |
      v                                        v
L3: experiment sweep (on demand) ---->  Human morning review: merge or reject
      |  sweep logs, best config               |
      v                                        v
L4: notebook health (weekly) --------->  Issues only        L5: quiz (weekly) --> human
```

### Rollout order and gates

Deploy one loop at a time; each must pass its gate for two consecutive weeks before the next
goes live:

1. **L4 first** (read-only, deterministic verifier — lowest risk). Gate: zero false alarms.
2. **L2 next** in *draft-branch-only* mode; it must never push to main. Gate: ≥50% of drafts
   accepted by you with under 30 minutes of edits.
3. **L3** on a machine with the benchmark installed, initially with `--max-iterations 5`.
   Gate: sweep logs complete and reproducible; no orphaned processes.
4. **L5 alongside L2/L3** — it exists because Section 9's cognitive-surrender failure mode is
   real: once L2 writes your reports, a mechanism must verify *you* still understand them.

### L2 prompt skeleton (the act-loop template)

```markdown
Read research_loop/weekly/<latest>.md and pick the single recommended deep-read paper.
Following .claude/commands/add-report.md conventions EXACTLY:
1. Create branch draft/<topic>. Never commit to main.
2. Write <category>/<topic>.md and generate + execute the companion notebook.
3. Run the reviewer checklist; fix findings; re-execute.
Verification: `uv run jupyter nbconvert --execute` exits 0; no Traceback in outputs.
Budget: stop after 15 turns even if incomplete.
On failure: leave the branch, write a STATUS.md at branch root with what remains, stop.
Do not push. Do not open a PR. Do not modify any file outside the new branch.
```

**Why this matters**: a portfolio beats a mega-loop. Each loop above has one unit of work, one
verifier, and one escalation path — so when something breaks at 3 a.m., the blast radius is one
loop's output, and the weekly review (Section 8) can tune each independently.

---

## 7. Operations Runbook

Design gets a loop built; operations keeps it running for months. The checklist below is ordered
as a deployment sequence.

### 7.1 Pre-flight (before first unattended run)

- [ ] The five design decisions (Section 4) written at the top of the loop's prompt file
- [ ] Dry run executed interactively, watching every step
- [ ] Read-only mode or draft-branch-only mode for the first week
- [ ] `--max-iterations` / turn cap set; wall-clock timeout in the wrapper script
- [ ] Explicit "do not" clauses: never push to main, never delete, never touch `.env`
- [ ] Escalation path tested by *forcing* a failure and confirming the notification arrives

### 7.2 Scheduling and logging (self-hosted pattern)

```bash
# crontab — every loop gets: log file, timeout, lock against overlap
0 3 * * *  flock -n /tmp/l1.lock timeout 3h /home/user/loops/l1_research.sh \
             >> ~/loops/log/l1_$(date +\%F).log 2>&1
0 4 * * 0  flock -n /tmp/l2.lock timeout 4h /home/user/loops/l2_draft_report.sh \
             >> ~/loops/log/l2_$(date +\%F).log 2>&1
```

```bash
# Loop wrapper skeleton: state file + headless invocation + exit accounting
#!/usr/bin/env bash
set -euo pipefail
STATE=~/loops/state/l2.json
claude -p "$(cat ~/loops/prompts/l2.md)" \
  --output-format json > /tmp/l2_result.json
python3 ~/loops/bin/record_run.py --state "$STATE" --result /tmp/l2_result.json
# record_run.py appends: timestamp, exit status, turns used, cost, verifier verdict
```

Non-negotiables: `flock` (a slow run must not overlap the next), `timeout` (hangs must not eat
the machine), and one JSON line per run in a state file (Section 8 reads it).

### 7.3 Cost control

Token cost per iteration is dominated by context re-reads. Two levers:

- **Fresh-context loops (Ralph-style)** pay a full repo re-read every iteration — keep the repo
  the loop sees small (worktree with only the relevant subtree, or a scoped `CLAUDE.md`).
- **Polling loops** interact with prompt-cache TTL (~5 minutes): polling faster than the TTL
  keeps the cache warm and each poll cheap; polling at 5–20 minute intervals pays a full
  cache-miss re-read per poll for no benefit. Poll either fast (< TTL) or slow (≥ 20 min),
  never in between. The notebook quantifies this cliff.

Set a monthly budget per loop and record cost per run in the state file. The YC-hackathon data
point — 6 repos for $297 — is the right order of magnitude to expect for heavy overnight
build loops; a nightly triage loop should cost cents.

### 7.4 Kill switches

Every loop needs three stop mechanisms, tested before deployment:

1. **Declarative**: a `PAUSED` file in the loop's directory; the wrapper checks it first.
2. **Scheduler-level**: comment out the crontab line / disable the routine.
3. **Hard**: `pkill -f l2_draft_report` + removing the lock file.

### 7.5 Update discipline

The loop's prompt file is code: version it, and change it only via the weekly review — never
mid-week in reaction to a single bad run (one bad run is noise; the retry math expects it).

**Why this matters**: the failure mode of loop operations is not dramatic explosions — it is a
loop that silently stopped three weeks ago, or silently degraded and kept shipping. Locks,
timeouts, state files, and tested escalation are what make silence impossible.

---

## 8. Metrics and the Weekly Loop Review

A loop portfolio needs its own loop: a weekly, human-run review of the state files. Four numbers
per loop suffice:

| Metric | Definition | Healthy | Action when unhealthy |
|---|---|---|---|
| Completion rate | Runs hitting verifier-pass / total runs | > 70% | Below: unit of work too large — decompose |
| Escalation rate | Runs ending in escalation / total | 5–25% | ~0%: verifier too lax (nothing is ever hard?). >30%: task mis-scoped |
| Acceptance rate | Loop outputs you accept ≈ unedited / outputs reviewed | > 50% and rising | Falling: conventions drifted — update skill file, not the prompt |
| Cost per accepted unit | Total spend / accepted outputs | Stable or falling | Rising: iterations wasted — check verifier false-blocks |

The escalation-rate band deserves emphasis: **zero escalations is a red flag, not a triumph.**
A loop that never asks for help has either a verifier passing everything or a prompt that
quietly skips hard cases. The notebook builds a 30-day synthetic ops log with a mid-month
silent degradation and shows the completion-rate alert firing within days of onset, cost
following, and human acceptance lagging by nearly two weeks — the human is the slowest sensor.

The review itself is 20 minutes: read the four numbers per loop, read every escalation, spot-read
one *accepted* output end-to-end (this is the anti-cognitive-surrender step), and make at most
one prompt/skill change per loop per week.

**Why this matters**: without measurements, the only signal that a loop degraded is a bad output
you happened to catch — and Section 5 showed that at loop volume, the ones you catch are a
minority of the ones that shipped.

---

## 9. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| No completion condition | Ralph loop burns budget on an impossible task all night | Completion promise + `--max-iterations` computed from retry math (Section 4) |
| Verifier built after the loop | Early outputs look fine; defect rate discovered weeks later | Build/verify the checker first; loop stays read-only until it exists |
| Mega-loop | One loop plans, codes, reviews, deploys; failures are undebuggable | One unit of work per loop; compose loops via files/branches (Section 6) |
| Overlapping runs | Two cron invocations write the same state; corrupted output | `flock` in every wrapper; wall-clock `timeout` |
| Loop writes where another loop reads/writes | Race conditions, phantom diffs | One-writer rule per path; worktrees or dedicated branches per loop |
| Polling in the cache dead zone | 5–15 min polls cost full context re-read each time | Poll < cache TTL or ≥ 20 min, never between |
| Zero escalations celebrated | "It never bothers me!" — verifier passes everything | Treat 0% escalation as an alert; audit verifier false-pass rate |
| Prompt tuned after every bad run | Loop behavior oscillates; no baseline to compare | Change prompts only at weekly review; one change per loop per week |
| Auto-merge to main | One bad verifier pass lands in production/master branch | Loops open PRs or write draft branches; a loop never merges |
| Comprehension debt / cognitive surrender | You cannot explain last week's loop outputs | Weekly spot-read of one accepted output; quiz loop (L5); write summaries from memory first |
| Loop silently dead | Discovered weeks later that cron stopped firing | Heartbeat entry in state file every run; weekly review checks recency first |

---

## 10. References

- Geoffrey Huntley, "Ralph Wiggum as a software engineer" — https://ghuntley.com/ralph/
- Anthropic, Ralph Wiggum plugin (official) —
  https://github.com/anthropics/claude-code/blob/main/plugins/ralph-wiggum/README.md
- Anthropic, "Introducing dynamic workflows in Claude Code" (Bun Zig→Rust port case) —
  https://claude.com/blog/introducing-dynamic-workflows-in-claude-code
- Anthropic, "How Anthropic teams use Claude Code" —
  https://claude.com/blog/how-anthropic-teams-use-claude-code
- Addy Osmani, "Loop Engineering" — https://addyosmani.com/blog/loop-engineering/
  (component-level framework; covered in depth in [loop_engineering.md](loop_engineering.md))
- Sabrina Ramonov, "AI Loop Engineering: Build Autonomous Agents with Claude Code /goal +
  Routines" — https://www.sabrina.dev/p/loop-engineering-claude-code-goal-routines
- codecentric, "The Ralph Wiggum loop: autonomous code generation with a fresh context" —
  https://www.codecentric.de/en/knowledge-hub/blog/the-ralph-wiggum-loop-autonomous-code-generation-with-a-fresh-context
- frankbria/ralph-claude-code — community implementation with exit detection —
  https://github.com/frankbria/ralph-claude-code
