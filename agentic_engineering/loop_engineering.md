# Loop Engineering — Building Autonomous AI Agent Systems

> This guide translates Addy Osmani's "Loop Engineering" framework into a practical reference for
> engineers who want to move beyond manually prompting agents and toward designing the systems that
> prompt agents for them. It covers all six architectural components — Automations, Worktrees,
> Skills, Plugins, Sub-agents, and Persistent State — with concrete examples, decision tables, and
> a step-by-step implementation roadmap. The final sections address the three critical debts that
> loops introduce and the pitfalls that cause them to fail silently.

---

## Table of Contents

1. [The Core Paradigm Shift](#1-the-core-paradigm-shift)
2. [Component 1 — Automations: The Heartbeat](#2-component-1--automations-the-heartbeat)
3. [Component 2 — Worktrees: Parallel Isolation](#3-component-2--worktrees-parallel-isolation)
4. [Component 3 — Skills: Embedded Knowledge](#4-component-3--skills-embedded-knowledge)
5. [Component 4 — Plugins and Connectors: External Integration](#5-component-4--plugins-and-connectors-external-integration)
6. [Component 5 — Sub-agents: Distributed Verification](#6-component-5--sub-agents-distributed-verification)
7. [Component 6 — Persistent State: Memory](#7-component-6--persistent-state-memory)
8. [Loop Architecture Patterns](#8-loop-architecture-patterns)
9. [Getting Started — Implementation Roadmap](#9-getting-started--implementation-roadmap)
10. [Critical Limitations — The Three Debts](#10-critical-limitations--the-three-debts)
11. [Common Pitfalls](#11-common-pitfalls)
12. [References](#12-references)

---

## 1. The Core Paradigm Shift

> "Loop engineering is replacing yourself as the person who prompts the agent. You design the
> system that does it instead." — Addy Osmani

### The Old Paradigm

In direct prompting, the engineer is the orchestrator. Every cycle requires human initiation:

```
Human writes prompt
      |
      v
Agent executes
      |
      v
Human reviews output
      |
      v
Human writes follow-up prompt
      |
      v
(repeat)
```

The bottleneck is the human. The agent can only work as fast as the engineer can context-switch,
read output, and compose the next instruction. Skilled prompt engineers learn to write longer,
more complete prompts — but the loop still closes through a human.

### The New Paradigm

Loop engineering replaces human-in-the-loop prompting with a designed system:

```
Human designs the loop
      |
      v
Loop generates prompts on schedule or trigger
      |
      v
Agents execute (in parallel, in isolated worktrees)
      |
      v
Verification sub-agents check output
      |
      v
Connectors act on results (open PR, update ticket)
      |
      v
Unresolvable items escalate to human inbox
      |
      v
Human reviews diff and escalations only
```

The engineer's job has shifted. Instead of writing prompts, you write the system that writes
prompts. Instead of reviewing every agent output, you review only what the system could not
resolve. The leverage point has moved from instruction-writing to system design.

### When Loop Engineering is Worth It vs Direct Prompting

| Signal | Prefer direct prompting | Prefer loop engineering |
|---|---|---|
| Task frequency | One-off or rare | Recurring (daily, per-commit, per-issue) |
| Task structure | Ambiguous, needs exploration | Well-defined, same steps each time |
| Review latency | Human review needed immediately | Async review is acceptable |
| Parallelism | Single task at a time | Multiple tasks can run concurrently |
| Context stability | Project conventions change often | Conventions are stable and codifiable |
| Cost tolerance | Minimize LLM calls | Willing to pay for unattended execution |
| Error consequence | Mistakes are expensive or irreversible | Mistakes are caught by tests or review |

The practical threshold: if you find yourself issuing the same category of prompt more than three
times a week, it belongs in a loop.

---

## 2. Component 1 — Automations: The Heartbeat

### What Automations Do

An automation is a scheduled or event-triggered invocation that runs without a human typing a
prompt. It performs discovery and triage — reading CI results, scanning open issues, checking
dependency alerts — and produces either an action or an escalation. The human never initiates it;
the loop does.

Claude Code provides two primitives for this:

- `/loop <interval> <prompt>`: runs a prompt on a recurring schedule (e.g., every 15 minutes,
  every hour). Useful for continuous monitoring.
- `/goal <prompt>`: runs the agent until a specified condition is met or the agent judges the goal
  complete. Useful for tasks that should terminate when done rather than recur indefinitely.

### Anatomy of a Well-Designed Automation

Every reliable automation has four stages:

```
Trigger (schedule / git push / CI event)
      |
      v
Context gathering (read CI logs, issues, commits, MEMORY.md)
      |
      v
Decision (classify: actionable / informational / urgent / skip)
      |
      v
Action or Escalate
  |                  |
  v                  v
Act autonomously   Write to MEMORY.md + notify human
(open PR, update    (Slack message, GitHub comment)
ticket, fix lint)
```

### What to Automate First

Start with tasks that have clear inputs and deterministic success criteria:

| Task | Input | Success criterion |
|---|---|---|
| CI failure triage | Failed test logs | Summary written to MEMORY.md; ticket created if new failure |
| Stale PR review | PRs older than N days | Comment posted listing blockers; author notified |
| Test coverage drop | Coverage report diff | Alert if coverage drops more than 2 pp |
| Dependency alerts | `npm audit` / `pip-audit` output | CVE summary created; patch PR opened for minor versions |
| Dead code detection | Static analysis output | Issue created for files with zero test coverage |

### GOAL Prompt Template: CI Failure Triage

```markdown
# CI Triage Goal

Read the last 24 hours of CI failures from `.github/workflows/` logs and the
`ci-failures.json` state file.

For each failure:
1. Classify as: flaky (seen before), regression (new failure), infrastructure (runner issue).
2. For regressions: identify the commit that introduced it using `git log`.
3. Write a structured summary to `MEMORY.md` under the `## CI Failures` section.
4. For critical regressions (main branch, blocking tests): open a GitHub issue via the
   GitHub MCP connector with label `ci-regression`.

Terminate when all failures in the last 24 hours have been classified and recorded.
Do not modify any source files. Do not open PRs.
```

### What Makes an Automation Reliable

| Property | What it means | How to enforce it |
|---|---|---|
| Idempotency | Running twice produces the same result | Check state file before acting; skip if already processed |
| Exit condition | The automation knows when it is done | Every `/goal` must have an explicit termination sentence |
| Escalation path | Automation never silently fails | Unclassified items → MEMORY.md + Slack ping |
| Scope limit | Automation cannot modify production systems | Least-privilege connectors; explicit "do not" clauses in prompt |
| Observability | Human can audit what it did | All actions logged to MEMORY.md with timestamp |

**Why this matters**: Without automations, a loop is just a set of capabilities waiting to be
invoked manually. The heartbeat is what makes the system autonomous — it is the mechanism by
which the loop wakes up, reads the world, and acts without being asked.

---

## 3. Component 2 — Worktrees: Parallel Isolation

### The Problem

Multiple agents sharing a single working directory corrupt each other's work. Agent A edits
`src/auth.py` to fix issue #42; Agent B edits the same file to implement issue #61. Neither
knows about the other. The second agent to write wins; the first agent's work is lost. In the
best case, a merge conflict is detected. In the worst case, one change silently overwrites the
other and all tests still pass.

Locking the working directory serializes agents and eliminates parallelism. Neither outcome is
acceptable for a production loop.

### The Solution: Git Worktrees

`git worktree add` creates a separate working directory linked to the same repository. Each
worktree has its own branch, its own index, and its own file state. Agents can operate in
different worktrees simultaneously without any conflict:

```bash
# Create a worktree for issue #42 on a new branch
git worktree add ../fix-issue-42 -b fix/issue-42

# Create a worktree for issue #61 on another new branch
git worktree add ../feat-issue-61 -b feat/issue-61

# List all active worktrees
git worktree list

# Remove a worktree after the PR is merged
git worktree remove ../fix-issue-42
```

Each directory (`../fix-issue-42`, `../feat-issue-61`) is a fully functional checkout. Commits
made in one worktree do not appear in the other until they are merged through the normal git
workflow.

### Agent Assignment Pattern

The rule is simple: one worktree per task, one agent per worktree.

```
Task queue
  |
  |-- Issue #42 --> git worktree add ../fix-42 -b fix/42 --> Agent A
  |
  |-- Issue #61 --> git worktree add ../feat-61 -b feat/61 --> Agent B
  |
  |-- Issue #77 --> git worktree add ../fix-77 -b fix/77 --> Agent C
```

No agent reads or writes to another agent's worktree. The only shared state is the remote
repository, which is updated only through normal push and PR operations.

### When Worktrees Are Essential vs Overkill

| Situation | Worktrees | Reason |
|---|---|---|
| Two agents editing overlapping files | Essential | Without isolation, conflict is guaranteed |
| Agent and human working simultaneously | Essential | Human edits and agent edits will collide |
| Single agent, sequential tasks | Overkill | One working directory is sufficient |
| Reviewer agent reading only (no edits) | Optional | Read-only access is safe without isolation |
| Testing in a clean environment | Useful | Worktree gives a clean checkout without stashing |

### Integration with Claude Code

When spawning agents via the Agent tool in Claude Code, pass `isolation: "worktree"`. The
harness creates the worktree, runs the agent inside it, and cleans up automatically if no
changes were made:

```python
Agent(
    description="Fix issue #42: null pointer in auth module",
    prompt="...",
    isolation="worktree"
)
```

If the agent makes changes, the worktree path and branch name are returned so a reviewer agent
or connector can open a PR against it.

**Why this matters**: Worktrees are what allow loops to operate at parallelism greater than one.
Without isolation, adding more agents adds more risk rather than more throughput. With worktrees,
the loop can handle ten issues simultaneously with no coordination overhead between agents.

---

## 4. Component 3 — Skills: Embedded Knowledge

### What Skills Are

A skill is a markdown file that codifies project conventions, build procedures, and institutional
knowledge. In Claude Code, these are typically placed in `.claude/commands/` and are loaded
automatically when the agent starts. They are the mechanism by which every agent session begins
with the same baseline knowledge — without requiring the human to explain the project from
scratch each time.

### Why Skills Matter

Without skills, every invocation starts from zero. The agent reads the codebase, tries to infer
the testing framework, discovers the linting rules, and re-derives what a "correct" PR looks
like. This takes tokens, takes time, and produces inconsistent results across sessions.

With skills, that context is pre-written and version-controlled. The agent reads the skill file
and immediately knows: run `make test` not `pytest .`; never edit `src/generated/`; use the
`Result<T, AppError>` pattern for error handling.

### What Belongs in a Skill

| Include | Exclude |
|---|---|
| Build command (`make build`) | Implementation details that change sprint-to-sprint |
| Test command and flags | Secrets, credentials, API keys |
| Lint command and auto-fix command | Things already documented in CLAUDE.md |
| Code style rules (formatter, type checker) | Environment-specific configuration |
| Which directories are off-limits | Transient state (current sprint goal, today's priority) |
| Domain vocabulary definitions | Anything derivable from reading the code |
| PR checklist and review criteria | Business logic that belongs in tests, not prompts |
| Escalation rules ("if X, ask human") | Long prose that is rarely referenced |

### Example Skill File: Python Backend Project

```markdown
# skill: python-backend
# last-verified: 2026-06-01

## Build and Test

```bash
# Install dependencies (always use the project venv)
source .venv/bin/activate

# Run all tests
make test

# Run a single test file
pytest tests/unit/test_auth.py -v

# Lint and auto-fix
ruff check src/ --fix
mypy src/ --strict
```

## Code Conventions

- All public functions must have type annotations.
- Error handling: use `Result[T, AppError]` from `src/core/result.py`.
  Never raise bare exceptions in business logic.
- Database access: use the repository pattern. Controllers must not import
  from `src/db/` directly; use `src/repositories/`.
- Async: the project uses asyncio throughout. All I/O functions must be `async def`.

## Off-Limits Directories

- `src/generated/` — auto-generated from protobuf; never hand-edit.
- `migrations/` — use `alembic revision --autogenerate` only; never write raw SQL migrations.
- `.env` and `.env.*` — never read or write these files.

## PR Checklist

Before opening a PR:
1. `make test` passes with zero failures.
2. `mypy src/ --strict` passes with zero errors.
3. New public functions have docstrings.
4. If you changed a database model, a migration was generated.

## Escalate to Human If

- The fix requires changing the authentication flow.
- You are uncertain whether a change is backward-compatible.
- Tests pass but you cannot explain why the bug occurred.
```

### Skill Discovery

Claude Code picks up all files in `.claude/commands/*.md` automatically. The convention is one
skill per concern: `python-backend.md`, `ci-triage.md`, `pr-review.md`. Skills can also be
invoked explicitly with `/command-name` as slash commands.

### Team Benefit

Skills are checked into the repository. When a new engineer joins, or when a new agent session
starts three months after the last one, the skills file reflects what the team has learned.
Knowledge that was previously in one engineer's head — or in a Slack thread — is now in a
version-controlled file that the agent reads automatically.

**Why this matters**: Skills are the difference between an agent that confidently applies project
conventions and one that guesses. A loop without skills re-derives context on every invocation,
making it slower, less consistent, and more likely to introduce changes that violate conventions
the agent did not know about.

---

## 5. Component 4 — Plugins and Connectors: External Integration

### The Reporting vs Acting Gap

An agent that cannot connect to external systems can only report. It can write "I fixed issue
#123" to a file. A human must then read that file, open GitHub, create the PR, post the comment,
and update the Linear ticket. The agent's output is useful, but the loop is not closed —
the human is still the connector.

MCP (Model Context Protocol) connectors close the loop. With connectors, the agent does not
report that it fixed the issue; it opens the PR, comments on the issue, updates the ticket to
"In Review," and posts a Slack message to the team channel. The human's remaining job is to
review the diff.

### Common MCP Connectors

| Connector | What it unlocks |
|---|---|
| GitHub MCP | Read/write PRs, issues, comments, reviews, branch creation |
| Linear | Read issues, update status, create tasks, assign to milestone |
| Jira | Read epics and stories, update status, log work |
| Slack | Post summaries, request human review, surface escalations |
| Postgres / SQLite | Read schema, run read-only queries, validate migrations |
| GitHub Actions | Trigger workflow runs, read logs, surface test failures |
| Filesystem / S3 | Read build artifacts, upload reports |

### What Connectors Enable in a Loop

The difference is whether the loop is self-closing:

```
Without connectors:
  Agent fixes bug
    --> writes "PR is ready" to MEMORY.md
    --> human reads MEMORY.md
    --> human opens GitHub
    --> human creates PR
    --> human posts Slack message

With connectors:
  Agent fixes bug
    --> GitHub MCP: creates PR
    --> GitHub MCP: requests review from code owners
    --> Slack MCP: posts to #eng-reviews with PR link
    --> Linear MCP: updates issue status to "In Review"
    --> human receives Slack notification and reviews diff
```

### Security Boundary

Connectors should operate with least-privilege tokens. The agent should have access to open PRs
but not merge them. It should be able to post Slack messages but not delete channels. It should
be able to read the production database schema but not execute write queries.

A connector with excessive permissions is a loop with excessive blast radius.

### CLAUDE.md Configuration Snippet for MCP Servers

```markdown
## MCP Connectors

The following MCP servers are available to all agents in this project.

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "slack": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-slack"],
      "env": {
        "SLACK_BOT_TOKEN": "${SLACK_BOT_TOKEN}",
        "SLACK_TEAM_ID": "${SLACK_TEAM_ID}"
      }
    },
    "linear": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-linear"],
      "env": {
        "LINEAR_API_KEY": "${LINEAR_API_KEY}"
      }
    }
  }
}
```

Permissions: agents may open PRs and post comments. Agents may NOT merge PRs
or delete branches. Production database connectors are read-only.
```

**Why this matters**: Connectors are what turn a loop that produces reports into a loop that
produces outcomes. Without them, the agent is a very capable analyst whose findings sit in a
file until a human acts on them. With them, the agent's analysis becomes immediate action and
the human's job is reduced to reviewing what was done, not doing it.

---

## 6. Component 5 — Sub-agents: Distributed Verification

### The Self-Review Problem

The model that wrote the code has strong priors that it is correct. It generated each line
with confidence; it will re-read those lines with confirmation bias. When asked to review its
own implementation, it tends to find what it expects to find: the approach it already chose,
validated. Edge cases that the implementation misses are also cases that the implementation's
author was not thinking about when they wrote it.

This is not a model-specific limitation; it is a structural property of self-review. The same
problem exists in human software development, which is why code review by a second person is
standard practice.

The solution is the same: a separate agent — with different instructions, different context
window, and optionally a different model — performs verification independently.

### Common Sub-agent Patterns

| Pattern | Agent A (ideation) | Agent B (verification) |
|---|---|---|
| Writer / Reviewer | Writes the implementation | Reviews against spec and runs tests |
| Coder / Tester | Writes the feature | Writes tests independently, finds edge cases |
| Planner / Critic | Proposes the approach | Lists failure modes and risks |
| Fast / Slow | Sonnet drafts quickly | Opus reviews carefully |
| Implementer / Security | Implements the feature | Scans for injection, auth bypass, data leaks |

### Spawning Sub-agents in Claude Code

```python
# Step 1: Writer agent in an isolated worktree
writer_result = Agent(
    description="Implement fix for issue #42",
    prompt="""
    Implement the fix described in GitHub issue #42 (null pointer in auth module).
    Read the issue, identify the root cause, write the fix, and run the test suite.
    Do not open a PR. Report the worktree path when done.
    """,
    isolation="worktree",
    model="sonnet"
)

# Step 2: Reviewer agent reads the writer's worktree
reviewer_result = Agent(
    description="Review fix for issue #42",
    prompt=f"""
    Review the code changes in the worktree at {writer_result.worktree_path}.
    Check against the project skill file at .claude/commands/python-backend.md.
    Run the full test suite. Verify the fix actually addresses issue #42.
    Look for edge cases the writer may have missed.
    Report: APPROVED (open PR) or REJECTED (list specific issues).
    """,
    model="opus"
)
```

The key constraint: the reviewer agent must not have access to the writer agent's reasoning
trace. It sees only the resulting code diff, the tests, and the specification. Giving it the
writer's chain-of-thought recreates the same confirmation bias the writer had.

### When Verification Sub-agents Are Worth the Cost

| Situation | Worth it | Reason |
|---|---|---|
| Changes to authentication or authorization | Yes | Mistakes are security vulnerabilities |
| Database schema migrations | Yes | Irreversible; corruption is expensive |
| Public API surface changes | Yes | Breaking changes affect downstream users |
| Internal utility refactors with good test coverage | Marginal | Tests already provide independent verification |
| Documentation updates | No | Low consequence; human review is fast |
| Dependency version bumps (minor) | No | CI provides verification |

**Why this matters**: Sub-agents convert the loop from a system that produces fast output into a
system that produces verified output. Without verification, the loop's speed advantage comes at
the cost of quality. The reviewer agent is not additional ceremony; it is the mechanism by which
the loop earns the right to operate with reduced human supervision.

---

## 7. Component 6 — Persistent State: Memory

### The Amnesia Problem

An agent session ends and everything in its context window disappears. The next invocation
starts with no knowledge of what the previous session found, what was tried, what failed, or
what is still pending. A loop that spans multiple sessions — a nightly triage that feeds into
a morning implementation run — cannot function without persistent state.

### Forms of Persistent State

| Storage | When to use | Access pattern |
|---|---|---|
| MEMORY.md in repo | Cross-session agent context, findings, decisions | Read at loop start; append findings; commit after each run |
| JSON state file | Structured task queue, progress tracking, idempotency keys | Append-only log with current-state snapshot |
| External board (Linear, GitHub Issues) | Human-visible task tracking, stakeholder communication | MCP connector reads/writes; source of truth for task status |
| Git log and blame | What changed, when, why, by whom | `git log --oneline -20`, `git blame`, `git diff` |
| Environment variables | Session-specific config, tokens, feature flags | Set by harness; not written by agents |

### MEMORY.md Pattern

The simplest form of persistent state is a markdown file that the loop reads at the start of
every invocation and writes to at the end:

```markdown
# Agent Memory

## Last Updated
2026-06-23 08:14 UTC — CI triage run

## CI Failures (last 7 days)

| Date | Test | Status | Issue |
|---|---|---|---|
| 2026-06-22 | test_auth_refresh | regression | #142 (opened) |
| 2026-06-21 | test_db_migrate | flaky | tracked, not opened |
| 2026-06-20 | test_auth_refresh | regression | (now #142) |

## Open Tasks

- [ ] Issue #142: auth token refresh null pointer — assigned to fix loop
- [ ] Issue #138: dependency CVE in requests 2.31 — patch PR opened (#144)

## Decisions Made

- 2026-06-20: Decided to pin requests to 2.32.x rather than upgrading to 3.x.
  Reason: 3.x breaks our retry middleware. Revisit in Q3.
```

### Loop Read-Act-Write Pattern

```python
# Pseudocode for a loop invocation that uses MEMORY.md

def run_triage_loop():
    # 1. Read current state
    memory = read_file("MEMORY.md")
    ci_failures = fetch_ci_failures(last_hours=24)

    # 2. Act based on state + new information
    new_findings = classify_failures(ci_failures, known=memory)

    # 3. For actionable items, create tasks
    for finding in new_findings:
        if finding.is_regression and not finding.issue_exists:
            github_mcp.create_issue(finding.to_issue())

    # 4. Write updated state
    updated_memory = merge_memory(memory, new_findings)
    write_file("MEMORY.md", updated_memory)
    git_commit("chore: triage run 2026-06-23 08:14")
```

### What NOT to Store in Memory

- Things derivable from reading the code or git history (the agent can derive them)
- Ephemeral task details that belong in the issue tracker (duplicate source of truth)
- Secrets or credentials (they will be committed and exposed)
- Reasoning traces or agent chain-of-thought (they grow without bound)
- Decisions that will be revisited every session anyway (they create false anchoring)

**Why this matters**: Without persistent state, a loop is stateless between invocations. Each
run re-discovers everything from scratch, re-classifies failures that were already classified,
and has no awareness of decisions made in previous sessions. MEMORY.md is the mechanism by
which the repository accumulates knowledge across time, making each loop invocation smarter
than the last.

---

## 8. Loop Architecture Patterns

### Pattern A: Single-Agent Triage Loop (Starter)

One agent, on a schedule, reads the environment and writes findings. No code changes, no PRs,
no connectors required. This is the right starting point: it delivers value immediately and
teaches you what to automate next.

```
Schedule trigger (e.g., 08:00 daily)
          |
          v
    Triage Agent
    reads: CI logs, open issues, git log, MEMORY.md
          |
     classify each item
          |
    +-----------+-----------+
    |           |           |
    v           v           v
informational  actionable  urgent
    |           |           |
write to     write to    write to
MEMORY.md   MEMORY.md   MEMORY.md
                         + Slack ping
                           to human
```

Skills used: `ci-triage.md`
Connectors: Slack (for escalation only)
Output: updated MEMORY.md, occasional Slack messages

---

### Pattern B: Writer and Reviewer Loop (Intermediate)

An automation triggers on a new issue. One agent implements a fix in an isolated worktree; a
separate reviewer agent checks the work. A connector opens the PR. The human reviews only the
diff.

```
New GitHub issue created
          |
          v
    Triage Agent
    classifies as: auto-fixable
          |
          v
    Writer Agent (worktree: fix/issue-N)
    reads issue, implements fix, runs tests
          |
     tests pass?
     No --> write to MEMORY.md, escalate to human
     Yes --> report worktree path
          |
          v
    Reviewer Agent (reads worktree, different context)
    checks against skill file, runs tests independently
          |
     approved?
     No --> write review comments, escalate to human
     Yes --> signal connector
          |
          v
    GitHub MCP Connector
    opens PR, requests review from code owners,
    posts Slack notification
          |
          v
    Human reviews diff
    (prompted by Slack, not by checking manually)
```

Skills used: `python-backend.md`, `pr-review.md`
Connectors: GitHub MCP, Slack MCP
Worktrees: one per issue

---

### Pattern C: Full Autonomous Engineering Loop (Advanced)

The most capable pattern. A triage agent surfaces a prioritised backlog. A planner proposes an
approach. A coder implements. A test agent writes tests independently. A reviewer checks both.
A connector opens the PR. The human reviews only the final diff.

```
Nightly triage agent
reads: CI, issues, commits, MEMORY.md
writes: prioritised backlog to MEMORY.md
          |
          v
Morning planning agent
reads: backlog, codebase
proposes: approach + effort estimate for top 3 items
creates: Linear tasks with sub-tasks
          |
          v (for each approved task)
    Coder Agent (worktree: feat/task-N)
    implements feature or fix
          |
          +----------------------------------+
          |                                  |
          v                                  v
    Test Agent                         Reviewer Agent
    writes tests independently         checks implementation
    finds edge cases coder missed      against spec and skill file
          |                                  |
          +----------------------------------+
                      |
               both pass?
               No --> escalate to human with specific blockers
               Yes --> connector
                      |
                      v
              GitHub MCP Connector
              opens PR with:
              - implementation diff
              - test diff
              - reviewer comments
              - Linear task link
              - explanation written by coder agent
                      |
                      v
              Human reviews diff
              (only the diff; all context is in the PR)
```

Note on human involvement: the human is not eliminated. They review PRs, handle escalations,
and make judgment calls the loop cannot make. What the loop eliminates is the overhead of
initiating, running, and coordinating each task. The human's contribution shifts from mechanical
execution to genuine review.

---

## 9. Getting Started — Implementation Roadmap

The order below is ranked by value delivered per unit of complexity. Each step builds on the
previous one but delivers value independently.

| Step | Action | Value | Complexity |
|---|---|---|---|
| 1 | Write your first SKILL.md | Eliminate repeated context-setting in every session | Low |
| 2 | Set up a daily `/goal` automation for CI triage | Passive visibility without manual effort | Low |
| 3 | Add MEMORY.md state management | Cross-session coherence; loop accumulates knowledge | Low |
| 4 | Add a GitHub MCP connector | Agent can open PRs and comment on issues | Medium |
| 5 | Introduce a reviewer sub-agent | Catches oversights before PRs reach human review | Medium |
| 6 | Add worktrees for parallel tasks | Concurrent execution; multiple issues at once | Medium |
| 7 | Connect to issue tracker (Linear / Jira) | Full task lifecycle: read, implement, close | Medium |
| 8 | Build a Writer + Reviewer loop | Significant automation; human reviews diffs only | High |

### Concrete Templates

**Step 1: Minimal SKILL.md**

```markdown
# skill: project-conventions
# last-verified: 2026-06-23

## Commands
- Build: `make build`
- Test: `make test`
- Lint: `make lint` (auto-fixes on save)

## Off-Limits
- `src/generated/` — never edit manually
- Any file ending in `_pb2.py` — protobuf generated

## PR Requirements
- All tests pass
- No new mypy errors
- New functions have docstrings
```

**Step 2: Daily CI Triage /goal Prompt**

```markdown
Read CI failure logs from the last 24 hours (check .github/workflows/ run logs
via the GitHub MCP connector). For each failure, classify as: regression, flaky,
or infrastructure. Record findings in MEMORY.md under ## CI Failures with today's
date. If any regressions are found on the main branch, post a summary to the
#eng-alerts Slack channel. Stop when all failures are classified and recorded.
Do not modify source files.
```

**Step 3: Agent Spawn Pattern for Writer + Reviewer**

```python
# In a Claude Code automation or hook:

writer = Agent(
    description=f"Implement fix for issue #{issue_id}",
    prompt=f"""
    Read GitHub issue #{issue_id}. Implement the fix.
    Follow the skill file at .claude/commands/project-conventions.md.
    Run `make test`. If tests fail, debug and fix.
    Do not open a PR. Report the branch name when done.
    """,
    isolation="worktree",
    model="sonnet"
)

reviewer = Agent(
    description=f"Review fix for issue #{issue_id}",
    prompt=f"""
    Review the changes on branch {writer.branch}.
    Verify the fix addresses issue #{issue_id}.
    Check against .claude/commands/project-conventions.md.
    Run `make test` independently.
    Output: APPROVED or REJECTED with specific reasons.
    """,
    model="opus"
)

if reviewer.output == "APPROVED":
    github_mcp.create_pull_request(branch=writer.branch, issue=issue_id)
```

---

## 10. Critical Limitations — The Three Debts

### Verification Debt

The loop's "done" is a claim. Tests passing in the agent's worktree is evidence, not proof.
The tests may not cover the case that was broken. The reviewer sub-agent may have the same
blind spots as the writer. The loop can be confidently wrong.

Mitigations:
- Require a human diff review before any merge to a protected branch. Never configure a loop
  to auto-merge to main.
- Require staging deployment and smoke tests before the PR can be approved.
- Treat reviewer sub-agent approval as a necessary but not sufficient condition for merge.
- Log the loop's "done" claims alongside test results so patterns of overconfidence are visible.

The rule: a loop can open a PR. A loop cannot merge a PR.

### Comprehension Debt

Loop-generated code ships faster than engineers can absorb it. If the loop opens five PRs today
and you approve them all in twenty minutes of diff review, you have shipped code you understand
at diff granularity, not at architecture granularity.

Osmani calls this "comprehension debt" — the gap between what has shipped and what you can
explain. It is analogous to technical debt: it accumulates silently, it compounds, and it
becomes expensive when you need to debug something the loop built six months ago.

Mitigations:
- Require the writer agent to produce a short explanation of every change (not just a commit
  message, but a one-paragraph description of why the approach was chosen).
- Review explanations alongside diffs, not just the code.
- Set a comprehension budget: 30 minutes per day reading loop output in depth.
- Run occasional "explain this codebase section to me" sessions to calibrate how well you
  understand what the loop has built.
- Write the weekly summary yourself, from memory, before reading the loop's output. The gap
  between your summary and the loop's summary is your comprehension debt.

### Cognitive Surrender

A well-designed loop makes it very easy to approve work without thinking. The Slack notification
arrives, you open the PR, the tests are green, the reviewer sub-agent says APPROVED, and you
click merge. You did not need to understand the change to approve it.

This is the most dangerous failure mode because it is entirely invisible. The loop continues to
function. Code continues to ship. But the engineer has become a merge button, not an engineer.

Signs of cognitive surrender:
- You approve PRs without reading the diff.
- You cannot explain what the loop changed last week without reading MEMORY.md.
- You do not know which parts of the codebase the loop has modified significantly.
- When a production issue appears, you have no intuition about where to look.

Mitigations:
- Set a personal rule: no approval without being able to state the change in one sentence.
- Write a weekly retrospective on loop output before reading the loop's own summary.
- Ask the loop to quiz you on what it changed ("ask me three questions about the changes
  you made this week; tell me if my answers are correct").
- Periodically work on a task manually, without the loop, to maintain hands-on skill.

Osmani's framing: "Build the loop. But build it like someone who intends to stay the engineer,
not just the person who presses go."

---

## 11. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| No exit condition | Loop runs forever; costs money; fills disk with logs | Every `/goal` prompt must contain an explicit termination sentence |
| Agents editing the same files | Merge conflicts; silent overwrites; lost changes | One worktree per agent; enforce before spawning |
| Skills become stale | Agent follows outdated convention; breaks build | Include "last-verified" date; review skills when conventions change |
| Over-trusted connectors | Agent merges bad code; sends premature Slack messages | Least-privilege tokens; require human approval on all merges |
| Self-review | Writer and reviewer have the same context; reviewer confirms writer's assumptions | Separate invocations; reviewer reads only the diff, not the writer's reasoning |
| Memory bloat | MEMORY.md grows to thousands of lines; agent cannot read it fully | Prune periodically; archive old findings; store only non-derivable facts |
| Comprehension debt | You cannot explain recent changes; production bugs are mysterious | Mandatory loop output review; comprehension budget; manual tasks to maintain skill |
| Automation without observability | Loop acts; you do not know what it did | Every action logged to MEMORY.md with timestamp and rationale |
| No escalation path | Loop encounters ambiguous situation; silently skips or makes a wrong decision | Every automation prompt must include explicit escalation instructions |
| Over-automation too early | Loop built before conventions are stable; skills need weekly updates | Start with Skills + triage loop; add connectors only when conventions are settled |

---

## 12. References

- Addy Osmani, "Loop Engineering": https://addyosmani.com/blog/loop-engineering/
- Claude Code Documentation — automations, `/loop`, `/goal`, worktrees:
  https://docs.anthropic.com/claude-code
- Model Context Protocol (MCP) — specification and server catalog:
  https://modelcontextprotocol.io
- Git Worktrees — official documentation:
  https://git-scm.com/docs/git-worktree
- Boris Cherny (head of Claude Code, Anthropic) on agentic development:
  referenced in Osmani (2025), "Loop Engineering"
- Peter Steinberger on moving from prompting to loop design:
  referenced in Osmani (2025), "Loop Engineering"
