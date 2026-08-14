---
title: "Coding Agents, Harness Engineering, and Graph Engineering — Designing What Surrounds the Model"
description: "A survey of the 2026 shift from prompt engineering to harness and graph engineering: the ETCLOVG taxonomy, guides and sensors, code-graph retrieval, and agent organizations as graphs."
---

# Coding Agents, Harness Engineering, and Graph Engineering — Designing What Surrounds the Model

> A survey of the discipline that emerged in 2026 around the claim `Agent = Model + Harness`: what a harness is, how it is layered, why swapping one moves benchmark scores as much as a model upgrade, and how graphs became the organizing structure for both code context and multi-agent work. For runnable simulations of every quantitative claim made here, see [harness_and_graph_engineering.ipynb](harness_and_graph_engineering.ipynb).

For three years the lever on agent behavior was the prompt. That stopped being true once agents began running for hours, calling dozens of tools, and editing repositories they could not fit in context. The binding constraint moved to the *software surrounding* the model — sandboxes, tool schemas, context policy, verifiers, permissions — and the empirical evidence for that shift is blunt: holding model weights fixed and changing only the harness moves SWE-bench pass rates by 10 to 20 percentage points, and in pathological cases by nearly 60. A model upgrade often buys less. This document maps the resulting discipline: the seven-layer ETCLOVG taxonomy from the 2026 harness survey, Böckeler's guides-and-sensors control framing, the three cross-layer tensions that make local optimization fragile, and the two distinct things "graph engineering" now means — code graphs as retrieval substrate, and work graphs as multi-agent structure. It is the layer above [loop_engineering.md](loop_engineering.md) and [loop_design_playbook.md](loop_design_playbook.md), which cover the single-agent loop this builds on.

---

## Table of Contents
1. [Three Eras: Prompt, Context, Harness](#1-three-eras-prompt-context-harness)
2. [Agent = Model + Harness](#2-agent--model--harness)
3. [The ETCLOVG Taxonomy](#3-the-etclovg-taxonomy)
4. [Guides and Sensors: The Control View](#4-guides-and-sensors-the-control-view)
5. [Three Cross-Layer Tensions](#5-three-cross-layer-tensions)
6. [Graph Engineering I: Code Graphs as Context](#6-graph-engineering-i-code-graphs-as-context)
7. [Graph Engineering II: Agent Organizations as Graphs](#7-graph-engineering-ii-agent-organizations-as-graphs)
8. [A Design Checklist, Grounded in a Real Harness](#8-a-design-checklist-grounded-in-a-real-harness)
9. [Common Pitfalls](#9-common-pitfalls)
10. [References](#10-references)

---

## 1. Three Eras: Prompt, Context, Harness

The harness survey frames the field as three successively broader units of design. Each era did not replace the previous one; it enclosed it.

| Era | Period | Unit of design | What you tune | Failure you are fighting |
|---|---|---|---|---|
| Prompt engineering | 2022–2024 | A single model call | Wording, examples, output format | The model misunderstands the request |
| Context engineering | 2024–2025 | What the model can see | Retrieval, compaction, memory, tool descriptions | The model cannot see what it needs, or drowns in what it does not |
| Harness engineering | 2026– | The closed loop around the model | Sandbox, tool contracts, lifecycle, verifiers, permissions, observability | The *system* fails even when each model call is reasonable |
| Graph engineering | mid-2026– | The structure of many agents and many artifacts | Org graphs, work graphs, code graphs | The workflow was never actually modeled |

The progression tracks how long agents run. A prompt is adequate when the interaction is one turn. Context engineering becomes necessary at tens of turns, when the transcript itself is the scarce resource. Harness engineering becomes necessary at hundreds of turns, when the agent's environment — not its reasoning — determines whether work survives.

### Key insight
> **The scope expands because the failure mode moves.** Prompt-era failures were comprehension failures, fixable in the text. Harness-era failures are systems failures: a sandbox that lacks a package, a tool description that silently consumes 8k tokens, a verifier that passes broken code. No amount of prompt rewriting reaches them.

### Why this matters

Recognizing which era a problem lives in tells you where to spend effort. If an agent repeatedly writes code that fails your CI, that is not a prompting problem — it is a missing sensor. Misdiagnosing the layer is the most common way teams spend weeks rewriting instructions that were never the constraint.

---

## 2. Agent = Model + Harness

The formula that anchors the discipline is deliberately crude:

$$
\text{Agent} = \text{Model} + \text{Harness}
$$

A harness is the software infrastructure surrounding a language model that lets it operate as an agent: tool dispatch, memory and state persistence, an isolated workspace, context management, and guardrails such as scoped permissions and approval tiers. The motivation is that models are stateless text producers. Everything that makes an agent *act* — across steps, across tools, across sessions — is harness.

The vocabulary stabilized in early 2026, with contested attribution: Mitchell Hashimoto's February 2026 post on engineering permanent fixes into agent environments, Vivek Trivedy's "Anatomy of an Agent Harness" at LangChain deriving components from the formula, and subsequent writing from OpenAI, Thoughtworks, and Anthropic. By mid-2026 it had become an academic subject.

### The empirical claim: harness swaps rival model upgrades

The reason this is a discipline rather than a vocabulary preference is measurement. Holding weights constant and varying only the surrounding software produces score movements comparable to changing models:

| Evidence | Model | Harness variation | Result |
|---|---|---|---|
| SWE-bench Pro, vendor comparison | Claude Opus 4.5 | SEAL (standardized) → Claude Code → Cursor → Auggie | 45.9% → 49.8% → 50.2% → 51.8% (about 6 pp spread) |
| Cross-harness study | GLM-5.2 | Multiple harnesses | 23% → 52% pass@1 (29 pp) |
| Cross-harness study | Gemma 4 26B | Multiple harnesses | 15% → 36% (21 pp) |
| Native vs foreign harness | OpenSWE-32B | OpenHands (native) → mini-swe-agent → Kimi-CLI | 62.4% → 54.9% → 3.6% |
| Scaffold study, SWE-bench Verified | one model | scaffold choice only | 62.3% → 70.2% (7.9 pp) |
| Longitudinal | model held constant | 35 sequential scaffolding releases | swings attributed to scaffolding, not the model |

The OpenSWE-32B row is the instructive one. A 58.8-point collapse from moving a model off its native harness is not a small effect on a small base — it is the difference between a usable agent and a broken one, produced entirely by software the model never sees.

### Key insight
> **A benchmark score is a property of a model-harness pair, not of a model.** The survey states this precisely: agent scores cannot be cleanly attributed to the model without specifying the surrounding controller. Under a closed-loop framing, changing context policy, tool schema, verifier, or recovery loop changes the controller and therefore the measured behavior of the *same* model. Reported deltas between agents that disclose no harness details are close to uninterpretable.

### Why this matters

Two practical consequences. When evaluating models, hold the harness fixed or you are measuring noise. When your agent underperforms, the prior should be that the harness is at fault, because the harness is what you actually control — and the measured leverage there is 10 to 20 points.

---

## 3. The ETCLOVG Taxonomy

The 2026 survey *Agent Harness Engineering: A Survey* (Li, Xiao, Zhang, Liu et al.; CMU, UAB, Tulane, Yale, NEU, Stanford, Amazon, UChicago, Virginia Tech, Rutgers) introduces a seven-layer taxonomy and maps more than 170 open-source projects onto it. The acronym is **ETCLOVG**.

Four layers form the structural core:

| Layer | Question it answers | Representative concerns |
|---|---|---|
| **E** — Execution environment & sandbox | Where does agent code run, and what bounds it? | Managed sandboxes, code-specialized sandboxes (Judge0-style), OS-level permission sandboxes, computer-use infrastructure, threat model and sandbox escape |
| **T** — Tool interface & protocol | How are external capabilities described, discovered, invoked? | MCP / ACP / A2A protocols, tool description and selection, tool-augmented training, session scalability |
| **C** — Context & memory management | What can the model see, over what horizon? | Active window management, session state and cross-run persistence, long-term memory, 100+ turn coherence, context drift |
| **L** — Lifecycle & orchestration | What control flow reads and writes that state? | Lifecycle state machines, the single-agent inner loop, multi-agent patterns, full issue-to-pull-request pipelines |

Three layers form the control plane around that core:

| Layer | Question it answers | Representative concerns |
|---|---|---|
| **O** — Observability & operations | What actually happened, and what did it cost? | Tracing platforms (Langfuse, Arize Phoenix, OpenLLMetry), OpenTelemetry instrumentation, cost attribution, anomaly detection, reliability engineering |
| **V** — Verification & evaluation | Did it work, and how do we know? | Benchmark grounding, pre-execution readiness validation, trace capture, multi-level judgement and failure attribution, continuous regression |
| **G** — Governance & security | What is it allowed to do, and can we prove what it did? | Permission models and identity, lifecycle hooks, component hardening, declarative constitutions, audit infrastructure, human approval |

Three design choices distinguish this from earlier component models. It classifies by *engineering surface* rather than minimal runtime role, so it can host a broad ecosystem. It promotes **Observability to an independent layer** rather than a side effect of hooks, because production observability has its own tooling ecosystem and practices. And it makes **Governance first-class**, spanning model-level defenses, system-level permission and gateway controls, and organization-level audit and approval.

### What the ecosystem mapping shows

Mapping 170+ projects reveals uneven coverage, and the unevenness is informative:

- **Dense**: E, T, L, V. Coding, web, terminal, and computer-use agents all need runnable environments, tool contracts, control loops, and repeatable evaluation before they are useful at all.
- **Embedded rather than standalone**: C. Context and memory appear everywhere but usually inside larger frameworks, not as reusable components.
- **Thin in open source**: O and G. Operational control shows up mainly through commercial platforms, SDK features, and engineering writeups — it matured later than runtime and benchmark infrastructure.

### Key insight
> **A harness design is a dependency structure, not a checklist.** The layers constrain each other: E determines which L strategies are practical; C affects V reproducibility; G imposes identity and audit constraints spanning every other layer. Most production failures occur at the *interfaces* between layers, which is exactly where a checklist has nothing to say.

### Why this matters

ETCLOVG is useful as an audit instrument. Walk your own setup layer by layer and the gaps become visible — most teams discover they have E, T, and L (they had to) and almost nothing in O, V, and G, which mirrors the ecosystem-wide gap. That is also the order in which unattended agents cause damage.

---

## 4. Guides and Sensors: The Control View

Böckeler's framing (Thoughtworks, published on martinfowler.com) supplies the control-theoretic vocabulary the taxonomy lacks. Two orthogonal splits do most of the work.

**Split 1: who built it.** Three concentric layers — the **model** at the core, the **builder harness** shipped by the agent's developers (system prompts, orchestration, retrieval), and the **user harness** a team assembles on top (instruction files, MCP servers, custom skills). You cannot change the inner layers; the outer one is your entire design surface.

**Split 2: when it acts.**

$$
\underbrace{\text{Guides}}_{\text{feedforward}} \;\longrightarrow\; \text{agent acts} \;\longrightarrow\; \underbrace{\text{Sensors}}_{\text{feedback}}
$$

- **Guides** anticipate behavior and steer *before* the agent acts. They raise the probability of a good result on the first attempt.
- **Sensors** observe *after* the agent acts and let it self-correct. They are "particularly powerful when they produce signals optimised for LLM consumption" — an error message written for a model, not a human.

Each may be **computational** (deterministic, CPU, milliseconds to seconds, highly reliable: tests, linters, type checkers) or **inferential** (semantic, model-based, slower, non-deterministic: LLM-as-judge, review agents).

| Control | Kind | Example |
|---|---|---|
| Guide | Computational | Language Server Protocol integration; bootstrap scripts; codemods (OpenRewrite) |
| Guide | Inferential | Architecture decision rules in `AGENTS.md` / `CLAUDE.md` |
| Sensor | Computational | ESLint / Semgrep; ArchUnit structural tests; coverage and mutation testing |
| Sensor | Inferential | Code review agent skill; semantic redundancy detection |

### Formalizing the guide-sensor tradeoff

Let $p$ be first-attempt success probability (what guides raise), $d$ the probability a sensor detects a failure that occurred (what sensors provide), and $k$ the retry budget. Each detected failure yields another attempt with success probability $p$, so the probability of eventually delivering correct work is a geometric series:

$$
P_{\text{success}}(p, d, k) \;=\; p \sum_{i=0}^{k-1} \big((1-p)\,d\big)^i \;=\; p \cdot \frac{1 - \big((1-p)d\big)^{k}}{1 - (1-p)d}
$$

The **escape rate** — wrong work delivered as done — is $1 - P_{\text{success}}$. Two limits explain Böckeler's anti-pattern:

- $d = 0$ (feedforward only): $P = p$. The agent encodes rules but never learns whether they worked. No retry can help because nothing detects failure.
- $p$ low, $d$ high (feedback only): $P \to 1$ as $k$ grows, but expected attempts $\approx 1/p$ and cost scales with it. The agent keeps repeating the same mistakes and pays for every one.

$$
\mathbb{E}[\text{attempts}] = \sum_{i=0}^{k-1}\big((1-p)d\big)^i \qquad\Longrightarrow\qquad C_{\text{total}} \approx \mathbb{E}[\text{attempts}]\cdot\big(c_{\text{model}} + c_{\text{sensor}}\big)
$$

This makes the allocation question concrete: with a fixed budget, does a marginal unit buy more $p$ (guides) or more $d$ (sensors)? The notebook solves it under concave returns and finds a clean monotone answer — the optimal share spent on guides falls as the retry budget rises:

| Retry budget $k$ | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Optimal share on guides | 100% | 75% | 24% | 6% |

With a single attempt, detection is worthless because nothing can act on it, so guides take everything. As $k$ grows, each unit of detection converts into several extra chances and sensors dominate. One nuance worth noting: in the same simulation, maximizing $P_{\text{success}}$ favors sensors while maximizing $P_{\text{success}}/\text{cost}$ favors guides, because sensors bill on every attempt and guides are paid for once. Reliability and efficiency pull in opposite directions here, which is the trilemma of §5.1 appearing at the smallest possible scale.

### Harnessability

**Harnessability** is "the degree to which a codebase is amenable to harness construction." Strongly-typed languages, definable module boundaries, and established frameworks afford good harnessing; the related notion of *ambient affordances* covers structural properties that make an environment legible and navigable to agents.

The uncomfortable corollary is stated plainly in the source: legacy teams with heavy technical debt face the harder problem, because **the harness is most needed where it is hardest to build.** A well-typed greenfield service gets a type checker as a free sensor; a 300k-line untyped monolith gets nothing for free.

### Key insight
> **Neither control alone works.** "You get either an agent that keeps repeating the same mistakes (feedback-only) or an agent that encodes rules but never finds out whether they worked (feed-forward-only)." A harness is a control loop, and a control loop needs both a setpoint and a measurement.

### Why this matters

This framing tells you what to build next. If your agent's failures are *novel* each time, you need guides. If they *repeat*, you need sensors. And if your sensors never fire, that is genuinely ambiguous — either quality is high or detection is inadequate — which is why Böckeler flags coverage ambiguity as an open problem rather than a solved one.

---

## 5. Three Cross-Layer Tensions

The survey's synthesis section names three effects that appear only after the layers are composed. These are the load-bearing claims for anyone designing a harness.

### 5.1 The cost-quality-speed trilemma

Stronger sandboxes and more faithful environments improve safety and reproducibility but increase startup latency and infrastructure cost. Richer context and memory policies improve task continuity but consume tokens and add retrieval overhead. Deeper evaluation and observability improve diagnosis but slow iteration and add storage, labeling, and trace-processing cost.

$$
\text{quality} \uparrow \;\Longrightarrow\; \text{cost} \uparrow,\ \text{latency} \uparrow
$$

The consequence is that quality cannot be treated as a scalar objective. Production systems must decide which checks run **synchronously**, which run **offline or in regression suites**, and which failures justify expensive recovery paths. Böckeler's temporal positioning — "keep quality left" — is the same decision expressed as scheduling: fast computational controls pre-commit, expensive sensors (mutation testing, comprehensive review) post-integration, drift detection continuous.

### 5.2 The capability-control tradeoff

More capable harnesses expose more authority, and every increase in authority expands the control problem:

| Capability added | Control problem created |
|---|---|
| Larger tool menu | More selection error; larger prompt-injection surface |
| Persistent memory | Provenance, staleness, and privacy risk |
| Permissive sandbox | Larger blast radius for misaligned or compromised actions |

This tradeoff belongs in core system design, not in a security review appended afterwards, because it links tool schemas, context policy, runtime permissions, identity, auditability, and human approval simultaneously.

### 5.3 The harness coupling problem

The most consequential claim: **harness layers are coupled such that local optimization is fragile.**

- The execution environment changes evaluation results through package availability, reset semantics, latency, and failure modes.
- Tool descriptions consume context budget and shape model behavior.
- Observability traces become governance evidence only if identity and permission state are captured at the same granularity.
- Evaluation design feeds back into orchestration by rewarding some recovery loops and penalizing others.

$$
\Delta_{\text{system}}(A \cup B) \;\neq\; \Delta(A) + \Delta(B)
$$

A prompt, tool, memory policy, sandbox, verifier, or monitor may look beneficial in isolation while degrading the whole rollout in combination. The operational rule that follows: **harness changes must be tested as system changes**, on the full loop, not as isolated A/B improvements to one layer.

### Key insight
> **The coupling problem is why "we improved our retrieval by 15%" is not a claim about agent quality.** Better retrieval that consumes more context can reduce end-to-end success by crowding out the transcript the model needs. The notebook constructs a concrete two-factor case where each intervention helps alone and the combination is worse than either.

### Why this matters

These three tensions predict most disappointment in agent projects. Teams add verification, observability, and governance expecting quality with no cost (trilemma), grant broad tool access expecting capability with no risk (capability-control), and validate components in isolation expecting the gains to add up (coupling). All three expectations are wrong in a specific, predictable direction.

---

## 6. Graph Engineering I: Code Graphs as Context

"Graph engineering" names two different practices. The first is a **context** technique: represent the repository as a graph and retrieve over its structure instead of over flat text.

The problem it addresses is stated bluntly by the Codebase-Memory authors: LLM coding agents "typically explore codebases through repeated file-reading and grep-searching, consuming thousands of tokens per query without structural understanding." Lexical search cannot see imports, call chains, type hierarchies, or code-test links — precisely the relations that determine which files a change must touch.

### The representation

A code graph is a directed heterogeneous graph: nodes are files, classes, and functions; edges are imports, invocations, inheritance, and test links. Given anchor nodes $A$ (usually lexical matches), retrieval returns a confidence-filtered neighborhood:

$$
N_r(A) \;=\; \{\, v \in V \;:\; \exists\, a \in A,\ \operatorname{dist}(a, v) \le r \,\}
$$

The design tension is immediate. Increasing $r$ raises recall of the truly relevant set but grows the neighborhood roughly geometrically in the mean out-degree $\bar{k}$:

$$
|N_r(A)| \;\approx\; |A| \cdot \frac{\bar{k}^{\,r+1} - 1}{\bar{k} - 1}
$$

so token cost explodes while marginal recall shrinks. Every system below is a different answer to "where do you stop expanding?"

### Systems and measured results

| System | Representation | Interface | Measured result |
|---|---|---|---|
| **LocAgent** (2025) | Directed heterogeneous graph over files/classes/functions with import, invocation, inheritance edges | Graph-guided agent tools | Up to **92.7% file-level localization accuracy**; **+12% issue-resolution success at Pass@10**; fine-tuned Qwen-2.5-Coder-32B reaches parity with proprietary SOTA at about **86% lower cost** |
| **LARGER** (2026) | Repository graph, entered from lexical anchors | Inside the agent's *existing* search loop — no external graph DB, no separate traversal stage | **+13.9 Acc@5** on LocBench tuned, **+11.8** with fixed hyperparameters, over the strongest baseline; consistent gains on MuLocBench, SWE-Atlas Test Writing, SWE-Atlas Codebase QA |
| **Codebase-Memory** (2026) | Persistent Tree-sitter knowledge graph, 66 languages, with call-graph traversal, impact analysis, community discovery | MCP server, auto-detected by 10 coding agents | **10x fewer tokens** and **2.1x fewer tool calls** than a file-exploration agent across 31 repositories; answer quality **83% vs 92%**; matches or beats the explorer on graph-native queries (hub detection, caller ranking) |
| **RepoGraph** | Repository graph | Subgraph retrieval for structured context | Structured context in place of flat file dumps |
| **CodexGraph** | Repository indexed into Neo4j | Agent writes Cypher queries | Query-language access to structure |

### Key insight
> **The Codebase-Memory numbers are the honest summary of this technique: 10x cheaper, slightly worse.** 83% versus 92% answer quality at one tenth the tokens is not a free win — it is a deliberate trade, excellent when you are budget-bound or running many queries, wrong when a single answer must be right. Graph-native queries ("who calls this?", "what are the hubs?") are where the graph is strictly better, because they are questions flat search cannot express at all.

### The architectural lesson from LARGER

LARGER's contribution is less the graph than *where the graph lives*. Earlier approaches "require separate graph tools or traversal stages that fragment the agent's interaction loop." LARGER integrates into existing CLI coding agents with no external graph database and no specialized interface: start from lexical matches, align them to graph anchors, expand locally with confidence filtering — all inside the search loop the agent already has.

This is a harness-engineering result restated in graph terms. A retrieval improvement that adds a tool, a service, and a traversal stage pays a coupling tax (more tools to select among, more context spent on descriptions, more failure modes) that can exceed the retrieval gain. Improvements that fit inside an existing loop avoid that tax.

### Why this matters

For a working ML engineer this is the most immediately actionable section. Reading a 35k-line unfamiliar repository is a graph problem — "who calls this function", "what breaks if I change this signature" — and lexical search answers neither. The measured 10x token reduction is the difference between an agent that can hold a large repository's structure in a working budget and one that cannot.

---

## 7. Graph Engineering II: Agent Organizations as Graphs

The second meaning of graph engineering targets **orchestration**: model the multi-agent system itself as an explicit graph rather than as a loop.

The framing that captures the shift: *"Loops made agent behavior programmable. Graphs make agent organizations programmable."* And the reason it is uncomfortable, from Luis Catacora: *"Loops are forgiving. Graphs force you to admit how much of the workflow you haven't actually modeled yet."* A loop defers structural decisions to runtime; a graph requires declaring structure, ownership, and dependencies up front.

### Two graphs, different lifetimes

| | **Org graph** (structural) | **Work graph** (dynamic) |
|---|---|---|
| Lifetime | Stable, long-lived | Ephemeral, regenerated per task |
| Nodes | Named roles with preserved memory and permanent zone ownership | Task nodes |
| Edges | Reporting and escalation paths | Data and ordering dependencies |
| Mutability | Rarely changes | "Can split, merge, reorder, or disappear as evidence arrives" |
| Analogy | An org chart | A project plan for one job |

Keeping these separate is the main design discipline. Persistent identity and memory belong to the org graph (a security agent that has accumulated context about your threat model); per-task decomposition belongs to the work graph (this issue needs three parallel investigations, then a merge).

### Patterns

| Pattern | Structure | Reported result |
|---|---|---|
| **Advisor-Orchestrator** | A strong orchestrator model coordinating multiple cheaper workers | About **92% of solo-strong-model quality at about 63% of the price** on SWE-bench Pro |
| **Zone Defense** | Agents own stable domains (security, data, API, frontend) with persistent context | Specialization with warm context per zone |
| **Council Deliberation** | Many personas deliberating through anti-groupthink gates | Diversity of critique at high token cost |

### Why explicit edges are the point

The named failure modes are all edge failures:

- **Node failure propagating downstream** without isolation — no edge-level error boundary.
- **Context leakage across boundaries** when edges are not explicitly defined — agents seeing state they were never granted.
- **State inconsistency** across the work graph during dynamic restructuring.

A graph makes each of these addressable because each has a location. In a loop, "the agent saw something it should not have" has no locus to fix.

### The scheduling bound

Making the work graph explicit also makes its performance ceiling computable. For a DAG with total work $W$ and critical path length $C$, any schedule obeys

$$
T_{\text{makespan}} \;\ge\; \max\!\left(C,\ \frac{W}{P}\right), \qquad S_{\max} = \frac{W}{C}
$$

for $P$ parallel agents. Fan-out cannot beat the longest dependency chain, so a work graph that is essentially a chain gains nothing from ten agents — and you can see that before paying for them. The notebook computes $W/C$ for realistic issue-to-PR graphs; the speedup ceiling is usually far below the agent count, which is the quantitative reason most multi-agent setups disappoint.

### Key insight
> **Graph engineering's real product is the admission of ignorance.** Its value is less in parallel speedup than in forcing you to write down ownership, dependencies, and boundaries that a loop lets you leave implicit. The cases where declaring the graph reveals that the workflow is a chain are successes, not failures — you learned it without buying the fan-out.

### Why this matters

This is the layer directly above [loop_engineering.md](loop_engineering.md). If your single-agent loop works and you are considering multi-agent, the graph view supplies the two prerequisite questions: what is the critical path (is there any speedup available?), and what are the edges (can a failure or a secret cross a boundary?). Both are answerable on paper.

---

## 8. A Design Checklist, Grounded in a Real Harness

ETCLOVG is abstract until mapped onto something concrete. Below, each layer is instantiated against a production coding-agent harness, with the *user-harness* levers marked — those are the only ones a user controls.

| Layer | Inner harness (shipped) | Outer harness (yours) |
|---|---|---|
| **E** Execution | Sandboxed bash, permission modes, isolated worktrees | Which directories are reachable; whether sandboxing is on; devcontainer or Nix definition for a reproducible environment |
| **T** Tooling | Built-in file, search, edit, bash tools | MCP servers; which tool families are allowed; scoped `Bash(cmd *)` permissions |
| **C** Context | Auto-compaction, context visualization, session persistence | Instruction files (`CLAUDE.md`); memory files; skill descriptions (a standing token cost); what you paste versus what you point at |
| **L** Lifecycle | The agent loop, subagents, background tasks, scheduled runs | Custom commands as named prompts; loop cadence; when to fan out to subagents versus stay in one context |
| **O** Observability | Session transcripts, cost and usage reporting, per-turn diffs | Whether anyone reads them; cost attribution per project; a habit of diffing before accepting |
| **V** Verification | Diff review surfaces, rewind | Your test suite and linters as computational sensors; review skills as inferential sensors; what runs pre-commit versus post-integration |
| **G** Governance | Permission prompts, allow/deny rules, hooks, audit trail | Allowlists narrow enough to be meaningful; hooks that enforce rather than remind; what is never permitted unattended |

Reading a real harness this way exposes the asymmetry Böckeler describes: the inner harness is mature across all seven layers, while a typical outer harness is strong in C and L (everyone writes instruction files and commands) and nearly empty in O, V, and G — the same gap the 170-project survey found ecosystem-wide.

### The order that matters

1. **V before L.** Do not add orchestration complexity before you can tell whether output is correct. Fan-out multiplies unverified work.
2. **G before E permissiveness.** Widen the sandbox only after the audit trail and allowlists exist, because blast radius grows with authority.
3. **C is a budget, not a bucket.** Every always-loaded skill description and tool schema is a standing tax on the transcript. Prune what does not earn its tokens.
4. **O is what makes the other six improvable.** Without traces you are tuning a system you cannot observe, which returns you to prompt-era guesswork.

### Why this matters

The checklist converts a taxonomy into a sequence. Most teams' instinct — add agents, widen permissions, then wonder why quality is unstable — inverts all four rules, and the coupling problem guarantees the resulting system cannot be debugged one layer at a time.

---

## 9. Common Pitfalls

**Comparing agents without disclosing harnesses.** A pass@1 delta between two agents on different harnesses measures the pair, not the models. With documented 10-20 pp harness variance (and a 58.8 pp outlier), undisclosed-harness comparisons cannot support model claims.

**Optimizing one layer in isolation.** The coupling problem is not a caveat, it is the central finding. Better retrieval that consumes more context, or a stricter verifier that triggers expensive recovery loops, can each reduce end-to-end success. Test harness changes on the whole loop.

**Feedforward-only harnesses.** Instruction files with no sensors produce an agent that encodes your rules and never learns whether it followed them. Formally $d = 0$, so $P_{\text{success}} = p$ regardless of retry budget.

**Feedback-only harnesses.** Sensors with no guides produce an agent that makes the same mistake repeatedly and bills you for each attempt: expected attempts scale as $1/p$.

**Trusting AI-generated tests as the behaviour sensor.** Named explicitly as the weakest link: "this approach puts a lot of faith into the AI-generated tests, that's not good enough yet." Tests written by the same system that wrote the code share its misunderstandings.

**Reading silent sensors as high quality.** "If sensors never fire, is that a sign of high quality or inadequate detection mechanisms?" Unfalsifiable either way without deliberate fault injection.

**Expecting graph retrieval to be a pure win.** Codebase-Memory's own numbers: 83% answer quality versus 92% for file exploration, at 10x fewer tokens. It is a cost-quality trade, not an upgrade — choose it when budget-bound or for graph-native queries.

**Expanding graph neighborhoods too far.** $|N_r(A)|$ grows roughly as $\bar{k}^{\,r}$. Recall gains saturate while token cost compounds; confidence-filtered local expansion exists precisely to stop this.

**Adding a graph tool instead of using the existing loop.** LARGER's lesson: an external graph DB plus a traversal stage fragments the interaction loop and pays a coupling tax that can exceed the retrieval gain.

**Fanning out without checking the critical path.** Speedup is capped at $W/C$. A work graph that is a dependency chain gains nothing from ten agents, and the ceiling is computable before spending.

**Conflating org graph and work graph.** Persistent identity and memory belong to long-lived role nodes; per-task decomposition belongs to ephemeral task nodes. Merging them yields agents that either forget their domain or cannot be re-planned.

**Assuming harnessability.** The harness is hardest to build exactly where it is most needed. A legacy untyped codebase gets no free computational sensors, and planning as if it will is how harness projects stall.

---

## 10. References

### Harness engineering

- *Agent Harness Engineering: A Survey* — Li, Xiao, Zhang, Liu, Zhao, Liao, Ji, Wang, Ge, Xu, Fang, Xu, Zhao, Kim, Hamm, Wang, Reddy (CMU / UAB / Tulane / Yale / NEU / Stanford / Amazon / UChicago / Virginia Tech / Rutgers). ETCLOVG taxonomy, 170+ project mapping, cross-layer synthesis. [Project page: Awesome-Agent-Harness](https://github.com/ai-boost/awesome-harness-engineering) · [survey PDF](https://picrew.github.io/LLM-Harness/main.pdf)
- Birgitta Böckeler, *Harness engineering for coding agent users* — inner/outer harness, guides and sensors, computational vs inferential, harnessability, keep-quality-left. [martinfowler.com](https://martinfowler.com/articles/harness-engineering.html)
- Böckeler, *Harness engineering and agent feedback: exploring AI coding sensors* — [Thoughtworks](https://www.thoughtworks.com/insights/blog/generative-ai/harness-engineering-agent-feedback-exploring-ai-coding-sensors) · interview: [SE Radio 730](https://se-radio.net/2026/07/se-radio-730-birgitta-boeckeler-on-harness-engineering-for-ai-agents/)
- *Agent harness* — consolidated definition, term history, `Agent = Model + Harness`. [Wikipedia](https://en.wikipedia.org/wiki/Agent_harness)
- *From Prompts to Contracts: Harness Engineering for Auditable Enterprise LLM Agents* — [arXiv:2607.08028](https://arxiv.org/abs/2607.08028)
- *Harness as an Asset: Enforcing Determinism via the Convergent AI Agent Framework (CAAF)* — [arXiv:2604.17025](https://arxiv.org/abs/2604.17025)
- *Harnessing Agent Skills: Architectural Patterns and a Reference Architecture for Skill-Mediated LLM Agents* — [arXiv:2606.20631](https://arxiv.org/abs/2606.20631)
- *ORACLE-SWE: Quantifying the Contribution of Oracle Information Signals on SWE Agents* — [arXiv:2604.07789](https://arxiv.org/abs/2604.07789)

### Harness effects on benchmarks

- *Coding Agent Harness Benchmarks: Why the Harness Changes the Score* — per-harness pass rates for a fixed model; "Binding Constraint Thesis". [futureagi.com](https://futureagi.com/blog/coding-agent-harness-benchmark/)
- *Does the Harness Matter? Lessons from ALE-Claw* — [agents-last-exam.org](https://agents-last-exam.org/blogs/harness-matters)
- *SWE-bench in 2026: Benchmarks vs Scaffolding Reality* — [digitalapplied.com](https://www.digitalapplied.com/blog/swe-bench-verified-june-2026-benchmark-vs-scaffolding-analysis)

### Code graphs for agents

- *LocAgent: Graph-Guided LLM Agents for Code Localization* — [arXiv:2503.09089](https://arxiv.org/abs/2503.09089)
- *LARGER: Lexically Anchored Repository Graph Exploration and Retrieval* — [arXiv:2605.16352](https://arxiv.org/abs/2605.16352)
- *Codebase-Memory: Tree-Sitter-Based Knowledge Graphs for LLM Code Exploration via MCP* — [arXiv:2603.27277](https://arxiv.org/abs/2603.27277)
- *CodeTeam: An LLM-Powered Multi-Agent Framework for Repository-Level Code Generation* — [arXiv:2606.22082](https://arxiv.org/abs/2606.22082)
- *Code Intelligence & Code-Graph Indexing for AI Agents* — indexer landscape (repo maps, tree-sitter, SCIP, LSP-to-MCP bridges). [anthonywest.co.uk](https://anthonywest.co.uk/research/code-intelligence-indexing-2026-openai)

### Graph engineering for multi-agent systems

- *Graph Engineering: Wire Multi-Agent Orgs After Loops* — loop-to-graph progression, org vs work graph, patterns, failure modes. [explainx.ai](https://www.explainx.ai/blog/graph-engineering-ai-agents-multi-agent-organizations-2026)
- *Graph Engineering for Multi-Agent Systems: Architecture, Governance, and Observability* — [TrueFoundry](https://www.truefoundry.com/blog/graph-engineering-enterprise-guide)
- *Orchard: An Open-Source Agentic Modeling Framework* — [arXiv:2605.15040](https://arxiv.org/abs/2605.15040)

### Companion reports in this repository

- [loop_engineering.md](loop_engineering.md) — the single-agent loop this layer sits on top of
- [loop_design_playbook.md](loop_design_playbook.md) — operations, retry math, verifier error models
- [claude_code_slash_commands.md](claude_code_slash_commands.md) — the command surface of one production harness

> **A note on evidence quality.** The harness literature is young and unevenly peer-reviewed. The ETCLOVG survey and the arXiv code-graph papers are the strongest sources here; several benchmark-variance figures come from vendor and practitioner blogs whose methodology is not fully disclosed. Where numbers in this document come from such sources they are attributed inline, and the direction of the effect (harness swaps move scores by double-digit percentage points) is corroborated across independent sources even where exact magnitudes are not.
