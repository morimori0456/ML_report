---
title: "How NAVSIM Computes PDMS from Driving Logs — Non-Reactive Simulation Explained"
description: "Why a non-reactive, not-closed-loop benchmark still yields collision and time-to-collision metrics — the two meanings of closed-loop, the LQR + bicycle ego rollout, and the gated PDMS formula."
---

> Answers a specific confusion: NAVSIM is called "non-reactive" and "not closed-loop," yet it reports a PDM Score (PDMS) built from collisions, drivable-area compliance, and time-to-collision — metrics that sound closed-loop. How? See the companion notebook [navsim_pdms_mechanism.ipynb](https://github.com/morimori0456/ML_report/blob/main/autonomous_driving/driving_benchmarks/navsim_pdms_mechanism.ipynb) for a from-scratch PDMS calculator and the correlation argument.

The word "closed-loop" is overloaded, and that is the whole source of the confusion. It bundles two independent things: (1) whether the ego's proposed trajectory is *physically simulated* forward, and (2) whether the *environment re-plans and reacts* to the ego over time. NAVSIM keeps (1) — it genuinely unrolls the ego through a bicycle model and measures what happens — but drops (2): other agents replay their logged futures and the ego commits to one trajectory for the whole horizon. That single design choice is what lets NAVSIM produce closed-loop-style safety metrics at the cost of a single forward pass, no reactive simulator required. This report explains the exact pipeline, the aggregation formula, why the non-reactive approximation holds, and where it breaks. It is the mechanistic follow-up to [nuplan_dataset.md](https://github.com/morimori0456/ML_report/blob/main/autonomous_driving/nuplan/nuplan_dataset.md) and [driving_benchmarks.md](https://github.com/morimori0456/ML_report/blob/main/autonomous_driving/driving_benchmarks/driving_benchmarks.md).

---

## Table of Contents
1. [The Real Question: Two Meanings of "Closed-Loop"](#1-the-real-question-two-meanings-of-closed-loop)
2. [The PDMS Pipeline, Step by Step](#2-the-pdms-pipeline-step-by-step)
3. [The PDMS Formula](#3-the-pdms-formula)
4. [Why Non-Reactive Over 4 Seconds Is a Good Proxy](#4-why-non-reactive-over-4-seconds-is-a-good-proxy)
5. [Frame Filtering: Why It Is Mandatory](#5-frame-filtering-why-it-is-mandatory)
6. [What NAVSIM Gives Up (and EPDMS)](#6-what-navsim-gives-up-and-epdms)
7. [Common Pitfalls](#7-common-pitfalls)
8. [References](#8-references)

---

## 1. The Real Question: Two Meanings of "Closed-Loop"

An open-loop displacement metric (ADE/FDE, or nuScenes "L2") compares your trajectory to the logged human path and never simulates anything — it cannot tell you whether you hit a car. NAVSIM is **not** that. NAVSIM is a *simulation-based* metric. The confusion comes from equating "closed-loop" with "simulation." Separate the two axes:

| Axis | Open-loop L2 | **NAVSIM (PDMS)** | Full closed-loop (nuPlan) |
|---|---|---|---|
| Ego trajectory physically simulated? | No | **Yes** (LQR + bicycle model) | Yes |
| Ego re-plans during rollout? | No | **No** — one trajectory, held fixed for 4 s | Yes — re-plans every 0.1 s |
| Background agents react to ego? | No | **No** — replay logged futures | Yes — reactive (IDM) |
| Cost | 1 forward pass | **1 forward pass** | Full iterative rollout + simulator |
| Measures safety/compliance? | No | **Yes** | Yes |

### Key insight
> **"Non-reactive" describes the *environment*, not the *ego*.** NAVSIM still takes your proposed trajectory, runs an LQR controller + kinematic bicycle model to get the ego states it actually reaches, and checks those states for collisions, off-road, TTC, and progress against the world. What it does *not* do is let other cars respond or let the ego re-plan. So it earns closed-loop-*style* metrics without a closed-loop *system*.

**Why this matters:** once you see that NAVSIM simulates the ego but freezes the world, "how can a non-reactive metric measure collisions?" stops being paradoxical — you collide with a car that is following its logged path, and the simulator sees the overlap.

---

## 2. The PDMS Pipeline, Step by Step

Given one evaluation frame from the data (drawn from **OpenScene**, a redistribution of nuPlan, sampled at 2 Hz), NAVSIM turns a single predicted trajectory into a PDMS as follows.

```
predicted trajectory (poses, 4 s)             logged agent futures (4 s)
        │                                              │
        ▼                                              ▼
 ┌──────────────────────┐                    ┌───────────────────────┐
 │ LQR controller       │  each 0.1 s        │ non-reactive replay:  │
 │ + kinematic bicycle  │ ─────────────────► │ agents follow logged  │
 │ → simulated ego state│                    │ future, ignore ego    │
 └──────────────────────┘                    └───────────────────────┘
        │  10 Hz over 4 s = 40 steps                    │
        └──────────────────────┬───────────────────────┘
                               ▼
             per-step checks against map + agents
   NC (no at-fault collision) · DAC (drivable area) · DDC (direction)
   TTC (time-to-collision)   · EP (ego progress)  · C (comfort)
                               ▼
                       aggregate → PDMS
```

1. **Model outputs a trajectory** — a sequence of future ego poses over a **4-second** horizon.
2. **Simulate the ego** — at each iteration an **LQR controller** computes steering/acceleration to track the proposed trajectory, and a **kinematic bicycle model** propagates the ego. This runs at **10 Hz over the 4 s horizon = 40 steps**. The executed path can differ from the planned one if the plan is dynamically infeasible or jerky.
3. **Propagate the world non-reactively** — background agents follow their **logged future tracks**; they do not respond to the ego. The ego also does not re-plan: the one trajectory is held fixed for the entire 4 s.
4. **Compute sub-metrics** from the 40 simulated ego states + agent boxes + HD map: at-fault collision, drivable-area compliance, driving-direction compliance, time-to-collision, ego progress (relative to a privileged PDM planner's progress), and comfort (accel/jerk/yaw bounds).
5. **Aggregate** into the PDMS scalar (next section).

### Key insight
> **The ego rollout is real; the environment rollout is cheap.** Because there is no re-planning loop and no reactive agents, the entire evaluation of one sample is a single deterministic 40-step unroll — embarrassingly parallel across the 12k `navtest` frames, which is exactly why NAVSIM is cheap enough to be a leaderboard.

**Why this matters:** the bicycle-model step is what makes a jittery or kinematically-infeasible trajectory score badly even though its waypoints look close to the human path — the metric punishes plans the car cannot actually execute.

---

## 3. The PDMS Formula

PDMS uses the same **gated** structure as nuPlan's closed-loop score: multiplicative safety **penalties** times a weighted average of quality **sub-scores**.

$$
\text{PDMS} = \underbrace{\text{NC}\cdot\text{DAC}}_{\text{penalty gates}}\;\times\;
\frac{5\,\text{EP} + 5\,\text{TTC} + 2\,\text{C}}{5+5+2}
$$

| Symbol | Metric | Type | Weight |
|---|---|---|---|
| NC | No at-fault Collision | multiplier (gate) | — |
| DAC | Drivable Area Compliance | multiplier (gate) | — |
| EP | Ego Progress (vs. privileged PDM planner) | weighted | 5 |
| TTC | Time-to-Collision within bound | weighted | 5 |
| C | Comfort (accel/jerk/yaw bounds) | weighted | 2 |

### Key insight
> **The gate multiplies, so safety dominates.** If NC = 0 (you caused a collision) or DAC = 0 (you left the drivable area), the product is 0 no matter how much progress or comfort you accumulated. Only among safe, on-road rollouts do EP/TTC/C differentiate planners. This is identical in spirit to nuPlan's `NC·DAC·DDC·MP × (weighted EP/TTC/SC/C)`; NAVSIM v1 simply uses `{NC, DAC}` as the gates and `{EP, TTC, C}` as the weighted terms.

**Why this matters:** it explains leaderboard numbers such as human ≈ **94.8** PDMS on filtered `navtest` and a constant-velocity agent ≈ **22** after filtering — the constant-velocity plan trips the gates in non-trivial scenes.

---

## 4. Why Non-Reactive Over 4 Seconds Is a Good Proxy

The gamble NAVSIM makes is: *a single non-reactive 4-second unroll ranks planners almost the same way a full reactive closed-loop rollout would.* Two reasons this holds:

| Reason | Explanation |
|---|---|
| **Short horizon** | Over 4 s, other agents rarely need to react to a *reasonable* ego plan; their logged futures are a fine approximation. Reactivity matters over long horizons where the ego's influence propagates. |
| **At-fault attribution** | Collisions are only counted when the ego is **at fault** (e.g. it drives into an agent). A logged agent rear-ending a stopped ego is not charged to the planner, so non-reactive replay does not manufacture unfair collisions. |

The NAVSIM authors validate this empirically: PDMS from non-reactive simulation **correlates strongly with true closed-loop scores**, and — critically — replacing the non-reactive background agents with **reactive IDM** agents *barely changes the correlation*. That is the empirical license to skip reactivity.

### Key insight
> **NAVSIM buys ~90% of closed-loop's signal for ~1% of its cost.** The remaining gap is exactly the long-horizon, ego-induced interactions that a 4 s non-reactive window cannot see — which is why NAVSIM is a *screening* benchmark, not a replacement for on-road or full closed-loop testing.

**Why this matters:** it tells you when to trust PDMS (ranking planners on typical urban frames) and when not to (scenarios whose difficulty *is* the reactive interaction, e.g. negotiating a merge where other cars must yield).

---

## 5. Frame Filtering: Why It Is Mandatory

Simulation-based metrics are only informative on frames where the driving decision is non-trivial. If you keep straight-road cruising frames, a dumb **constant-velocity** planner scores well and the metric loses discriminative power. NAVSIM curates `navtest`/`navtrain` by removing "too easy" and "degenerate" frames:

- **Remove trivially easy scenes:** drop any frame where a constant-velocity agent already exceeds **PDMS 0.8**.
- **Remove degenerate scenes:** drop any frame where even the **human** trajectory scores **below 0.8** (data/label problems, unavoidable situations).

The effect is dramatic: the constant-velocity baseline falls from **~79% PDMS (unfiltered) to ~22% (filtered)** — the filtering is what turns PDMS into a metric that actually separates planners.

| Split | Size | Note |
|---|---|---|
| `navtrain` | ~103k samples | training |
| `navtest` | ~12k samples | leaderboard evaluation |
| `navmini` | ~396 samples | rapid local testing (the split used for the CV 0.308 vs Human 0.914 check in [navsim_hands_on.md](https://github.com/morimori0456/ML_report/blob/main/autonomous_driving/driving_benchmarks/navsim_hands_on.md)) |

**Why this matters:** PDMS numbers are only comparable *on the same filtered split*. Reporting PDMS on unfiltered frames inflates weak planners and is a common way to accidentally cheat the metric.

---

## 6. What NAVSIM Gives Up (and EPDMS)

Non-reactive, single-shot evaluation has real blind spots: no reactive negotiation, no error compounding beyond 4 s, and no penalty for behaviours that only bite over long horizons. NAVSIM v2 introduces the **Extended PDM Score (EPDMS)** to close some gaps:

$$
\text{EPDMS} = \text{NC}\cdot\text{DAC}\cdot\text{DDC}\cdot\text{TLC}\;\times\;
\frac{5\,\text{EP}+5\,\text{TTC}+2\,\text{LK}+2\,\text{HC}+2\,\text{EC}}{16}
$$

| Change from PDMS | What it adds |
|---|---|
| DDC promoted to a **gate** | Wrong-way driving now zeros the score |
| **TLC** gate | Traffic-Light Compliance (running reds zeros the score) |
| **LK** weighted term | Lane Keeping |
| C → **HC** | History Comfort (comfort judged against recent history, penalising abrupt changes) |
| **EC** weighted term | Extended Comfort |
| two-stage weighting | separates trajectory proposals from a second stage; false-positive/history penalties |

| Metric | Reactivity | Re-planning | Horizon | Cost | Best for |
|---|---|---|---|---|---|
| L2 / ADE | none | none | n/a | trivial | quick sanity only |
| **PDMS** | non-reactive | none | 4 s | 1 pass | planner ranking |
| **EPDMS** | non-reactive (+TLC/LK/DDC gates) | none | 4 s | 1 pass | stricter planner ranking |
| nuPlan CL | reactive IDM | every 0.1 s | ~15 s | full rollout | closed-loop validation |

**Why this matters:** EPDMS tightens the rules (more gates, more comfort/lane terms) but keeps the same core trick — non-reactive, single-shot simulation — so everything in this report still applies; only the sub-metric set grows.

---

## 7. Common Pitfalls

- **Thinking "non-reactive" means "no simulation."** NAVSIM simulates the ego through an LQR+bicycle model; it only freezes the *other* agents. Collisions are measured on the simulated ego rollout.
- **Expecting closed-loop numbers to match PDMS.** PDMS is a *proxy*. It correlates with closed-loop but is not identical, especially on reactive-negotiation scenarios.
- **Reporting PDMS on unfiltered frames.** Without NAVSIM's frame filtering a constant-velocity baseline scores ~79%; always evaluate on the official filtered split.
- **Ignoring the bicycle-model step.** A trajectory whose waypoints match the human path but is jerky/infeasible will be tracked poorly by the LQR controller and lose comfort/TTC — good L2 does not imply good PDMS.
- **Forgetting the gate multiplies.** Optimising EP/TTC/comfort is wasted if NC or DAC is 0. Fix collision/off-road behaviour first.
- **Confusing PDMS (v1) with EPDMS (v2).** They have different gates and weighted terms and are not directly comparable; state which one you report.
- **Mismatched at-fault logic.** A collision where the ego is not at fault is not charged; do not "fix" phantom collisions caused by non-reactive replay — the metric already ignores them.

---

## 8. References

- Dauner et al., "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking," NeurIPS 2024. arXiv: https://arxiv.org/abs/2406.15349
- Dauner et al., "Parting with Misconceptions about Learning-based Vehicle Motion Planning" (PDM / PDM-Closed), CoRL 2023. arXiv: https://arxiv.org/abs/2306.07962
- Caesar et al., "nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles," CVPR 2021 ADP3 Workshop. arXiv: https://arxiv.org/abs/2106.11810
- NAVSIM GitHub (metric implementation): https://github.com/autonomousvision/navsim
- OpenScene dataset: https://github.com/OpenDriveLab/OpenScene
- Companion reports in this repo: [nuplan_dataset.md](https://github.com/morimori0456/ML_report/blob/main/autonomous_driving/nuplan/nuplan_dataset.md), [driving_benchmarks.md](https://github.com/morimori0456/ML_report/blob/main/autonomous_driving/driving_benchmarks/driving_benchmarks.md), [navsim_hands_on.md](https://github.com/morimori0456/ML_report/blob/main/autonomous_driving/driving_benchmarks/navsim_hands_on.md)
