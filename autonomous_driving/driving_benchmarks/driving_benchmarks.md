# Modern Autonomous-Driving Benchmarks & Their Evaluation Metrics

A survey of four recent (2024–2025) autonomous-driving benchmarks/models and **exactly how each one scores a model**:

| Name | Year / Venue | What it evaluates | Headline metric |
|---|---|---|---|
| **NAVSIM** | NeurIPS 2024 | Closed-loop-style planning on real data | **PDMS** / EPDMS |
| **ROADWork** | ICCV 2025 (CMU) | Work-zone perception → navigation | Task-specific (AP, 1-NED, SPICE, AE%) |
| **Impromptu VLA** | NeurIPS 2025 D&B | Corner-case driving for VLA models | nuScenes L2, **NeuroNCAP**, diagnostic QA |
| **Alpamayo-R1 (AR1)** | arXiv 2025 (NVIDIA) | Reasoning + action prediction | Open/closed-loop + **reasoning-quality** |

The point of reading these together is that they cover the **whole evaluation spectrum** of an end-to-end driving stack:

```
perception accuracy ──► planning quality ──► closed-loop safety ──► reasoning faithfulness
   ROADWork              NAVSIM (PDMS)         Impromptu/NeuroNCAP      Alpamayo-R1
```

---

## 1. NAVSIM — PDMS (Predictive Driver Model Score)

### 1.1 Position
NAVSIM sits **between open-loop and closed-loop** evaluation. Pure open-loop (e.g., nuScenes L2/collision) just compares a predicted trajectory to the human log — it is cheap but rewards "average" trajectories and is weakly correlated with real driving ability. Full closed-loop (CARLA) is faithful but expensive and uses synthetic sensors. NAVSIM keeps **real sensor data** but runs a **non-reactive simulation**: the ego trajectory is rolled out for a short horizon and scored by a rule-based oracle, so it approximates closed-loop behaviour without a reactive simulator. It was the official platform of the CVPR 2024 End-to-End Driving Challenge (143 teams, 463 entries).

### 1.2 PDMS formula (NAVSIM v1)
PDMS is named after the **PDM** rule-based planner. It combines safety **gates** (multipliers) with weighted **quality** terms:

$$
\text{PDMS} = \left( \prod_{m \in \{NC,\, DAC\}} m \right) \cdot \frac{\sum_{m \in \{TTC,\, EP,\, C\}} w_m \, m}{\sum_{m \in \{TTC,\, EP,\, C\}} w_m}
$$

The **left product** is the *safety gate* — any `0` (an at-fault collision or leaving the drivable area) zeroes the whole score. The **right fraction** is the *weighted quality average* over the remaining terms.

| Sub-metric | Type | Range / weight | Meaning |
|---|---|---|---|
| **NC** — No at-fault Collision | multiplier | {0, ½, 1} | At-fault collision zeroes (or halves) the score |
| **DAC** — Drivable Area Compliance | multiplier | {0, 1} | Leaving the drivable area zeroes the score |
| **EP** — Ego Progress | weighted | w=5, [0,1] | Forward progress vs. the PDM reference |
| **TTC** — Time-to-Collision | weighted | w=5, {0,1} | Maintains a safe time-to-collision buffer |
| **C** — Comfort | weighted | w=3, {0,1} | Acceleration/jerk within comfort bounds |

**Key idea**: the two multipliers act as *hard safety gates* — any unsafe trajectory collapses to ~0 regardless of how smooth or progressive it is. Only trajectories that pass the gates get graded on the weighted average.

### 1.3 EPDMS (NAVSIM v2, Extended)
v2 adds more gates and quality terms and a **filter** that disables a penalty when the *human* driver also violated it (avoiding punishing the model for unavoidable situations):

$$
\text{EPDMS} = \left( \prod_{m \in \{NC,\, DAC,\, DDC,\, TLC\}} f_m \right) \cdot \frac{\sum_{m \in \{TTC,\, EP,\, HC,\, LK,\, EC\}} w_m \, f_m}{\sum_{m \in \{TTC,\, EP,\, HC,\, LK,\, EC\}} w_m}
$$

where each `f_m` is the *filtered* sub-score (the penalty is disabled when the human driver also violates metric `m`).

New gates: **DDC** (Driving Direction Compliance, {0,½,1}), **TLC** (Traffic Light Compliance, {0,1}).
New quality terms: **LK** (Lane Keeping, w=2, disabled at intersections), **HC** (History Comfort), **EC** (Extended Comfort, w=2 — frame-to-frame trajectory consistency).

> Takeaway: PDMS/EPDMS is a **gated weighted score in [0,1]×100**. Safety is multiplicative (one violation ≈ fail), comfort/progress are additive. This is the de-facto standard for real-data planning benchmarks today.

---

## 2. ROADWork — Work-Zone Perception & Navigation

### 2.1 Position
ROADWork (CMU) targets a **long-tail but safety-critical** scenario that most datasets ignore: construction/work zones. It collects images and sequences from **18 U.S. cities** with 15 segmented object classes (workers, cones, barriers, equipment…), sign graphics, sign text, scene descriptions, and auto-estimated passable trajectories. Unlike NAVSIM (one planning score), ROADWork is a **multi-task benchmark** — each task has its own standard metric.

### 2.2 Tasks and metrics
| # | Task | Metric | What it measures | Baseline |
|---|---|---|---|---|
| 1 | Object detection & segmentation | **AP** @IoU 0.50:0.95 | Localize/segment 15 work-zone classes | 39.0 AP (Mask DINO + video prop.; foundation models 2.9–4.2) |
| 2 | Fine-grained sign graphics | **AP** @IoU 0.50:0.95 | Recognize 62 sign-graphic types | 32.8 AP (vs 0–1.1 for zero-shot) |
| 3 | Sign **text** reading | **1-NED** (1 − Normalized Edit Distance) | OCR accuracy on sign text | 81.3% (Glass + 3× crop) |
| 4 | Work-zone description | **SPICE** | Semantic correctness of VLM caption | 46.6 (vs 3.9–9.9 pretrained) |
| 5 | Work-zone **discovery** | **Precision** + discovery-rate × | Mine new work-zone images from unlabeled data | 84.9% precision, 12.8× more zones |
| 6 | Goal & pathway prediction | **AE%<θ** (Angular Error) | % of predicted goals/paths within θ° of GT direction in image space | 53.6% of goals AE<0.5° (+9.9%) |

### 2.3 Why these metrics
- **AP** (COCO-style, averaged over IoU thresholds 0.50–0.95) is the standard for detection/segmentation — rewards both correct class and tight masks.
- **1-NED** turns edit distance into a [0,1] "higher-is-better" OCR score (1.0 = perfect string).
- **SPICE** scores captions on a *scene-graph* of objects/attributes/relations — better aligned with semantic correctness than BLEU/CIDEr for descriptions.
- **Angular Error (AE)** is a lightweight planning proxy: rather than metric L2, it asks "is the predicted heading toward the goal/path correct within θ degrees?" — robust to image-space scale.

> Takeaway: ROADWork measures the **perception→understanding→navigation chain** for one rare scenario, with a different well-established metric per task rather than a single aggregate.

---

## 3. Impromptu VLA — Corner-Case Evaluation for VLA Models

### 3.1 Position
Impromptu VLA is a **dataset** (>80k curated clips distilled from 2M+ clips across 8 open datasets) built on a taxonomy of **four unstructured corner-case types**: unclear road boundaries, temporary traffic-rule changes, unconventional dynamic obstacles, and challenging road conditions. Its contribution to *evaluation* is twofold: (a) it improves models that are then scored on **established** benchmarks, and (b) it provides its own **diagnostic QA** validation set.

### 3.2 Metrics
**(a) Open-loop — nuScenes L2**
- **L2 distance error** (meters) between predicted and GT ego trajectory at **1s / 2s / 3s** horizons, plus the average. Lower is better. (e.g., 3B model 0.30 m avg vs 0.34 m baseline.)
- Usually reported with **collision rate** (% of predicted trajectories overlapping objects).

**(b) Closed-loop — NeuroNCAP**
- **NeuroNCAP Score (NNS) ∈ [0, 5]**: 5.0 if no collision; otherwise
$$ \text{NNS} = 4.0 \cdot \max\left(0,\; 1 - \frac{v_i}{v_r}\right) $$
where $v_i$ = impact speed, $v_r$ = reference speed → softer collisions score higher. (e.g., 1.77 → 2.15.)
- **Collision Rate (%)** broken down by interaction type (frontal / side / stationary). (e.g., 72.5% → 65.5%.)

**(c) Diagnostic QA (their own validation set)** — accuracy per capability:
| Capability | Task | Example gain |
|---|---|---|
| Perception | VRU identification | 0.87 → 0.91 |
| Perception | Traffic-light detection | 0.95 → 0.96 |
| Prediction | Dynamic-object behaviour | 0.20 → 0.92 |
| Planning | Meta-planning (decision) | 0.56 → 0.84 |
| Planning | Trajectory L2 (1–4s) | 6.62 m → 0.69 m |

> Takeaway: Impromptu VLA evaluates a model on **three levels** — open-loop accuracy (nuScenes L2), closed-loop safety (NeuroNCAP graded by impact speed, not just yes/no collision), and **per-capability QA accuracy** that isolates where a VLA fails.

---

## 4. Alpamayo-R1 (AR1) — Reasoning + Action Prediction

### 4.1 Position
AR1 (NVIDIA) is a **reasoning-centric VLA**: it couples a *Chain-of-Causation* (CoC) reasoning trace with a diffusion trajectory decoder. Its evaluation is notable because, beyond the usual trajectory metrics, it explicitly **scores the quality of the reasoning itself** — something the other three do not.

### 4.2 Metrics

**(a) Open-loop trajectory**
- **minADE** (minimum Average Displacement Error): smallest mean L2 distance between any of the K sampled trajectories and the GT — rewards multimodal planners that include a correct mode.
- **L2 @ horizons** (1s/3s/5s): position error at fixed future times.
- **Off-road rate**: % of trajectory leaving valid driving regions.
- **Collision / close-encounter rate**: collisions, plus near-misses within a danger threshold.

**(b) Closed-loop simulation** (model output feeds back into the sim)
- **35% off-road reduction** — trajectories stay in valid regions 35% more often than the trajectory-only baseline.
- **25% close-encounter reduction** — 25% fewer near-collision situations.

**(c) Reasoning-quality** (the distinctive part)
- Uses an **LLM critic (GPT-5)** over a curated **2K-sample** set, asking **structured True/False** questions (not free-form grading) to reduce evaluator hallucination.
- **Causal-relationship score**: does the stated reasoning express genuine cause→effect for the chosen action? Structured CoC traces improved this by **+132.8%** vs free-form reasoning.
- **Reasoning–action consistency**: does the explicit reasoning chain actually match the executed trajectory? Reported **+37%**.
- Overall **+45% reasoning quality** and **+12% planning accuracy on challenging (long-tail) cases** vs a trajectory-only baseline; on-vehicle road test at **99 ms** latency.

> Takeaway: AR1 adds a **reasoning-faithfulness axis** on top of standard open/closed-loop planning metrics — measuring not just *what* the car does but whether its stated *why* is causally valid and consistent with the action.

---

## 5. Side-by-Side Summary

| Axis | NAVSIM | ROADWork | Impromptu VLA | Alpamayo-R1 |
|---|---|---|---|---|
| **Primary unit** | Benchmark/sim | Dataset + benchmark | Dataset + diagnostic set | Model + eval suite |
| **Loop type** | Non-reactive sim (pseudo closed-loop) | Open-loop per-task | Open-loop + closed-loop | Open-loop + closed-loop |
| **Headline metric** | PDMS / EPDMS ∈ [0,1] | AP, 1-NED, SPICE, AE% | L2, NeuroNCAP, QA acc | minADE/L2 + reasoning score |
| **Safety modelling** | Multiplicative gates (NC, DAC, TTC…) | — (perception focus) | NeuroNCAP impact-speed grading | off-road / close-encounter rate |
| **Distinctive idea** | Gated weighted score on real data | One metric per sub-task, long-tail work zones | Corner-case taxonomy + per-capability QA | Scores reasoning faithfulness, not just trajectory |
| **Good for** | Comparing planners fairly on real logs | Diagnosing work-zone perception | Stress-testing VLA corner cases | Evaluating reasoning VLAs end-to-end |

### How to choose
- **"Is my planner safe and competent on real data?"** → NAVSIM **PDMS/EPDMS**.
- **"Does my perception stack survive construction zones?"** → ROADWork per-task metrics.
- **"Does my VLA handle weird corner cases, and where does it break?"** → Impromptu VLA (NeuroNCAP + diagnostic QA).
- **"Is my model's reasoning actually causal and consistent with its driving?"** → Alpamayo-R1's reasoning-quality metrics.

---

## References

- **NAVSIM**: Dauner et al., *NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking*, NeurIPS 2024. [arXiv:2406.15349](https://arxiv.org/abs/2406.15349) · [metrics doc](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md)
- **ROADWork**: Ghosh et al., *ROADWork: A Dataset and Benchmark for Learning to Recognize, Observe, Analyze and Drive Through Work Zones*, ICCV 2025. [arXiv:2406.07661](https://arxiv.org/abs/2406.07661) · [project](https://www.cs.cmu.edu/~roadwork/)
- **Impromptu VLA**: *Impromptu VLA: Open Weights and Open Data for Driving Vision-Language-Action Models*, NeurIPS 2025 D&B. [arXiv:2505.23757](https://arxiv.org/abs/2505.23757) · [code](https://github.com/ahydchh/Impromptu-VLA)
- **Alpamayo-R1**: Wang, Luo et al., *Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail*, NVIDIA, 2025. [arXiv:2511.00088](https://arxiv.org/abs/2511.00088) · [model](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- Related metrics: **NeuroNCAP** (closed-loop NCAP-style safety), **nuScenes** open-loop L2/collision, **PDM** rule-based planner (nuPlan).
