---
title: "Knowledge Distillation — with a Focus on Intermediate-Feature Distillation"
description: "Why distillation works and the feature-matching family — logit KD, FitNets, Attention Transfer, FSP, NST, PKT, RKD, CRD, OFD, ReviewKD — with the adapters, transforms, and loss weighting that make them work."
---

> A teacher (large, accurate) transfers its "knowledge" to a student (small, fast).
> This guide builds intuition for **why** distillation works, then dives into the family
> of methods that match **intermediate feature maps** rather than only the final logits.

For a runnable comparison (logit-KD vs FitNets vs Attention Transfer on a small CNN), see
[knowledge_distillation_demo.ipynb](https://github.com/morimori0456/ML_report/blob/main/distillation/knowledge_distillation_demo.ipynb). For the **rest** of the
field — better logit losses (DKD), online/self distillation, data-free, and the
detection/segmentation/LLM recipes — see [distillation_methods_survey.md](https://github.com/morimori0456/ML_report/blob/main/distillation/distillation_methods_survey.md).

---

## Table of Contents
1. [Why Distillation Works](#1-why-distillation-works)
2. [The Three Families](#2-the-three-families)
3. [Response-Based (Logit) KD — the baseline](#3-response-based-logit-kd--the-baseline)
4. [Feature-Based Distillation: the core problem](#4-feature-based-distillation-the-core-problem)
5. [FitNets — hints + regressor](#5-fitnets--hints--regressor)
6. [Attention Transfer (AT)](#6-attention-transfer-at)
7. [FSP, NST, PKT — other feature transforms](#7-fsp-nst-pkt--other-feature-transforms)
8. [Relation-Based: RKD and CRD](#8-relation-based-rkd-and-crd)
9. [Stronger feature methods: OFD and ReviewKD](#9-stronger-feature-methods-ofd-and-reviewkd)
10. [Design Choices Cheat-Sheet](#10-design-choices-cheat-sheet)
11. [Common Pitfalls](#11-common-pitfalls)

---

## 1. Why Distillation Works

A trained classifier outputs more than the top-1 label. The **full probability vector**
encodes how the teacher relates classes — "this 4 looks 30% like a 9, 5% like a 7." These
**soft targets** carry *dark knowledge*: the relative similarities the one-hot label throws away.

Training the student to match the teacher's soft distribution gives it a much richer signal
than the hard label alone — effectively many "soft constraints" per example. The student often
reaches accuracy it could **not** reach training on labels alone, because the teacher has
already smoothed the loss landscape and encoded inter-class structure.

> **When does distillation actually help?** Only when the teacher knows something the student
> can't get from its own labels. The clearest case is a **transfer set**: inputs the student
> has *no labels* for, on which the teacher provides soft targets (and features). The companion
> notebook uses exactly this semi-supervised regime — 150 labels for the student, teacher
> guidance on 1297 images — so the distillation signal carries real information and the gains
> are large (+12 pts). On an easy task where the teacher is barely better than the student,
> distillation adds little; the *method* is correct, but there is no extra knowledge to transfer.

> Key insight that motivates this whole guide: the logits are only the **last** thing the
> teacher computes. Everything useful was already present in its **intermediate
> representations** — the feature maps. Feature-based distillation taps that signal directly,
> which matters most when the student is *much* smaller or the task is *hard* (detection,
> segmentation, dense prediction) where a single logit vector is a weak target.

---

## 2. The Three Families

| Family | What is matched | Representative methods |
|---|---|---|
| **Response-based** | Final outputs (logits / soft labels) | Hinton KD, DKD, decoupled-KD |
| **Feature-based** | Intermediate activations (feature maps) | **FitNets, AT, FSP, NST, PKT, OFD, ReviewKD** |
| **Relation-based** | Relations *between* samples or layers | RKD, CRD, similarity-preserving KD |

The dividing line that matters in practice:
- **Response-based** is architecture-agnostic and trivial to add (just need both logits), but
  the signal is low-dimensional (C numbers per sample).
- **Feature-based** is high-bandwidth (C×H×W numbers per sample) but must solve a
  **representation-matching problem**: teacher and student features live in different spaces.

This document focuses on the feature-based family (§4–§9).

---

## 3. Response-Based (Logit) KD — the baseline

Hinton et al. (2015). Soften both distributions with **temperature** `T`:

$$
p_i^{T} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Loss = weighted sum of distillation (match teacher) + the usual hard-label CE:

$$
\mathcal{L} = \alpha \, T^2 \, \mathrm{KL}\left(p^{T}_{\text{student}} \, \Vert \, p^{T}_{\text{teacher}}\right) + (1-\alpha) \, \mathrm{CE}(y, p^{1}_{\text{student}})
$$

- **T > 1** softens the distribution, amplifying the small "dark-knowledge" probabilities.
- The **T²** factor rescales the KD gradient so it stays comparable to the CE gradient
  (the softmax gradient shrinks by ~1/T²).

This is the baseline every feature method is compared against — and usually **combined** with.

---

## 4. Feature-Based Distillation: the core problem

We want the student's intermediate feature map `F_S ∈ ℝ^{C_S×H_S×W_S}` to carry the same
information as the teacher's `F_T ∈ ℝ^{C_T×H_T×W_T}`. Three sub-problems must be solved:

1. **Where to distill** — which (teacher layer, student layer) pairs to connect.
   Usually end-of-stage feature maps (after each spatial downsampling).
2. **Dimension mismatch** — `C_S ≠ C_T` (and possibly different H,W). Solved with an
   **adapter / regressor**: a 1×1 conv (or linear) `r(·)` that lifts the student feature
   into the teacher's channel space, plus interpolation for spatial size.
3. **What to compare, and how** — apply a transform `g(·)` to each feature, then a distance:
   $$ \mathcal{L}_{\text{feat}} = d\big(g(r(F_S)),\; g(F_T)\big) $$

The methods below differ almost entirely in the **transform `g`** and **distance `d`**:

| Method | Transform g(F) | Distance d | Intuition |
|---|---|---|---|
| FitNets | identity (after regressor) | MSE | match raw activations |
| AT | spatial attention `∑_c F_c²` | MSE on L2-normalized | match *where* the net looks |
| NST | per-neuron activation distribution | MMD | match neuron selectivity patterns |
| FSP | Gram between two layers | MSE | match the *flow* (how features evolve) |
| PKT | pairwise sample affinity | KL | match sample-similarity structure |
| OFD | margin-ReLU feature | partial (masked) L2 | transfer only useful (positive) info |

---

## 5. FitNets — hints + regressor

Romero et al. (2015), the first feature-distillation method. The teacher's chosen layer is a
**hint**; the student's corresponding layer is the **guided layer**.

Because `C_S ≠ C_T`, attach a **regressor** `r` (a conv layer) on top of the student feature
so its channel count matches the teacher's, then minimize MSE:

$$
\mathcal{L}_{\text{hint}} = \tfrac12 \big\Vert\, r(F_S) - F_T \,\big\Vert_2^2
$$

Classic two-stage recipe:
1. **Stage 1 (hint training)**: train student up to the guided layer using only
   `L_hint` — a good initialization for the lower half of the student.
2. **Stage 2**: train the full student with logit-KD (§3).

In practice it's common to combine everything in **one stage**:
`L = CE + α·KD + β·L_hint`. The regressor `r` is trained jointly and **thrown away at
inference** — it costs nothing in the deployed student.

> Why a regressor and not just "pick C_S=C_T"? Forcing equal channels would constrain the
> student architecture. The 1×1-conv adapter decouples "what we distill" from "the student's
> shape," so any student can learn from any teacher.

---

## 6. Attention Transfer (AT)

Zagoruyko & Komodakis (2017). Instead of matching raw activations (which is a *very* strict
target — exact values), match the **spatial attention map**: where in the image the network
concentrates its response.

For a feature map `F ∈ ℝ^{C×H×W}`, collapse channels into a single `H×W` attention map:

$$
A(F) = \sum_{c=1}^{C} \lvert F_c \rvert^{p}\quad(\text{typically } p=2)
$$

Then flatten, **L2-normalize**, and take MSE:

$$
\mathcal{L}_{\text{AT}} = \left\Vert
\frac{\mathrm{vec}(A(F_S))}{\Vert \mathrm{vec}(A(F_S))\Vert_2}
- \frac{\mathrm{vec}(A(F_T))}{\Vert \mathrm{vec}(A(F_T))\Vert_2}
\right\Vert_2
$$

Why this is often **better than FitNets**:
- The channel sum makes it **dimension-agnostic** — `C_S` and `C_T` need not match, *no
  adapter needed* (only same H×W, which interpolation can fix).
- Matching *where* the net attends is a **softer, more transferable** target than matching
  exact activation values — the student keeps freedom in *how* it represents that region.

This is a strong, cheap default for CNNs and is implemented in the companion notebook.

---

## 7. FSP, NST, PKT — other feature transforms

**FSP (Flow of Solution Procedure)** — Yim et al. (2017). Knowledge = how features *change
between* two layers, captured by a Gram matrix:
$$ G = \frac{1}{H W}\sum_{h,w} F^{(1)}_{h,w}\, {F^{(2)}_{h,w}}^{\top} \in \mathbb{R}^{C_1\times C_2} $$
Match `G_S ≈ G_T` with MSE. Transfers the *process*, not the static features.

**NST (Neuron Selectivity Transfer)** — Huang & Wang (2017). Treat each spatial location's
activation pattern as a sample from a distribution and match teacher/student distributions
with **Maximum Mean Discrepancy (MMD)**. Matches *which neurons fire together* rather than
exact values.

**PKT (Probabilistic Knowledge Transfer)** — Passalis & Tefas (2018). Build a **pairwise
affinity** matrix over the batch (cosine-similarity → conditional probability that sample i
picks j as neighbor) in both spaces and match them with KL divergence. This is the bridge to
relation-based methods: it preserves the *geometry of the sample manifold*, ignoring the
coordinate system entirely (so no adapter needed).

---

## 8. Relation-Based: RKD and CRD

These match relations *between examples* instead of individual features — naturally solving
the dimension-mismatch problem because relations are scalars.

**RKD (Relational KD)** — Park et al. (2019). Two structural losses over a batch:
- **Distance-wise**: match normalized pairwise distances `‖f_i − f_j‖` between students and teacher.
- **Angle-wise**: match the angle `∠(f_i, f_j, f_k)` formed by triplets — a higher-order
  relation that captures the *shape* of the embedding, invariant to scale/rotation.

**CRD (Contrastive Representation Distillation)** — Tian et al. (2020). Frame distillation as
a **contrastive** problem: the student feature of sample *x* should be close (positive) to the
teacher feature of the *same* `x`, and far (negative) from teacher features of other samples.
Maximizes a lower bound on the **mutual information** between teacher and student
representations. One of the strongest general-purpose feature methods, at the cost of a
memory bank of negatives.

---

## 9. Stronger feature methods: OFD and ReviewKD

**OFD (Overhaul of Feature Distillation)** — Heo et al. (2019). Three careful design fixes:
- **Distillation position**: distill *before* ReLU (pre-activation), keeping negative info.
- **Margin ReLU** on the teacher: pass negative responses only if below a learned negative
  margin — transfer "this neuron should be off" only when it's meaningfully off.
- **Partial L2 distance**: don't penalize the student for being positive where the teacher is
  negative (those neurons are simply "off"; exact value is irrelevant). A masked L2 that only
  pulls on informative dimensions.

**ReviewKD (Knowledge Review)** — Chen et al. (2021). Key realization: a student's **deep**
layer can learn from the teacher's **shallow** layers too. Instead of only same-stage pairs,
it routes **multi-level** teacher features into each student stage through an **Attention-Based
Fusion (ABF)** + **Hierarchical Context Loss (HCL)**. Cross-stage "review" connections give a
strong, consistent boost across tasks.

---

## 10. Design Choices Cheat-Sheet

When adding feature distillation, you are choosing 4 things:

| Choice | Options | Default that usually works |
|---|---|---|
| **Which layers** | last-of-each-stage / penultimate / all | end-of-stage feature maps |
| **Adapter** | 1×1 conv / linear / none (if dim-agnostic) | 1×1 conv to teacher channels |
| **Transform g** | identity / attention `∑F²` / Gram / affinity | attention (cheap, no adapter) |
| **Distance d** | MSE / L2-normalized MSE / MMD / KL / contrastive | L2-normalized MSE |
| **Combine with** | always keep CE + logit-KD | `L = CE + α·KD + β·L_feat` |

Rule of thumb on **β**: feature losses operate on a different scale than CE/KD. Normalize the
feature (L2 / BN) and tune β so `β·L_feat` is the same order of magnitude as the KD term at
the start of training (the demo uses a small β with normalized attention maps).

---

## 11. Common Pitfalls

1. **Scale mismatch swamps training** — raw-activation MSE (FitNets) can dominate or vanish vs
   CE. Normalize features or schedule β; AT sidesteps this by L2-normalizing the attention map.
2. **Wrong layer pairing** — distilling a teacher stage at a *different spatial resolution*
   than the student stage forces aggressive interpolation and hurts. Pair stages with matching
   (or interpolatable) H×W.
3. **Forgetting the T² factor** in logit-KD makes the KD gradient T² times too small.
4. **Distilling post-ReLU when you meant pre-ReLU** — ReLU discards all negative information;
   OFD shows pre-activation distillation matters.
5. **Teacher in train mode** — always run the teacher in `eval()` with `torch.no_grad()`; a
   teacher with active dropout/BN-update gives noisy, drifting targets.
6. **Expecting miracles when the student is too small** — distillation narrows but does not
   erase the capacity gap; a 100× smaller student won't reach the teacher.
7. **Adapter left in at inference** — the regressor/ABF modules are training-only; the
   deployed student must not include them (or you've changed its FLOPs).

---

## References

- Hinton, Vinyals, Dean — *Distilling the Knowledge in a Neural Network*, 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Romero et al. — *FitNets: Hints for Thin Deep Nets*, ICLR 2015. [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)
- Zagoruyko & Komodakis — *Paying More Attention to Attention (AT)*, ICLR 2017. [arXiv:1612.03928](https://arxiv.org/abs/1612.03928)
- Yim et al. — *A Gift from KD: FSP*, CVPR 2017.
- Huang & Wang — *Like What You Like: Neuron Selectivity Transfer (NST)*, 2017. [arXiv:1707.01219](https://arxiv.org/abs/1707.01219)
- Passalis & Tefas — *Probabilistic Knowledge Transfer (PKT)*, ECCV 2018.
- Park et al. — *Relational Knowledge Distillation (RKD)*, CVPR 2019. [arXiv:1904.05068](https://arxiv.org/abs/1904.05068)
- Tian et al. — *Contrastive Representation Distillation (CRD)*, ICLR 2020. [arXiv:1910.10699](https://arxiv.org/abs/1910.10699)
- Heo et al. — *A Comprehensive Overhaul of Feature Distillation (OFD)*, ICCV 2019. [arXiv:1904.01866](https://arxiv.org/abs/1904.01866)
- Chen et al. — *Distilling Knowledge via Knowledge Review (ReviewKD)*, CVPR 2021. [arXiv:2104.09044](https://arxiv.org/abs/2104.09044)
