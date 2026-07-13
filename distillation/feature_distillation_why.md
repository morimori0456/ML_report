---
title: "Feature Distillation — Why Intermediate Features Beat Final Logits"
description: "Why matching intermediate feature maps transfers more knowledge than matching only final logits, and how to do it well."
---

> Companion notebook: [feature_distillation_why.ipynb](https://github.com/morimori0456/ML_report/blob/main/distillation/feature_distillation_why.ipynb) — runs 5 methods on `sklearn digits`, measures CKA, and visualises attention maps.
> See also [knowledge_distillation.md](knowledge_distillation.md) for the full method reference and [distillation_methods_survey.md](distillation_methods_survey.md) for detection/NLP recipes.

This document answers one targeted question:

> **Why does matching intermediate feature maps transfer more knowledge than matching only the final logits, and how do you do it well?**

---

## Table of Contents
1. [The Core Argument in One Picture](#1-the-core-argument-in-one-picture)
2. [Information Bottleneck: Logits vs Features](#2-information-bottleneck-logits-vs-features)
3. [Gradient Path Analysis](#3-gradient-path-analysis)
4. [The Representation-Matching Problem](#4-the-representation-matching-problem)
5. [Five Methods — What Exactly Gets Matched](#5-five-methods--what-exactly-gets-matched)
6. [Empirical Evidence: Accuracy and CKA](#6-empirical-evidence-accuracy-and-cka)
7. [When Logit KD Is Sufficient](#7-when-logit-kd-is-sufficient)
8. [Design Cheatsheet](#8-design-cheatsheet)
9. [Common Pitfalls](#9-common-pitfalls)
10. [References](#10-references)

---

## 1. The Core Argument in One Picture

```
Teacher CNN
  ├── Conv Block 1  [B, 32, H, W]  ─┐  rich spatial + channel signal
  ├── Conv Block 2  [B, 64, H, W]  ─┤  ← Feature distillation taps here
  ├── Pool + Flatten               │    (thousands of values)
  └── Linear head  [B, 10]        ─┘  ← Logit KD taps here
                                        (only 10 values per sample)
```

Every number the teacher computes about an image — where it looks, how strongly each neuron fires, which spatial regions matter — is encoded in the intermediate feature maps. The final logit vector is a **compressed summary** of that rich signal: it tells you which class wins, but has already discarded spatial structure, multi-scale patterns, and per-channel selectivity.

Distilling only the logits means the student must reverse-engineer the teacher's reasoning from its conclusion alone. Distilling intermediate features gives the student a direct view of the reasoning process itself.

---

## 2. Information Bottleneck: Logits vs Features

Consider a teacher trained on a 10-class task (e.g., digit recognition):

| Signal | Dimensionality | What it encodes |
|---|---|---|
| Final logit vector | **10** | Class-conditional probability mass; inter-class similarity (dark knowledge) |
| Stage-2 feature map (64ch × 8 × 8) | **4,096** | Where the network attends (spatial), what patterns fire (channel), how strongly |

The ratio is **410×** more information per sample in the features.

### What logits can transfer (dark knowledge)

Hinton et al. (2015) showed that the teacher's soft distribution over classes encodes inter-class similarity: `p("4") ≈ 0.01 × p("9")` signals that 4 and 9 are visually similar. This is valuable and completely invisible in the one-hot label.

### What logits cannot transfer

1. **Spatial location**: Which pixels triggered the response? Logits have no spatial axis.
2. **Multi-scale patterns**: Whether a feature fires at a corner, an edge, or a global shape is lost after global pooling.
3. **Per-channel selectivity**: Which filters activated (texture detectors, curve detectors) is collapsed into a scalar per class.
4. **Negative activations (pre-ReLU)**: Logits only carry post-nonlinearity information.

These missing signals matter most when:
- The student is **much smaller** than the teacher (capacity gap → hard to infer the reasoning from the conclusion).
- The task requires **spatial precision** (detection, segmentation, depth estimation).
- The teacher has learned **complex intermediate representations** (many stages, large receptive fields).

---

## 3. Gradient Path Analysis

Training with only logit KD:

```
∂L_KD / ∂W_conv = ∂L_KD/∂logits · ∂logits/∂head · ∂head/∂conv
                                     ↑ 10-dim bottleneck
```

The gradient must pass through a 10-dimensional linear bottleneck before reaching the convolutional layers. For a C-class head projecting from a D-dimensional feature, only the top-C directions in feature space receive a strong gradient signal; the remaining `D − C` directions are invisible to the logit loss.

Training with feature distillation at layer `l`:

```
∂L_feat / ∂W_conv_l = ∂L_feat/∂F_l · ∂F_l/∂W_conv_l
                        ↑ H×W×C-dim signal; no bottleneck
```

Every spatial location, every channel, every activation can carry an independent gradient. The convolutional layers receive a **full-rank** supervision signal rather than a rank-10 projection of it.

> **Key insight**: Feature distillation does not replace CE or logit KD — it adds high-bandwidth supervision to lower layers that the logit path reaches only weakly. Always combine: `L = CE + α·KD_logit + β·L_feat`.

---

## 4. The Representation-Matching Problem

You want `F_student ∈ ℝ^{C_S×H_S×W_S}` to carry the same information as `F_teacher ∈ ℝ^{C_T×H_T×W_T}`. Three sub-problems arise:

### 4.1 Dimension mismatch

`C_S ≠ C_T` in virtually every practical case (the student is smaller). Solutions:

| Approach | How | Used by |
|---|---|---|
| **Adapter / regressor** | 1×1 conv `r: C_S → C_T` trained jointly, discarded at inference | FitNets |
| **Dimension-agnostic transform** | Collapse channels before comparing (e.g., `∑_c F_c²`) | AT, FSP |
| **Relation-based** | Compare pairwise distances/angles; scalars are always compatible | RKD, CRD |

### 4.2 Where to distill

End-of-stage feature maps (after each spatial downsampling) are the standard choice. They:
- Have matched spatial resolution (easy to compare after optional interpolation)
- Aggregate information from all earlier layers in that stage
- Keep the pairing tractable (one loss per stage rather than one per layer)

### 4.3 What transform to apply

Raw activations (FitNets) impose the strictest constraint: match exact values. Attention maps (AT) impose a softer constraint: match where the network looks. The softer the constraint, the more freedom the student retains in *how* it represents that region — which matters when the student is much smaller.

---

## 5. Five Methods — What Exactly Gets Matched

### 5.1 Hinton KD (Logit baseline)

$$\mathcal{L}_{\text{KD}} = T^2 \cdot \text{KL}\!\left(\sigma(z_s/T)\,\Vert\,\sigma(z_t/T)\right)$$

- **What is matched**: Final class probabilities softened by temperature `T`.
- **Why T²**: The softmax gradient shrinks by ~1/T²; the T² factor restores gradient magnitude to the same scale as CE.
- **What is not matched**: Everything spatial and multi-channel in the intermediate layers.

### 5.2 DKD — Decoupled Knowledge Distillation (CVPR 2022)

Standard KD implicitly couples two orthogonal signals:
- **TCKD** (Target-Class KD): the binary distribution {target class, all others} — encodes sample difficulty.
- **NCKD** (Non-Target-Class KD): the distribution among non-target classes — this is the actual dark knowledge.

The coupling factor `(1 − p_t^T)` suppresses NCKD precisely on the most informative samples (those where the teacher is confident). DKD decouples them:

$$\mathcal{L}_{\text{DKD}} = \alpha \cdot \text{TCKD} + \beta \cdot \text{NCKD}, \quad \beta > 1$$

Setting `β = 2` (used in the companion notebook) un-suppresses the non-target distribution. **Best logit-only method** — pure drop-in replacement, no feature access needed.

### 5.3 FitNets — Feature MSE (ICLR 2015)

$$\mathcal{L}_{\text{FitNets}} = \tfrac{1}{2}\|\,r(F_s) - F_t\,\|_2^2$$

- **`r`**: 1×1 conv adapter `C_S → C_T`, trained jointly with the student, **discarded at inference**.
- **What is matched**: Raw activation values after projecting to the teacher's channel space.
- **Strength**: Strictest, most information-dense constraint.
- **Weakness**: Raw activations are scale-sensitive; MSE can dominate the loss. The adapter must be initialised carefully (small gain Xavier) to avoid initial gradient explosion.

### 5.4 AT — Attention Transfer (ICLR 2017)

Collapse the feature map's channel dimension into a spatial attention map, then match:

$$A(F) = \sum_{c=1}^{C} F_c^2, \quad \mathcal{L}_{\text{AT}} = \left\|\frac{\text{vec}(A(F_s))}{\|\text{vec}(A(F_s))\|_2} - \frac{\text{vec}(A(F_t))}{\|\text{vec}(A(F_t))\|_2}\right\|_2^2$$

- **No adapter needed**: `∑_c F_c²` maps any number of channels to a single H×W map.
- **What is matched**: Where the network concentrates its response — not the exact activation values.
- **Why softer is often better**: The student is free to choose *how* it activates each region, as long as the spatial distribution matches. This is a less constrained, more transferable target — especially when `C_S ≪ C_T`.
- **Scale compensation**: Because the loss is on L2-normalised maps (bounded in [0, 4]), a large `feat_w` (~50) is typically needed to match the KD loss scale.

### 5.5 Summary Table

| Method | Transform g(F) | Adapter? | Scale-sensitive? | Constraint strength |
|---|---|---|---|---|
| Hinton KD | identity on logits | — | — | — (logit only) |
| DKD | decoupled logit | — | — | — (logit only) |
| FitNets | identity | **yes** | yes | strongest |
| AT | `∑_c F_c²`, L2-norm | no | no (normalised) | softer, spatial-only |
| CRD | pairwise affinity (contrastive) | no | no | relation-level |

---

## 6. Empirical Evidence: Accuracy and CKA

The companion notebook trains a teacher (64ch, ~29K params) and student (16ch, ~4K params — **7.6× fewer parameters**) on `sklearn digits` with the student receiving only 150 labels. Results on the held-out test set:

| Method | Test Accuracy | Δ vs Scratch | CKA with Teacher |
|---|---|---|---|
| Teacher (1347 labels) | 0.991 | — | 1.000 |
| Scratch (150 labels) | 0.944 | — | 0.958 |
| Logit KD | 0.991 | +0.047 | 0.977 |
| DKD | 0.993 | **+0.049** | 0.979 |
| FitNets + KD | 0.989 | +0.044 | 0.980 |
| AT + KD | 0.993 | **+0.049** | **0.982** |

### Reading the CKA column

**CKA (Centered Kernel Alignment)** measures how similar two feature spaces are, regardless of dimensionality:

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K,K) \cdot \text{HSIC}(L,L)}}, \quad K = XX^\top,\ L = YY^\top$$

CKA = 0: unrelated representations. CKA = 1: identical structure.

Key observations:
1. **FitNets and AT have higher CKA** than either logit-only method (Logit KD, DKD), even when accuracy is similar. This confirms that feature distillation genuinely aligns internal representations, not just output behaviour.
2. **AT achieves the highest CKA** despite using a softer (spatial-only) target, because the L2-normalised attention loss provides a stable, consistent gradient signal throughout training.
3. The CKA gap is clearest at lower accuracy regimes (harder tasks, larger capacity gaps) — this is where feature distillation pays off most in practice.

---

## 7. When Logit KD Is Sufficient

Feature distillation adds complexity (adapters, weight tuning, feature access). Logit KD is sufficient when:

| Condition | Why logits are enough |
|---|---|
| **Easy task / large student** | The student can match teacher behaviour from logit supervision alone |
| **No intermediate access** | API-only or black-box teacher |
| **Classification only** | No spatial precision required; dark knowledge is the main signal |
| **DKD is available** | With decoupling, logit KD approaches feature KD accuracy on simple tasks |

Feature distillation becomes essential when:

| Condition | Why features are needed |
|---|---|
| **Dense prediction** (detection, segmentation) | Logits carry zero spatial information |
| **Large capacity gap** | Student can't infer teacher reasoning from its conclusion alone |
| **Hard task / low accuracy ceiling** | Teacher's feature activations discriminate what logits cannot |
| **Representation quality matters** | Downstream fine-tuning, embedding, retrieval |

---

## 8. Design Cheatsheet

When adding feature distillation, make four decisions:

| Decision | Options | Default that works |
|---|---|---|
| **Which layers** | End-of-stage (after pool) | Stage-2 feature map (one loss) |
| **Adapter** | 1×1 conv / none if dim-agnostic | 1×1 conv (FitNets) or skip (AT) |
| **Transform g** | identity / `∑F²` (attention) / Gram / affinity | AT — stable, no adapter |
| **feat_w** | tune so β·L_feat ~ α·L_KD at epoch 1 | AT: ~50; FitNets: ~0.5 |

**Combined loss**:
$$\mathcal{L} = \underbrace{(1-\alpha)\,\text{CE}(y, s)}_{\text{labeled only}} + \underbrace{\alpha\,T^2\,\text{KL}(s_T \| t_T)}_{\text{full pool}} + \underbrace{\beta\,\mathcal{L}_{\text{feat}}}_{\text{full pool}}$$

Typical values: `α = 0.7–0.9`, `β` tuned per method, `T = 4`.

---

## 9. Common Pitfalls

1. **Scale mismatch kills training**: Raw-activation MSE (FitNets) can be orders of magnitude larger than CE. Initialise the adapter with small weight gain and validate that `β·L_feat` is comparable to `α·L_KD` after one step.

2. **Mismatched spatial resolution**: Distilling a teacher stage at a different H×W than the student stage forces aggressive interpolation and breaks spatial alignment. Always pair stages of matching resolution.

3. **Teacher in train mode**: A teacher with active dropout and updating BatchNorm statistics produces noisy, shifting targets. Always call `teacher.eval()` and `torch.no_grad()`.

4. **Forgetting T² in logit KD**: Omitting the T² scale factor makes the KD gradient T² times too small relative to CE, effectively disabling the distillation signal.

5. **Including the adapter at inference**: The 1×1 conv adapter (FitNets) is a training-only module. Leaving it in the deployed student changes its parameter count and FLOPs.

6. **Distilling post-ReLU when pre-ReLU matters**: ReLU discards all negative activations. OFD (§9 of the full guide) shows that distilling pre-activation features, with a margin-ReLU on the teacher side, transfers substantially more information.

7. **Expecting miracles when the student is too small**: A 100× smaller student cannot represent the teacher's diversity. Distillation narrows but does not erase the capacity gap; consider a Teacher Assistant (TAKD) if the gap is extreme.

---

## 10. References

- Hinton, Vinyals, Dean — *Distilling the Knowledge in a Neural Network*, 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Romero et al. — *FitNets: Hints for Thin Deep Nets*, ICLR 2015. [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)
- Zagoruyko & Komodakis — *Paying More Attention to Attention*, ICLR 2017. [arXiv:1612.03928](https://arxiv.org/abs/1612.03928)
- Zhao et al. — *Decoupled Knowledge Distillation*, CVPR 2022. [arXiv:2203.08679](https://arxiv.org/abs/2203.08679)
- Tian et al. — *Contrastive Representation Distillation (CRD)*, ICLR 2020. [arXiv:1910.10699](https://arxiv.org/abs/1910.10699)
- Heo et al. — *A Comprehensive Overhaul of Feature Distillation (OFD)*, ICCV 2019. [arXiv:1904.01866](https://arxiv.org/abs/1904.01866)
- Kornblith et al. — *Similarity of Neural Network Representations Revisited (CKA)*, ICML 2019. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
