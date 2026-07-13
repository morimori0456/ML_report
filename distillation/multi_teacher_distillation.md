---
title: "Multi-Teacher Distillation — Combining Multiple Teachers into One Student"
description: "How to aggregate knowledge from multiple teacher networks — uniform, weighted, adaptive, and gradient-space schemes — including modern foundation-model agglomeration."
---

> Comprehensive guide on knowledge distillation from **multiple** teacher networks: why ensembles of teachers help, how to aggregate their knowledge (uniform / weighted / adaptive / gradient-space), and modern foundation-model agglomeration (RADIO, Theia). Companion notebook: [multi_teacher_distillation_demo.ipynb](multi_teacher_distillation_demo.ipynb). For single-teacher basics see [knowledge_distillation.md](knowledge_distillation.md); for foundation-model teachers see [foundation_model_distillation.md](foundation_model_distillation.md).

A single teacher gives the student one view of the data. Multiple teachers — trained with different seeds, architectures, data subsets, modalities, or even objectives — give complementary views whose errors partially cancel. Multi-teacher distillation (MTKD) is how you compress an *ensemble* into a single deployable model, and more recently, how labs merge several foundation models (CLIP + DINOv2 + SAM) into one general-purpose backbone. The catch: teachers disagree, their logit scales differ, and naive averaging can wash out exactly the "dark knowledge" you wanted to transfer.

---

## Table of Contents
1. [Why Multiple Teachers?](#1-why-multiple-teachers)
2. [Problem Setup and Notation](#2-problem-setup-and-notation)
3. [Aggregation Taxonomy](#3-aggregation-taxonomy)
4. [Fixed-Weight Methods](#4-fixed-weight-methods)
5. [Adaptive-Weight Methods](#5-adaptive-weight-methods)
6. [Gradient-Space and Selection Methods](#6-gradient-space-and-selection-methods)
7. [Feature-Level Multi-Teacher Distillation](#7-feature-level-multi-teacher-distillation)
8. [Foundation-Model Agglomeration](#8-foundation-model-agglomeration)
9. [Teacher Diversity — When Do Extra Teachers Help?](#9-teacher-diversity--when-do-extra-teachers-help)
10. [Practical Recipe](#10-practical-recipe)
11. [Common Pitfalls](#11-common-pitfalls)
12. [References](#12-references)

---

## 1. Why Multiple Teachers?

Three distinct motivations, often conflated:

| Motivation | Setting | Representative work |
|---|---|---|
| **Ensemble compression** | N models of the same task; want ensemble accuracy at 1-model cost | Bucilua et al. 2006; Hinton et al. 2015 |
| **Complementary expertise** | Teachers specialize (classes, domains, modalities, tasks) | Specialist ensembles; cross-modal KD; BEVDistill-style LiDAR→camera |
| **Backbone unification** | Merge several foundation models into one general encoder | AM-RADIO (CVPR 2024), Theia (CoRL 2024) |

The ensemble-compression argument is the original one: an ensemble of N networks averages away variance (bias–variance decomposition), so its soft labels are *better-calibrated* targets than any single teacher's. Hinton's 2015 paper is literally titled "Distilling the Knowledge in an **Ensemble** of Networks" in spirit — the single-teacher case everyone cites is the simplification.

### Key insight
> **The ensemble's value is in error decorrelation, not head count.** N copies of the same teacher (same seed, same data) give exactly the single-teacher soft labels. The gain comes from teachers that are wrong on *different* examples — then the average target is right more often than any individual.

**Why this matters**: Before adding teachers, measure their error correlation. If two teachers make the same mistakes, the second one adds compute cost and zero signal.

---

## 2. Problem Setup and Notation

Given K teachers with logits $z^{(k)}(x)$ and a student with logits $z^{S}(x)$, the generic MTKD loss is

$$\mathcal{L} = (1-\alpha)\,\mathcal{L}_{CE}(y, \sigma(z^S)) \;+\; \alpha T^2 \sum_{k=1}^{K} w_k(x)\; \mathrm{KL}\!\left(\sigma(z^{(k)}/T)\;\|\;\sigma(z^S/T)\right)$$

where $\sigma$ is softmax, $T$ is temperature, and $w_k(x)$ are teacher weights with $\sum_k w_k(x) = 1$. Everything in this survey is a choice of $w_k(x)$:

- **Uniform**: $w_k = 1/K$ (average the teachers)
- **Fixed**: $w_k \propto$ teacher validation accuracy (constant per teacher)
- **Adaptive**: $w_k(x)$ depends on the sample — entropy, confidence, agreement with the label
- **Selection**: $w_k(x) \in \{0, 1\}$ — pick one teacher per sample (hard routing)

An equivalent alternative aggregates **logits before** the KL instead of KL terms after:

$$\bar{z}(x) = \sum_k w_k(x)\, z^{(k)}(x), \qquad \mathcal{L}_{KD} = T^2\,\mathrm{KL}\!\left(\sigma(\bar{z}/T)\,\|\,\sigma(z^S/T)\right)$$

Averaging logits then softmaxing produces a *sharper* target than averaging the softmax outputs (a geometric-mean vs arithmetic-mean effect); averaging probabilities preserves multi-modality when teachers disagree. Both appear in the literature — be explicit about which you use.

---

## 3. Aggregation Taxonomy

```
Multi-Teacher KD
├── Logit-level
│   ├── Uniform average            (vanilla ensemble KD, Hinton 2015)
│   ├── Fixed weights              (accuracy-weighted)
│   ├── Adaptive weights
│   │   ├── Entropy-based          (EBKD: low-entropy teacher = confident = trust more)
│   │   ├── Label-aware confidence (CA-MKD: CE(teacher, GT) → weight)
│   │   └── Learned gating         (AMTML-KD: latent instance-teacher affinity)
│   └── Sample-wise selection      (Reinforced Teacher Selection, RL-based)
├── Gradient-level
│   └── AEKD: resolve teacher conflicts in gradient space
├── Feature-level
│   ├── Per-teacher adapters + weighted hint losses (AMTML-KD multi-level)
│   └── CA-MKD feature term weighted by same confidence
└── Backbone unification (multi-objective feature matching)
    ├── AM-RADIO: CLIP + DINOv2 + SAM → one backbone
    └── Theia: off-the-shelf VFMs → robot-learning encoder
```

| Strategy | $w_k(x)$ | Extra cost | When it wins |
|---|---|---|---|
| Uniform average | $1/K$ | none | teachers comparable & decorrelated |
| Accuracy-weighted | fixed per teacher | one val pass | teacher quality varies a lot |
| Entropy-based (EBKD) | per sample, from teacher entropy | negligible | some teachers confidently wrong off-domain |
| CA-MKD | per sample, needs GT label | negligible (train only) | noisy/weak teachers mixed with strong |
| AMTML-KD | learned per sample | gating net | large K, heterogeneous teachers |
| AEKD | implicit, in gradient space | per-teacher grads | strong teacher disagreement |
| RL selection | hard $\{0,1\}$ | RL training loop | very heterogeneous teacher pool |

---

## 4. Fixed-Weight Methods

### 4.1 Uniform averaging (the baseline that refuses to die)

$$\bar{q}(x) = \frac{1}{K}\sum_k \sigma\!\left(z^{(k)}(x)/T\right)$$

This is the correct first thing to try. It inherits the classic ensemble guarantee: if teachers have equal error rates and their errors are independent, the majority target is exponentially better in K. In practice teachers trained from different seeds on the same data are heavily correlated, so gains saturate around K = 3–5.

### 4.2 Accuracy-weighted

$$w_k \propto \exp(\mathrm{acc}_k / \tau_w)$$

or simply normalized validation accuracy. One line of code, and it prevents a clearly weaker teacher (e.g., an older production model kept for diversity) from dragging the average target toward its mistakes. The limitation is obvious: the weights are global, but teacher reliability is *local* — a LiDAR-trained teacher is excellent at night scenes and useless on a camera-only domain where LiDAR features don't transfer.

```python
# Fixed-weight ensemble target (logit-level averaging)
import torch
import torch.nn.functional as F

def ensemble_target(teacher_logits: list[torch.Tensor],
                    weights: torch.Tensor, T: float) -> torch.Tensor:
    # teacher_logits: K tensors of [B, C]; weights: [K], sums to 1
    z = torch.stack(teacher_logits, dim=0)            # [K, B, C]
    z_bar = (weights.view(-1, 1, 1) * z).sum(dim=0)   # [B, C]
    return F.softmax(z_bar / T, dim=-1)
```

**Why this matters**: fixed weights solve the "one bad teacher" problem but not the "every teacher has blind spots" problem. That requires per-sample weights — the next section.

---

## 5. Adaptive-Weight Methods

### 5.1 Entropy-based weighting (EBKD)

Trust the teacher that is confident on *this* sample. With teacher entropy $H_k(x) = -\sum_c q_c^{(k)}\log q_c^{(k)}$:

$$w_k(x) = \frac{\exp(-H_k(x))}{\sum_j \exp(-H_j(x))}$$

Cheap and label-free (works on unlabeled transfer sets). Its failure mode is a teacher that is **confidently wrong** — entropy cannot distinguish justified confidence from miscalibrated overconfidence, and off-domain inputs notoriously produce confident garbage.

### 5.2 Confidence-Aware Multi-teacher KD (CA-MKD, ICASSP 2022)

Fixes exactly that failure mode by using the ground-truth label during training. Teacher weight is derived from the cross-entropy between the teacher prediction and the GT one-hot:

$$w_k(x) = \frac{\exp\!\left(-\mathcal{L}_{CE}(y, q^{(k)}(x))\right)}{\sum_j \exp\!\left(-\mathcal{L}_{CE}(y, q^{(j)}(x))\right)}$$

A teacher that assigns low probability to the true class on this sample is down-weighted *on this sample*, regardless of how confident it is. CA-MKD applies the same weights to intermediate-feature losses, so the student also imitates the features of whichever teacher "understood" the example. Requires labels — so it does not apply to purely unlabeled distillation sets.

```python
def ca_mkd_weights(teacher_logits: list[torch.Tensor],
                   y: torch.Tensor) -> torch.Tensor:
    # Per-sample teacher weights from agreement with ground truth
    ce = torch.stack([F.cross_entropy(z, y, reduction='none')
                      for z in teacher_logits], dim=0)   # [K, B]
    return F.softmax(-ce, dim=0)                          # [K, B]
```

### 5.3 Learned gating (AMTML-KD, IJCAI 2019)

Adaptive Multi-Teacher Multi-Level KD learns an **instance–teacher affinity**: a small gating module takes the sample's (student or pooled teacher) representation and emits $w_k(x)$, trained end-to-end with the distillation loss. It also distills at multiple feature levels, with per-level weights. More flexible than closed-form weights, at the cost of another module to tune — and the gate can collapse onto one teacher if not regularized (add an entropy bonus on $w$, exactly like mixture-of-experts load balancing).

### Key insight
> **Adaptive weighting is a per-sample router, and inherits router pathologies.** Collapse onto one teacher, oscillation early in training, and miscalibration off-domain are the same failure modes seen in MoE gating. The MoE toolbox (entropy regularization, warmup with uniform weights) transfers directly.

---

## 6. Gradient-Space and Selection Methods

### 6.1 AEKD — Adaptive Ensemble KD in gradient space (NeurIPS 2020)

When teachers disagree, their KD gradients on the student can point in *conflicting directions*, and averaging probabilities blurs the target into something no teacher believes (high-entropy mush between two sharp but different modes). AEKD (Du et al.) instead treats each teacher's KD loss as one objective of a multi-objective problem and seeks a descent direction in the convex hull of teacher gradients — conceptually the same machinery as MGDA (Multiple Gradient Descent Algorithm) in multi-task learning:

$$\min_{\lambda \in \Delta^K} \left\| \sum_k \lambda_k \, g_k \right\|^2, \qquad g_k = \nabla_\theta \, \mathrm{KL}\!\left(q^{(k)} \| q^S\right)$$

The resulting direction reduces no teacher's loss at the expense of violently increasing another's. Cost: one backward pass per teacher (or gradients w.r.t. shared student outputs only, which is the cheap standard trick).

### 6.2 Reinforced teacher selection (AAAI 2021)

Yuan et al. formulate per-sample teacher choice as an RL policy: state = sample features + training progress, action = pick a teacher, reward = student improvement. Hard selection avoids target blurring entirely and handles very heterogeneous pools (teachers of different tasks), but the RL loop adds real engineering cost; in most reported settings, well-tuned soft weighting gets most of the benefit.

**Why this matters**: gradient-space and selection methods matter precisely when disagreement is *structural* (different modalities, different label spaces) rather than noise. For same-task same-data teacher pools, CA-MKD-style weighting is usually enough.

---

## 7. Feature-Level Multi-Teacher Distillation

Everything from [feature distillation](feature_distillation_why.md) applies, with one new wrinkle: teachers have **different feature spaces** (dimensions, spatial layouts, statistics). The standard pattern:

1. One **adapter** (1×1 conv / linear projection) per teacher, mapping student features into each teacher's space (or vice versa).
2. Per-teacher hint losses, combined with the same weights $w_k(x)$ as the logit term (CA-MKD does exactly this).
3. Normalize features per teacher (LayerNorm or per-channel standardization) before the loss — teacher feature magnitudes differ wildly, and without normalization the largest-scale teacher silently dominates the loss.

$$\mathcal{L}_{feat} = \sum_k w_k(x) \left\| \phi_k(F^S) - \mathrm{norm}(F^{(k)}) \right\|_2^2$$

This is also where **heterogeneous-architecture** pools (CNN teacher + ViT teacher) get resolved: match at the most abstract representation level that both share — pooled global features or attention maps — rather than forcing spatial alignment between incompatible layouts.

---

## 8. Foundation-Model Agglomeration

The most consequential modern use of MTKD: merging several vision foundation models (VFMs) into one backbone. Labels are absent; the "task" is *feature matching against each teacher simultaneously* on web-scale unlabeled images.

### AM-RADIO (NVIDIA, CVPR 2024)

Distills **CLIP + DINOv2 + SAM** into a single student (RADIO). Per teacher: an adapter head on the shared student trunk, matching that teacher's summary token (cosine loss) and spatial features (smooth-L1). Findings that generalize beyond the paper:

- The student can **exceed individual teachers** on some benchmarks — the teachers regularize each other's weaknesses (CLIP's poor dense features, DINOv2's lack of language alignment).
- Teacher-specific heads + shared trunk is the load-bearing design: the trunk learns the intersection of what all teachers know; heads absorb the incompatibilities.
- Balancing losses across teachers matters more than the exact loss form; feature-magnitude normalization per teacher is mandatory.

### Theia (CoRL 2024)

Same recipe aimed at robot learning: distills off-the-shelf VFMs (CLIP, DINOv2, SAM, Depth-Anything, ...) into a compact encoder for visuomotor policies, showing better downstream robot-manipulation performance than any single teacher's features at a fraction of the inference cost. Directly relevant to autonomous driving: the recipe transfers to "distill {CLIP for semantics + DINOv2 for geometry-ish dense features + a detection teacher} into one drivable backbone."

| Aspect | Classic MTKD (Sections 4–6) | FM agglomeration |
|---|---|---|
| Teachers | same task, same label space | different objectives, no shared labels |
| Signal | logits (+ features) | features only, per-teacher heads |
| Weighting problem | which teacher is right | how to balance incommensurable losses |
| Data | task dataset / transfer set | large unlabeled corpus |
| Student goal | match/exceed ensemble accuracy | inherit all teachers' downstream abilities |

### Key insight
> **When teachers solve different tasks, "weighting" becomes "loss balancing", and adapters become mandatory.** There is no notion of a per-sample correct teacher; every teacher is right about its own aspect. The architecture (shared trunk, per-teacher heads) does the reconciliation that $w_k(x)$ does in the classic setting.

---

## 9. Teacher Diversity — When Do Extra Teachers Help?

The empirical rules of thumb, consistent across the MTKD literature:

1. **Diversity source ranking** (roughly, weakest to strongest): different seeds < different data subsets/augmentation < different architectures < different modalities/objectives. Seed-only ensembles decorrelate the least but are cheapest.
2. **Diminishing returns are fast** for homogeneous pools: most of the gain arrives by K = 3; beyond K = 5 the averaged target barely moves.
3. **A much weaker teacher can hurt** under uniform averaging (it pulls the target toward its errors) but can still *help* under adaptive weighting — it contributes on the subset where it happens to be right (this is the CA-MKD selling point).
4. **Capacity gap compounds**: an ensemble target is effectively a "larger teacher" — sharper and more structured than any member. A tiny student may track a single teacher better than the ensemble ([TAKD](foundation_model_distillation.md) mitigations apply here too).
5. Measure diversity directly: **error correlation matrix** on a validation set, or pairwise prediction disagreement rate. If mean pairwise disagreement is below a few percent, extra teachers are decorative.

---

## 10. Practical Recipe

A production-ordered checklist:

1. **Precompute teacher outputs offline.** K forward passes per step is the real cost of MTKD. Cache logits (and pooled features) to disk; training then costs the same as single-teacher KD. Only skip this if you need teacher-side augmentation consistency.
2. **Start with uniform logit averaging**, temperature per the usual KD sweep (T ∈ {2, 4}). This is the baseline everything must beat.
3. **Standardize per-teacher logits** (zero-mean, unit-variance per sample — the [Logit Standardization](advanced_kd_practical.md) trick) before averaging when teachers differ in architecture/training; otherwise the sharpest-logit teacher dominates.
4. **Add CA-MKD weighting** if you have labels and teacher quality is uneven. It is ~5 lines (Section 5.2) and rarely hurts.
5. **Weight features too** if you distill features; use one adapter per teacher + per-teacher normalization.
6. **Check the error-correlation matrix** before growing the pool. Add a teacher only if it disagrees with the pool on ≥ 5–10% of samples *and* is competitive on its disagreement region.
7. **For heterogeneous/FM teachers**: shared trunk + per-teacher heads, feature-space losses, balance losses so per-teacher gradient norms are comparable.
8. **Evaluate against the ensemble**, not just the best single teacher — the ensemble accuracy is the ceiling you are trying to compress into one model.

---

## 11. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Averaging probabilities of *disagreeing* teachers | high-entropy blurred target; student learns neither mode | adaptive weights / selection; or logit-level averaging; AEKD if conflict is structural |
| Unstandardized logit scales across teachers | one teacher dominates target regardless of weights | per-sample logit standardization before averaging |
| Redundant teacher pool (seed clones) | K× compute, ≈ single-teacher accuracy | check error correlation first; diversify data/architecture |
| Weak teacher under uniform weights | student below single-best-teacher baseline | accuracy weights minimum; CA-MKD preferred |
| Gating collapse (learned weights → one teacher) | adaptive method ≈ single-teacher KD | entropy regularization on $w$; uniform-weight warmup |
| Online teachers in the training loop | GPU memory explosion, slow steps | precompute & cache teacher outputs offline |
| Feature-loss scale mismatch across teachers | one teacher's feature term dominates | per-teacher normalization + gradient-norm balancing |
| Ensemble target too sharp for a tiny student | multi-teacher *worse* than single teacher | raise T; TAKD intermediate assistant; smaller K |
| Using entropy weights off-domain | confidently-wrong teacher gets high weight | CA-MKD (needs labels) or OOD filtering of the transfer set |

---

## 12. References

- Bucilua, Caruana, Niculescu-Mizil. *Model Compression*. KDD 2006.
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network*. NeurIPS-W 2015. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- You, Xu, Xu, Tao. *Learning from Multiple Teacher Networks*. KDD 2017.
- Liu, Zhang, Wang. *Adaptive Multi-Teacher Multi-level Knowledge Distillation* (AMTML-KD). Neurocomputing 2020. [arXiv:2103.04062](https://arxiv.org/abs/2103.04062)
- Du et al. *Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space* (AEKD). NeurIPS 2020.
- Kwon et al. *Adaptive Knowledge Distillation Based on Entropy* (EBKD). ICASSP 2020.
- Yuan et al. *Reinforced Multi-Teacher Selection for Knowledge Distillation*. AAAI 2021. [arXiv:2012.06048](https://arxiv.org/abs/2012.06048)
- Zhang, Chen, Wang. *Confidence-Aware Multi-Teacher Knowledge Distillation* (CA-MKD). ICASSP 2022. [arXiv:2201.00007](https://arxiv.org/abs/2201.00007)
- Ranzinger et al. *AM-RADIO: Agglomerative Vision Foundation Model — Reduce All Domains Into One*. CVPR 2024. [arXiv:2312.06709](https://arxiv.org/abs/2312.06709)
- Shang et al. *Theia: Distilling Diverse Vision Foundation Models for Robot Learning*. CoRL 2024. [arXiv:2407.20179](https://arxiv.org/abs/2407.20179)
- Mirzadeh et al. *Improved Knowledge Distillation via Teacher Assistant* (TAKD). AAAI 2020. [arXiv:1902.03393](https://arxiv.org/abs/1902.03393)
- Sun et al. *Logit Standardization in Knowledge Distillation*. CVPR 2024. [arXiv:2403.01427](https://arxiv.org/abs/2403.01427)
