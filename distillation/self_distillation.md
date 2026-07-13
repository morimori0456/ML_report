---
title: "Self-Distillation — Why a Model Teaching Itself Gets Better"
description: "A deep dive into self-distillation's four variants and the three leading theoretical explanations for why a model teaching itself gains accuracy."
---

> A deep dive into self-distillation: distilling a model into an identical (or smaller-but-same-family)
> architecture and gaining accuracy despite adding zero new information. Covers the four main variants
> (born-again, in-network, temporal/EMA, self-supervised), the three leading theoretical explanations,
> and practical recipes. See [self_distillation_demo.ipynb](self_distillation_demo.ipynb) for
> experiments reproducing the born-again effect under label noise — including *which channel*
> the gain actually flows through — and the regularization-amplification theory in closed form.

Standard knowledge distillation (see [knowledge_distillation.md](knowledge_distillation.md)) makes
sense on its face: a big teacher compresses knowledge into a small student. Self-distillation breaks
that intuition — teacher and student have the *same* architecture, the same data, and the student
still outperforms the teacher. No new parameters, no new labels, no new data, yet a reliable accuracy
gain that has been reproduced across CNNs, transformers, and even kernel regression. Understanding
*why* this works reveals what soft labels actually carry, and the same mechanism underlies systems
you already use: EMA teachers, DINO, and modern LLM self-improvement pipelines.

---

## Table of Contents

1. [The Setup and the Paradox](#1-the-setup-and-the-paradox)
2. [The Four Variants](#2-the-four-variants)
3. [Why It Works — Three Theories](#3-why-it-works--three-theories)
4. [Born-Again Networks in Detail](#4-born-again-networks-in-detail)
5. [In-Network Self-Distillation (BYOT)](#5-in-network-self-distillation-byot)
6. [Temporal Self-Distillation — EMA Teachers and DINO](#6-temporal-self-distillation--ema-teachers-and-dino)
7. [Self-Distillation in LLMs](#7-self-distillation-in-llms)
8. [Practical Recipes](#8-practical-recipes)
9. [Common Pitfalls](#9-common-pitfalls)
10. [References](#10-references)

---

## 1. The Setup and the Paradox

### Definition

Train a teacher $f_T$ on dataset $\mathcal{D} = \{(x_i, y_i)\}$. Then train a student $f_S$ with the
**same architecture** on the same data, replacing (or mixing) the hard labels with the teacher's
soft predictions:

$$
\mathcal{L}_S = (1-\lambda)\, \mathrm{CE}\big(y,\; \sigma(z_S)\big)
\;+\; \lambda\, T^2\, \mathrm{KL}\big(\sigma(z_T / T)\;\|\;\sigma(z_S / T)\big)
$$

where $z$ are logits, $T$ is the temperature, and $\lambda$ balances hard and soft targets. In
standard KD the capacity gap justifies the transfer. In self-distillation, $f_S$ and $f_T$ have
identical capacity — and yet, empirically:

$$
\mathrm{Acc}(f_S) > \mathrm{Acc}(f_T)
$$

Furlanello et al. (2018) report CIFAR-100 gains of 1–3 pp for identical architectures, with the
effect repeating (diminishingly) over multiple generations.

### Key insight

> **The information the student gains is not in the labels — it is in the *geometry of the
> teacher's mistakes*.** The soft label for a dog image that reads 0.7 dog / 0.25 wolf / 0.05 car
> encodes inter-class similarity structure ("dark knowledge") that the one-hot label destroys.
> Self-distillation is a mechanism for feeding a model's own learned similarity structure back
> into its training signal.

**Why this matters**: self-distillation is the cleanest available probe of what soft targets do,
because the capacity-compression explanation is removed by construction. Whatever survives this
ablation — regularization, ensembling, similarity structure — is the true active ingredient of all
distillation methods.

---

## 2. The Four Variants

| Variant | Teacher | Student | Trained | Canonical paper |
|---|---|---|---|---|
| Born-again (sequential) | Converged gen $k$ | Fresh gen $k{+}1$, same arch | Sequentially, full runs | Furlanello et al. 2018 (BAN) |
| In-network (BYOT) | Deepest exit of the net | Shallow auxiliary exits of the *same* net | Jointly, one run | Zhang et al. 2019 |
| Temporal / EMA | EMA of student weights (or earlier snapshot) | Current student | Jointly, one run | Tarvainen & Valpola 2017 (Mean Teacher) |
| Self-supervised | EMA network + centering, no labels at all | Online network | Jointly, one run | Caron et al. 2021 (DINO) |

Two axes organize these: **when the teacher exists** (before training vs during training) and
**where it lives** (separate run, same network, weight average). The variants look different but
Section 3's theory applies to all of them: each constructs a smoothed, ensembled, or temporally
averaged version of the model and regresses the model toward it.

**Why this matters**: recognizing the shared skeleton lets you transfer tricks between variants —
DINO's centering trick prevents the same collapse that plagues naive BAN chains, and BYOT's
deep-supervision view explains why auxiliary-exit gains survive even without the KD term.

---

## 3. Why It Works — Three Theories

### 3.1 Regularization amplification (Mobahi, Farajtabar, Bartlett 2020)

For kernel ridge regression, self-distillation has a closed form. With kernel eigenbasis
$\{\phi_j\}$ and eigenvalues $\{d_j\}$, one ridge fit shrinks the coefficient of basis function
$j$ by a factor $\frac{d_j}{d_j + c}$ (regularizer $c$). Fitting the *previous fit's outputs*
applies the shrinkage again — after $t$ rounds of self-distillation the effective shrinkage is:

$$
\left(\frac{d_j}{d_j + c}\right)^{t}
$$

Small-eigenvalue (high-frequency, wiggly) components decay geometrically faster than
large-eigenvalue (smooth) ones. Consequences, proven in the paper and reproduced in the notebook:

- **Few rounds help**: progressive sparsification of the basis acts as a stronger regularizer,
  improving generalization when the single-fit solution was under-regularized.
- **Many rounds hurt**: the shrinkage eventually eats signal components too — the solution
  collapses toward zero and *underfits*. Self-distillation cannot be iterated indefinitely.

### 3.2 Implicit ensembling and multi-view features (Allen-Zhu & Li 2020)

Data has multiple predictive "views" (a car is identifiable by wheels, windows, headlights). SGD
with a random seed learns a *subset* of views. The teacher's soft labels on an image where its
views fire weakly push the student — which is learning a *different* random subset — to keep both
its own views and the teacher's. Self-distillation thus performs **implicit knowledge ensembling**
of two independently-initialized training runs inside one model. This predicts, correctly, that:

- gains shrink over generations (the view union saturates),
- an actual ensemble of the generations beats any single generation (BAN-E in Furlanello et al.),
- distilling from an ensemble teacher gives a larger boost than from one teacher.

### 3.3 Instance-specific label smoothing

Label smoothing replaces one-hot targets with $ (1-\epsilon)\,\mathbf{1}_y + \epsilon/K $ —
a *uniform* softening. Teacher soft labels are label smoothing with an instance-dependent,
similarity-aware $\epsilon$: ambiguous or mislabeled examples get heavily smoothed targets, clean
canonical examples stay sharp. Yuan et al. (2020) push this to the limit: a *weaker* teacher, or
even a handcrafted smoothing distribution, still improves the student — evidence that a large part
of the effect is regularization from target softening rather than privileged knowledge.

### Theory comparison

| Theory | Active ingredient | Predicts multi-round collapse | Predicts ensemble gain | Predicts weak-teacher gain |
|---|---|---|---|---|
| Regularization amplification | Geometric shrinkage of high-freq components | Yes (core result) | No | Partially |
| Multi-view ensembling | Union of features across random seeds | Weakly (saturation, not collapse) | Yes (core result) | No |
| Instance-specific smoothing | Per-example target softening | No | No | Yes (core result) |

The three are complementary, not competing: smoothing explains the cheap baseline gain, multi-view
explains why real teachers beat handcrafted smoothing, and Mobahi explains the iteration dynamics.

**Why this matters**: which theory dominates changes what you should do. If you are after the
smoothing effect, label smoothing is free and needs no teacher. If after the ensemble effect, train
2–3 generations and ensemble them. If your model is already heavily regularized, expect
self-distillation to help little or hurt — the Mobahi view says you are adding regularization to a
solution that did not need more.

---

## 4. Born-Again Networks in Detail

### Procedure

```python
# Generation 0: standard training
teacher = train(model_fn(), data, labels=hard)

# Generations 1..K: each learns from the previous generation's soft labels
generations = [teacher]
for k in range(1, K + 1):
    student = train(
        model_fn(),                        # fresh init — important (multi-view theory)
        data,
        labels=soft(generations[-1], T=4), # optionally mixed with hard labels
    )
    generations.append(student)

# Optional final boost: average the generations' predictions (BAN-E)
ensemble = lambda x: mean(g(x) for g in generations)
```

### What the paper found (CIFAR-100, DenseNet / ResNet)

- Gen 1 beats gen 0 consistently; gains fade by gen 2–3.
- **BAN-E** (ensembling generations) beats every individual generation.
- Cross-architecture works too: DenseNet teacher → ResNet student of similar size still gains.
- Two ablations dissect the mechanism: keeping only the argmax weight of the teacher
  (Confidence-Weighted by Teacher Max) and permuting the non-argmax probabilities both retain part
  of the gain — the *weighting* of examples by teacher confidence matters, not only the dark
  knowledge ordering.

### Cost model

Born-again training multiplies training cost by $(K{+}1)$ for a 1–3 pp gain. It is the right
tool when training is cheap relative to inference lifetime (the deployed model runs unchanged),
and the wrong tool when a single training run is already the budget ceiling — use EMA-teacher
variants (Section 6) that get most of the benefit in one run.

**Why this matters**: BAN is the reference experiment for the whole field — same data, same
architecture, sequential generations — and its ablations are the strongest published evidence that
soft-label geometry, not capacity transfer, is what distillation moves.

---

## 5. In-Network Self-Distillation (BYOT)

Be Your Own Teacher (Zhang et al. 2019) attaches auxiliary classifiers to intermediate layers and
distills the deepest classifier's knowledge into them during one training run:

$$
\mathcal{L} = \sum_{e=1}^{E} \Big[ \mathrm{CE}(y, \hat y_e)
\;+\; \alpha\, T^2\, \mathrm{KL}\big(\sigma(z_E/T) \,\|\, \sigma(z_e/T)\big)
\;+\; \beta\, \big\| F_e - F_E \big\|^2 \Big]
$$

where $e$ indexes exits, $z_E$/$F_E$ are the final exit's logits/features, and the three terms are
hard-label CE, logit distillation, and feature-hint alignment (cf.
[feature_distillation_why.md](feature_distillation_why.md)).

Properties:

- **One training run** — no generation chain; ~4x cheaper than 3-generation BAN.
- The final exit *also* improves (+1–2 pp on CIFAR-100/ResNet in the paper): shaping shallow
  layers to be predictive regularizes the whole feature hierarchy, a stronger form of deep
  supervision.
- Free bonus: the trained network supports **anytime inference** — early exits give a
  latency/accuracy dial at deployment, useful for embedded targets (e.g., in-vehicle compute
  budgets that vary with scene complexity).

**Why this matters**: BYOT converts self-distillation from a training-time luxury into an
architecture pattern with deployment value. If you are already adding auxiliary heads for deep
supervision, upgrading them to distillation heads costs a few lines.

---

## 6. Temporal Self-Distillation — EMA Teachers and DINO

### The EMA teacher

Instead of a converged previous generation, use an exponential moving average of the student's own
weights as the teacher, updated each step (see [weight_ema.md](../ema/weight_ema.md) for EMA
mechanics):

$$
\theta_T \leftarrow \tau\, \theta_T + (1-\tau)\, \theta_S, \qquad \tau \approx 0.996 \text{–} 0.9995
$$

The EMA teacher is a *temporal ensemble* — an average over the student's recent trajectory — so
this is the multi-view/ensemble theory implemented continuously: Mean Teacher (semi-supervised
learning), BYOL, and DINO all instantiate it.

### DINO: self-distillation with no labels

DINO trains the student to match the EMA teacher's output distribution across augmented views of
the same image, with **no ground-truth labels in the loss at all**:

$$
\mathcal{L} = - \sum_{x \in \text{global views}} \; \sum_{x' \neq x}
\sigma\!\big((z_T(x) - c)/T_T\big) \cdot \log \sigma\!\big(z_S(x')/T_S\big)
$$

Two asymmetries prevent the trivial solution (all inputs mapping to one output):

- **Centering** ($c$, an EMA of teacher outputs) removes the dominant direction — stops one
  dimension from winning;
- **Sharpening** ($T_T < T_S$, e.g. 0.04 vs 0.1) stops the uniform distribution from winning.

Remove either and training collapses. This is the Mobahi collapse phenomenon appearing in practice:
iterated self-distillation drifts toward degenerate fixed points unless something re-injects
diversity.

| Mechanism | BAN chain | DINO |
|---|---|---|
| Iteration | Discrete generations | Every step (EMA) |
| Collapse pressure | Multi-round shrinkage to zero function | All views to one point |
| Countermeasure | Stop at 1–3 generations; mix hard labels | Centering + sharpening asymmetry |

**Why this matters**: EMA-teacher self-distillation is the production form — one training run,
no label requirement, and it is the backbone of the pretraining behind current vision foundation
models (DINOv2 features are a common frozen backbone for driving perception stacks; cf.
[foundation_model_distillation.md](foundation_model_distillation.md) for distilling *from* them).

---

## 7. Self-Distillation in LLMs

The same pattern operates in modern LLM pipelines under different names:

- **Self-improvement / STaR-style loops**: sample the model's own chain-of-thought answers, keep
  the ones that verify (correct final answer, passing tests), fine-tune on them. The "soft label"
  is replaced by a *verifier-filtered sample* — self-distillation where the verifier plays the
  role that temperature played in BAN (deciding what part of the model's own output distribution
  to feed back).
- **Self-distillation fine-tuning (SDFT)**: when fine-tuning on a task dataset, first have the
  model *rewrite the target responses in its own words*, then fine-tune on the rewrites. Matching
  the target distribution to the model's own distribution reduces catastrophic forgetting versus
  fitting the original (distribution-mismatched) targets.
- **Iterated self-training caution**: training generation $k{+}1$ predominantly on generation
  $k$'s unfiltered outputs degrades tail knowledge over iterations (model-collapse literature) —
  the LLM-scale version of Mobahi's shrinkage-to-zero, with the verifier/filter as the
  anti-collapse mechanism (DINO's centering, in role).

**Why this matters**: if you design agent loops that fine-tune on their own outputs (see
[loop_design_playbook.md](../agentic_engineering/loop_design_playbook.md) — the verifier
discussion applies verbatim), self-distillation theory gives the design rule: the filter/verifier
is not optional plumbing, it is the term that decides between amplified regularization and
collapse.

---

## 8. Practical Recipes

### Recipe selection

| Situation | Use | Expected gain | Extra cost |
|---|---|---|---|
| Training cheap, deployment fixed, want free accuracy | BAN, 1–2 generations, then BAN-E if you can serve an ensemble | 1–3 pp | +1–2 training runs |
| One training run only | BYOT auxiliary exits, or EMA-teacher consistency | 1–2 pp | ~10–30% step overhead |
| Need latency/accuracy dial at inference | BYOT (keep the exits) | early exits at small accuracy cost | as above |
| Unlabeled data available | Mean-Teacher / DINO-style consistency on unlabeled + CE on labeled | task-dependent, often large | augmentation pipeline |
| Already using strong label smoothing + heavy augmentation | Probably skip — smoothing effect is already taken | ~0 | — |

### Hyperparameters that matter (in order)

1. **Temperature $T$**: 3–5 for classification logits. Too low ($T{=}1$) throws away dark
   knowledge; too high flattens everything toward uniform.
2. **Hard-label mixing $\lambda$**: keep a CE term with true labels ($\lambda \approx 0.5$–0.9 on
   the KD side works; pure-soft works in BAN but is less robust when the teacher has systematic
   errors). The hard labels are the anti-collapse anchor.
3. **Generations**: 1–2. Expect nothing after 3; expect degradation eventually (theory and
   notebook both show it).
4. **Fresh initialization per generation** (BAN): required by the multi-view mechanism;
   warm-starting the student from teacher weights kills the ensembling part of the gain. The
   warm-start ablation doubles as a *channel probe* — if your gain survives warm-starting, it
   is the denoising channel, not ensembling (companion notebook, Section 4).
5. **EMA decay $\tau$** (temporal variants): 0.996–0.9995; ramp $\tau$ up over training (DINO
   schedule) so the teacher tracks fast early, stabilizes late.

**Why this matters**: the recipes are cheap to try and the failure modes are known. The single most
common implementation mistake — warm-starting the student from the teacher — silently removes the
ensembling mechanism while leaving the pipeline looking correct.

---

## 9. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Warm-starting student from teacher weights | Gain vanishes; generations converge to identical models | Fresh random init each generation (multi-view mechanism needs seed diversity) |
| Too many generations | Accuracy peaks at gen 1–2, then declines toward trivial solutions | Stop early; monitor validation per generation; mix hard labels |
| Pure soft labels with a flawed teacher | Student faithfully inherits teacher's systematic errors | Keep a hard-label CE term as anchor ($\lambda < 1$) |
| $T^2$ factor forgotten in the KD term | Soft loss gradient vanishes at high $T$; KD term does nothing | Multiply KL term by $T^2$ (gradient scale correction) |
| Comparing student vs teacher with different budgets | "Self-distillation gain" is actually a longer schedule or better LR | Identical epochs/schedule/augmentation for teacher and student |
| Expecting gains on top of heavy regularization | No improvement; occasional degradation | Theory-consistent: smoothing benefit already consumed; skip or reduce other regularizers |
| EMA/DINO-style training collapses to constant output | Loss drops to trivial minimum; features useless | Centering + teacher sharpening (or equivalent asymmetry); check $T_T < T_S$ |
| Distilling on training set the teacher memorized | Teacher soft labels on train data are near one-hot; nothing to transfer | Use held-out/unlabeled data for the KD term, or strong augmentation to de-memorize |
| Expecting dark-knowledge gains on easy, well-separated classes | Soft-KL term contributes ~0 (companion notebook: digits + MLP); similarity structure is too poor to carry signal | The gain channel there is argmax denoising (self-training with pseudo-labels); use hard pseudo-labels under label noise, and reserve soft-label KD for fine-grained/many-class problems |

---

## 10. References

- Furlanello, Lipton, Tschannen, Itti, Anandkumar, *Born-Again Neural Networks*, ICML 2018.
  [arXiv:1805.04770](https://arxiv.org/abs/1805.04770)
- Mobahi, Farajtabar, Bartlett, *Self-Distillation Amplifies Regularization in Hilbert Space*,
  NeurIPS 2020. [arXiv:2002.05715](https://arxiv.org/abs/2002.05715)
- Allen-Zhu, Li, *Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in
  Deep Learning*, ICLR 2023. [arXiv:2012.09816](https://arxiv.org/abs/2012.09816)
- Zhang, Song, Gao, Chen, Bao, Ma, *Be Your Own Teacher: Improve the Performance of Convolutional
  Neural Networks via Self Distillation*, ICCV 2019. [arXiv:1905.08094](https://arxiv.org/abs/1905.08094)
- Yuan, Tay, Li, Wang, Feng, *Revisiting Knowledge Distillation via Label Smoothing Regularization*,
  CVPR 2020. [arXiv:1909.11723](https://arxiv.org/abs/1909.11723)
- Tarvainen, Valpola, *Mean Teachers Are Better Role Models*, NeurIPS 2017.
  [arXiv:1703.01780](https://arxiv.org/abs/1703.01780)
- Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin, *Emerging Properties in
  Self-Supervised Vision Transformers* (DINO), ICCV 2021.
  [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- Zelikman, Wu, Mu, Goodman, *STaR: Bootstrapping Reasoning With Reasoning*, NeurIPS 2022.
  [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- Yang et al., *Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning* (SDFT),
  ACL 2024. [arXiv:2402.13669](https://arxiv.org/abs/2402.13669)
- Shumailov et al., *AI Models Collapse When Trained on Recursively Generated Data*, Nature 2024.
  (model collapse) [arXiv:2305.17493](https://arxiv.org/abs/2305.17493)
- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*, 2015.
  [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
