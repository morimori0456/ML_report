# Distillation Methods — A Broader Survey

> Companion to [knowledge_distillation.md](knowledge_distillation.md), which focuses on
> **intermediate-feature** distillation (FitNets, AT, FSP, NST, PKT, RKD, CRD, OFD, ReviewKD).
> This document maps the **rest** of the field: better logit losses, the offline/online/self
> *schemes*, data-free distillation, and the task-specific recipes used in **detection /
> segmentation** and **NLP / LLMs**.

---

## Table of Contents
1. [Three Orthogonal Axes](#1-three-orthogonal-axes)
2. [Better Logit Losses (DKD, WSLD, NKD)](#2-better-logit-losses-dkd-wsld-nkd)
3. [Bridging the Capacity Gap (TAKD)](#3-bridging-the-capacity-gap-takd)
4. [Distillation Schemes: Offline / Online / Self](#4-distillation-schemes-offline--online--self)
5. [Data-Free Distillation](#5-data-free-distillation)
6. [Object Detection & Segmentation](#6-object-detection--segmentation)
7. [NLP & LLM Distillation](#7-nlp--llm-distillation)
8. [How to Choose](#8-how-to-choose)
9. [References](#9-references)

---

## 1. Three Orthogonal Axes

The first guide classified methods by **what is transferred** (response / feature / relation).
But two methods that transfer the same thing can still differ on:

| Axis | Options | Question it answers |
|---|---|---|
| **What is transferred** | logits / features / relations | (covered in the feature guide) |
| **Distillation scheme** | offline / online / self | *Is there a fixed pretrained teacher?* |
| **Data setting** | data-driven / data-free | *Do we have the original training data?* |
| **Domain recipe** | classification / detection / segmentation / NLP / LLM | *What structure does the task add?* |

A real system picks one option on **each** axis, e.g. "offline + feature + data-driven +
detection-specific (FGD)". The sections below walk the axes the first guide skipped.

---

## 2. Better Logit Losses (DKD, WSLD, NKD)

Hinton KD (the `T²·KL(soft student ‖ soft teacher)` baseline) leaves performance on the table.
These reformulations keep the logit-only simplicity but extract more signal.

### DKD — Decoupled Knowledge Distillation (CVPR 2022)
Algebraically split the KD loss into two parts:

$$ \text{KD} = \underbrace{\text{TCKD}}_{\text{target vs non-target}} + \;(1 - p_t^{T})\,\underbrace{\text{NCKD}}_{\text{among non-target classes}} $$

- **TCKD** — a binary distribution {target, all-others}: transfers the *difficulty* of a sample.
- **NCKD** — the distribution *among the non-target classes*: this is the actual "dark
  knowledge" (which wrong classes look plausible) and is **the main reason KD works**.

The problem: the coupling factor `(1 − p_tᵀ)` **suppresses NCKD exactly on the samples the
teacher is most confident about** — the well-classified ones that carry the cleanest dark
knowledge. DKD decouples them with independent weights:

$$ \text{DKD} = \alpha\,\text{TCKD} + \beta\,\text{NCKD} $$

Letting `β > (1 − p_tᵀ)` un-suppresses NCKD. Pure-logit method, no features, often matches
feature-based methods while being far cheaper. Strong default if you only have logits.

### WSLD / NKD and the label-smoothing view
KD with temperature is closely related to **label smoothing** and adaptive regularization.
**WSLD** (Weighted Soft Label Distillation) reweights each sample's soft-label loss by a
bias-variance analysis; **NKD** (Normalized KD) normalizes the non-target logits so their
distribution — not just its scale — is matched. All are drop-in replacements for the KD term.

---

## 3. Bridging the Capacity Gap (TAKD)

Counter-intuitive but important: **a better/larger teacher can distill *worse***, because the
representational gap to a tiny student is too wide to imitate. **TAKD** (Teacher-Assistant KD)
inserts one or more **intermediate-size "teacher assistants"**: teacher → TA → student. Each
hop is a manageable gap. Related ideas: **DGKD** (densely-guided, all TAs supervise the
student jointly) and simply choosing a teacher of *moderate* size. Keep this in mind whenever
"the SOTA teacher made my student worse."

---

## 4. Distillation Schemes: Offline / Online / Self

### Offline (the default)
A **fixed, pretrained** teacher; only the student trains. Everything in the feature guide and
§2 is offline. Simple and stable; needs a good teacher to exist already.

### Online (mutual) — train teacher & student together
No pretrained teacher; a **cohort** of networks teach each other on the fly.
- **DML (Deep Mutual Learning)** — two (or more) peers, each trained on `CE + KL(to the other
  peer)`. They explore different minima and regularize each other; even equal-size peers both
  improve over solo training.
- **ONE / KDCL** — build an **on-the-fly ensemble** (e.g., a multi-branch net whose averaged
  logits are the teacher) and distill the ensemble into each branch within one training run.

Online is useful when no strong teacher exists, or to avoid a separate expensive teacher stage.

### Self-distillation — the model is its own teacher
- **Born-Again Networks (BAN)** — train a student of the **same architecture** from the
  teacher, then use *that* student as the teacher for the next generation, and iterate.
  Surprisingly, gen-k often beats the original.
- **BYOT (Be Your Own Teacher)** — attach auxiliary classifiers to **shallow layers** and
  distill the **deepest** layer's output/features back into them; one network, one pass, no
  external teacher. Deep→shallow self-supervision.
- **Snapshot / temporal self-distillation** — use an EMA or earlier-epoch copy of the model as
  the teacher (the mechanism behind Mean-Teacher and many SSL methods).

> Why self-distillation helps at all (same capacity!): the soft targets act as a strong,
> per-sample **regularizer** that smooths the labels and encodes learned inter-class structure
> — a free accuracy/calibration bump without changing the architecture.

---

## 5. Data-Free Distillation

When the original training data is unavailable (privacy, licensing, size) but a **trained
teacher** is, *synthesize* inputs that make the teacher "comfortable", then distill on them.

- **DeepInversion** — optimize random-noise images so that (a) the teacher confidently predicts
  a target class and (b) the per-layer **BatchNorm running mean/variance** of the synthetic
  batch match the teacher's stored BN statistics. The BN-matching prior is what makes the
  images look natural enough to transfer.
- **DAFL** — train a **generator** adversarially so its outputs maximize teacher response
  (treating the teacher as a fixed discriminator-like critic), then distill student from
  generator samples.
- **ZSKD (Zero-Shot KD)** — craft "data impressions" by sampling soft labels from a Dirichlet
  fit to the teacher's classifier weights and inverting them.

Trade-off: no data needed, but synthesis is expensive and quality-sensitive; accuracy usually
trails data-driven KD.

---

## 6. Object Detection & Segmentation

Dense tasks add spatial structure and a severe **foreground/background imbalance**, so naive
"distill all pixels equally" fails. These are the recipes used in practice (and in many
autonomous-driving stacks).

### Detection
- **FGFI (Fine-Grained Feature Imitation)** — only imitate teacher features at locations **near
  ground-truth objects** (an anchor-overlap mask), not the vast background. Distilling
  everything lets background dominate and hurts.
- **FGD (Focal and Global Distillation, CVPR 2022)** — split into **focal** (separate
  foreground/background, weight by the teacher's spatial+channel attention) and **global**
  (relations between pixels via a GcBlock) terms. A very common, strong detector-KD baseline.
- **LD (Localization Distillation)** — distill the **bounding-box localization** as a
  probability distribution over box edges (general distribution), showing that "where the box
  is" deserves its own KD, not just the classification logits.
- **MGD (Masked Generative Distillation)** — randomly mask student feature pixels and force it
  to **regenerate** the teacher's full feature from the partial input; transfers via generation
  rather than direct mimicry. Works for both detection and segmentation.

### Segmentation
- **SKD (Structured KD)** — match **pair-wise pixel similarity** maps and use an adversarial
  **holistic** term, since per-pixel KD ignores spatial structure.
- **CWD (Channel-Wise Distillation)** — normalize each channel's `H×W` activation into a
  **soft spatial distribution** (softmax over positions) and minimize KL per channel. Cheap,
  asymmetric (emphasizes the most-activated regions), and a strong dense-prediction default.
- **IFVD** — match intra-class feature variation (how each class's features are distributed).

---

## 7. NLP & LLM Distillation

### Encoder / BERT-family (classification, embeddings)
- **DistilBERT** — halve the layers (12→6), initialize from alternating teacher layers, train
  with a **triple loss**: MLM + **cosine** embedding alignment + KL on soft logits. ~40%
  smaller, ~60% faster, ~97% of BERT's GLUE.
- **PKD (Patient KD)** — don't only match the final layer; distill the **[CLS] hidden states of
  multiple intermediate layers** (skip or last-k), so the student learns the teacher's
  *process*, not just its answer.
- **TinyBERT** — distill at three levels — **embedding**, **Transformer layer** (both the
  **attention matrices** and the **hidden states**), and **prediction** — in a **two-stage**
  scheme: *general* distillation on a large corpus, then *task-specific* distillation (with data
  augmentation). One of the most thorough feature-distillation recipes for Transformers.
- **MiniLM** — distill only the **self-attention relations of the last Transformer layer**:
  the **Query-Key** attention distribution and the **Value-Value** relation (scaled dot-product
  of values). Being relation-based, it is **layer-count and width agnostic** (no layer-mapping
  or adapter needed) — elegant and effective. MiniLMv2 generalizes to multi-head relations.

### Generative LLMs (sequence models)
The objective shifts from matching a single softmax to matching a **distribution over
sequences**, which changes what "KL" should be.

- **Sequence-level KD (Kim & Rush, 2016)** — the classic trick: run the teacher to generate
  output sequences (beam search), then train the student with plain cross-entropy on **teacher-
  generated text**. Approximates sequence-level distribution matching with zero KD machinery —
  still a strong, simple baseline (and the basis of much "synthetic data" training today).
- **Word/token-level KD** — KL on the teacher's next-token distribution at every position
  (teacher-forced). Simple but suffers **exposure bias** (trained only on teacher-forced
  context) and **mode-covering** (forward KL forces the student to put mass on *all* teacher
  modes, which a small student cannot represent → blurry, hallucination-prone outputs).
- **MiniLLM (2023)** — replace **forward KL** with **reverse KL** `KL(student ‖ teacher)`,
  optimized at sequence level with a policy-gradient. Reverse KL is **mode-seeking**: the
  student commits to the teacher's *major* modes instead of smearing over low-probability
  regions it can't model — empirically better, more faithful generations.
- **GKD — Generalized KD (2023)** — the key issue is the **train/inference mismatch**: the
  student is distilled on teacher (or fixed) sequences but at inference runs on its **own**
  outputs. GKD trains **on-policy**, on the **student's own generated sequences** scored by the
  teacher, and **unifies the divergence choice** (forward/reverse KL, JSD) into one framework.
  Finding: the **sampling policy (on-policy)** usually matters more than the exact divergence.
  This on-policy paradigm underlies most modern LLM distillation.

> Practical note: production small models (DistilGPT-2, and the distilled variants in modern
> Gemma/Llama/Qwen families) typically combine **sequence-level KD / synthetic teacher data**
> with **on-policy** refinement — the §7 generative methods, not the CNN feature methods.

---

## 8. How to Choose

| Situation | Recommended approach |
|---|---|
| Only have logits, want max gain cheaply | **DKD** (decoupled logit KD) |
| Strong teacher made the student *worse* | **TAKD** (teacher assistant) / smaller teacher |
| No pretrained teacher available | **DML** (online mutual) |
| Same architecture, want a free boost | **BAN / BYOT** (self-distillation) |
| Original data is gone | **DeepInversion / DAFL** (data-free) |
| Object detection | **FGD** or **FGFI** (foreground-focused) |
| Semantic segmentation | **CWD** (channel-wise) / **SKD** |
| BERT-style encoder | **TinyBERT** (thorough) or **MiniLM** (no layer mapping) |
| Generative LLM | **GKD / on-policy** (+ reverse-KL ideas from MiniLLM) |
| Tiny student, hard dense task | **MGD** (masked generative) |

General rules that survive across all of them:
1. **Keep the task loss** (CE / MLM / LM) alongside any distillation term.
2. **Mask by relevance** on dense tasks — never distill background uniformly.
3. **Match the divergence to the model**: forward KL for classification; reverse/on-policy for
   open-ended generation.
4. **Mind the capacity gap** — if the student can't represent the teacher, a softer target
   (attention, relations, mode-seeking KL) beats exact mimicry.

---

## 9. References

- **DKD**: Zhao et al., *Decoupled Knowledge Distillation*, CVPR 2022. [arXiv:2203.08679](https://arxiv.org/abs/2203.08679)
- **TAKD**: Mirzadeh et al., *Improved KD via Teacher Assistant*, AAAI 2020. [arXiv:1902.03393](https://arxiv.org/abs/1902.03393)
- **DML**: Zhang et al., *Deep Mutual Learning*, CVPR 2018. [arXiv:1706.00384](https://arxiv.org/abs/1706.00384)
- **BAN**: Furlanello et al., *Born-Again Neural Networks*, ICML 2018. [arXiv:1805.04770](https://arxiv.org/abs/1805.04770)
- **BYOT**: Zhang et al., *Be Your Own Teacher*, ICCV 2019. [arXiv:1905.08094](https://arxiv.org/abs/1905.08094)
- **DeepInversion**: Yin et al., *Dreaming to Distill*, CVPR 2020. [arXiv:1912.08795](https://arxiv.org/abs/1912.08795)
- **DAFL**: Chen et al., *Data-Free Learning of Student Networks*, ICCV 2019. [arXiv:1904.01186](https://arxiv.org/abs/1904.01186)
- **FGFI**: Wang et al., *Distilling Object Detectors with Fine-grained Feature Imitation*, CVPR 2019. [arXiv:1906.03609](https://arxiv.org/abs/1906.03609)
- **FGD**: Yang et al., *Focal and Global KD for Detectors*, CVPR 2022. [arXiv:2111.11837](https://arxiv.org/abs/2111.11837)
- **LD**: Zheng et al., *Localization Distillation for Object Detection*, CVPR 2022. [arXiv:2102.12252](https://arxiv.org/abs/2102.12252)
- **CWD**: Shu et al., *Channel-wise KD for Dense Prediction*, ICCV 2021. [arXiv:2011.13256](https://arxiv.org/abs/2011.13256)
- **SKD**: Liu et al., *Structured KD for Semantic Segmentation*, CVPR 2019. [arXiv:1903.04197](https://arxiv.org/abs/1903.04197)
- **MGD**: Yang et al., *Masked Generative Distillation*, ECCV 2022. [arXiv:2205.01529](https://arxiv.org/abs/2205.01529)
- **DistilBERT**: Sanh et al., 2019. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- **PKD**: Sun et al., *Patient KD for BERT Compression*, EMNLP 2019. [arXiv:1908.09355](https://arxiv.org/abs/1908.09355)
- **TinyBERT**: Jiao et al., 2020. [arXiv:1909.10351](https://arxiv.org/abs/1909.10351)
- **MiniLM**: Wang et al., 2020. [arXiv:2002.10957](https://arxiv.org/abs/2002.10957)
- **Seq-level KD**: Kim & Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016. [arXiv:1606.07947](https://arxiv.org/abs/1606.07947)
- **MiniLLM**: Gu et al., *MiniLLM: KD of LLMs (reverse KL)*, ICLR 2024. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **GKD**: Agarwal et al., *On-Policy / Generalized KD for LLMs*, ICLR 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
