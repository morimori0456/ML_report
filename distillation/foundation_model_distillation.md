---
title: "Foundation Model Distillation — Leveraging Large Pretrained Models as Teachers"
description: "A guide to distilling knowledge from foundation models like BERT, CLIP, DINOv2, and GPT into compact, deployable student models."
---

> Comprehensive guide on distilling knowledge from foundation models (BERT, CLIP, DINOv2, GPT, etc.) into compact students. Companion notebook: [foundation_model_distillation_demo.ipynb](https://github.com/morimori0456/ML_report/blob/main/distillation/foundation_model_distillation_demo.ipynb). For the theory of why features matter, see [feature_distillation_why.md](feature_distillation_why.md); for modern KD techniques, see [advanced_kd_practical.md](advanced_kd_practical.md).

Foundation model distillation sits at the intersection of two major forces in modern ML: the dramatic capability jump that occurs at scale, and the deployment constraint that forces models to fit within memory, latency, and energy budgets. Understanding when and how to tap foundation models as teachers is now a core skill for production ML engineers.

---

## Table of Contents
1. [What is a Foundation Model?](#1-what-is-a-foundation-model)
2. [Why Use a Foundation Model as a Teacher?](#2-why-use-a-foundation-model-as-a-teacher)
3. [The Capacity Gap Problem](#3-the-capacity-gap-problem)
4. [Distillation Taxonomy for Foundation Models](#4-distillation-taxonomy-for-foundation-models)
5. [NLP Foundation Model Distillation](#5-nlp-foundation-model-distillation)
6. [Vision Foundation Model Distillation](#6-vision-foundation-model-distillation)
7. [Cross-Modal Distillation (CLIP-style)](#7-cross-modal-distillation-clip-style)
8. [Data-Free and Black-Box Teacher Distillation](#8-data-free-and-black-box-teacher-distillation)
9. [Advantages — Detailed Analysis](#9-advantages--detailed-analysis)
10. [Pitfalls and Cautions](#10-pitfalls-and-cautions)
11. [Practical Checklist](#11-practical-checklist)
12. [References](#12-references)

---

## 1. What is a Foundation Model?

The term **foundation model** was coined by the Stanford HAI Center in 2021 (Bommasani et al.) to describe models that are:

> Trained on broad data at scale, and adaptable to a wide range of downstream tasks.

Three necessary conditions distinguish a foundation model from a conventional pretrained model:

| Property | Description | Examples |
|---|---|---|
| **Scale** | Billions of parameters, internet-scale training data, massive compute | GPT-4, Gemini, CLIP-ViT-L |
| **Emergent capability** | Behaviors that appear unpredictably beyond a scale threshold; not explicitly trained | In-context learning, chain-of-thought reasoning |
| **Generality** | Adaptable to diverse tasks via fine-tuning, prompting, or zero-shot | BERT → QA/NER/classify; DINOv2 → depth/segmentation/retrieval |

### Foundation model families

```
Foundation Models
├── Language (LLM)
│   ├── Encoder-only:  BERT, RoBERTa, DeBERTa
│   ├── Decoder-only:  GPT-4, Claude, LLaMA, Mistral
│   └── Encoder-decoder: T5, BART
├── Vision (VFM)
│   ├── Supervised:    MAE, BEiT, SwinV2
│   └── Self-supervised: DINO, DINOv2, SAM
├── Multimodal
│   ├── Contrastive:   CLIP, SigLIP, ALIGN
│   └── Generative:    LLaVA, Flamingo, GPT-4V
└── Domain-specific
    ├── Code:          Codex, StarCoder, DeepSeek-Coder
    ├── Science:       AlphaFold, ESMFold (protein)
    └── Driving:       DriveTransformer, UniAD, NAVSIM
```

### What "emergent" means in practice

Scale changes the shape of the loss surface. Beyond a threshold, models develop:
- **In-context learning**: GPT-3 can translate a new language after seeing 3 examples in the prompt, despite never being explicitly trained to translate.
- **Compositional generalization**: DINOv2 features, trained only with self-supervised objectives, produce SOTA dense prediction on ADE20K without any semantic label supervision.
- These capabilities are **not** present in smaller models trained with identical objectives — they arise from scale alone.

### Key insight
> **Emergent knowledge is transferable**: A student that correctly mimics a foundation model's output distribution inherits the emergent behaviors implicitly, even without understanding the mechanism that produced them.

---

## 2. Why Use a Foundation Model as a Teacher?

### 2.1 Richer soft labels

Standard hard labels encode `P(y|x) = 1` for the correct class and `0` elsewhere. Foundation model soft labels encode:

$$q_T^{(i)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

These non-zero probabilities over incorrect classes reveal **inter-class similarity structure** the teacher learned from its large training set. A student trained on 10k images inherits knowledge implicitly gathered from billions of examples.

### 2.2 Better intermediate representations

Foundation model intermediate features are **semantically richer** than those of models trained from scratch on small datasets. DINOv2, for instance, produces features that:
- Cluster by semantic content without any label supervision
- Support segmentation, depth estimation, and retrieval with a simple linear head
- Are more transferable than any supervised ResNet feature at the same FLOPs

### 2.3 Reduced annotation cost

The teacher's soft labels provide signal even for **unlabeled data**. This enables:
- Semi-supervised distillation: label a small set, use FM to generate pseudo-labels for the rest
- Data augmentation: teacher provides consistent labels for augmented views

### 2.4 Task-agnostic teacher, multi-task student

A single foundation model teacher can teach multiple tasks simultaneously:

```python
# One teacher → many specializations
teacher_features = fm_teacher.extract(image)    # computed once

student_loss = (
    alpha * task_loss_detection(student, labels) +
    beta  * feature_loss(student.backbone_out, teacher_features) +
    gamma * task_loss_depth(student, depth_labels)
)
```

### 2.5 Zero-shot and few-shot capability transfer

CLIP-based distillation enables **zero-shot classification** in the student without ever seeing the target classes during distillation training. The student inherits CLIP's alignment between images and text descriptions.

### Summary: when FM distillation pays off

| Scenario | FM distillation benefit | Magnitude |
|---|---|---|
| Small labeled dataset (<10k) | Teacher provides rich pseudo-labels | Very high |
| OOD generalization required | FM features generalize better | High |
| Multi-task deployment | One teacher, many student heads | High |
| Large→small compression (>10x) | FM soft labels prevent info collapse | Medium |
| Domain-specific task (e.g., driving) | General FM + domain fine-tune → student | Medium |

---

## 3. The Capacity Gap Problem

This is the most important failure mode specific to foundation model distillation.

### The problem

When the teacher is extremely large and the student is very small, the student **cannot absorb** the teacher's knowledge — its representational capacity is too limited to fit what the teacher produces. Counterintuitively, a student distilled from a 6B-parameter teacher can perform **worse** than one distilled from a 300M-parameter teacher:

$$\text{Acc}(\text{student} \leftarrow \text{huge teacher}) < \text{Acc}(\text{student} \leftarrow \text{medium teacher})$$

This happens because:
1. The student's gradient updates are dominated by the mismatch loss
2. The teacher's class probabilities are very peaked (high confidence, low temperature diversity)
3. The student cannot represent the teacher's rich feature space, so feature-level loss becomes noisy

### Solutions

| Strategy | Mechanism | Key paper |
|---|---|---|
| **TAKD (Teacher-Assistant KD)** | Chain: FM → medium TA → small student | Mirzadeh et al., AAAI 2020 |
| **Progressive distillation** | Start from FM, iteratively compress | Distil-Whisper approach |
| **Logit standardisation** | z-score teacher logits before KD; removes scale mismatch | Sun et al., CVPR 2024 |
| **DIST** | Pearson correlation loss; scale/shift invariant | Huang et al., NeurIPS 2022 |
| **Partial layer alignment** | Match only the last few FM layers, skip early ones | TinyBERT approach |
| **Adapter projection** | Learnable linear to project student dim → teacher dim | FitNets, PKD |

### TAKD architecture

```
Foundation Model (7B)
       ↓  KD
Teacher-Assistant (1B, pre-trained from FM)
       ↓  KD
Small Student (100M, production model)
```

Each step halves the parameter count — the capacity gap at each stage is bridgeable.

---

## 4. Distillation Taxonomy for Foundation Models

```
FM Distillation
├── By what is transferred
│   ├── Logit-level (black-box compatible)
│   │   ├── Standard KD (Hinton)
│   │   ├── DIST (Pearson correlation)
│   │   └── Logit Standardisation
│   └── Feature-level (white-box, architecture-dependent)
│       ├── Layer-to-layer (TinyBERT, FitNets)
│       ├── Attention-to-attention (MiniLM)
│       └── Relation-based (RKD, CRD)
├── By teacher access
│   ├── White-box (full access to weights + activations)
│   └── Black-box / API-only (logits or text output only)
└── By training data
    ├── Data-present (same dataset)
    ├── Semi-supervised (subset labeled + FM pseudo-labels)
    └── Data-free (generator or dataset distillation)
```

---

## 5. NLP Foundation Model Distillation

### DistilBERT (Sanh et al., 2019)

The first widely-deployed BERT distillation. Uses three losses:

$$\mathcal{L} = \alpha \mathcal{L}_{CE}^{\text{hard}} + \beta \mathcal{L}_{KD}^{\text{soft}} + \gamma \mathcal{L}_{cos}^{\text{hidden}}$$

- $\mathcal{L}_{KD}^{\text{soft}}$: KL divergence on softmax outputs with temperature T=8
- $\mathcal{L}_{cos}^{\text{hidden}}$: cosine similarity between teacher and student hidden states at the same layer index
- Result: 40% smaller, 60% faster, retains 97% of BERT's GLUE performance

### TinyBERT (Jiao et al., 2020) — Full transformer distillation

TinyBERT distills at **every layer** of the transformer:

$$\mathcal{L}_{TinyBERT} = \mathcal{L}_{emb} + \sum_{i=1}^{N} \mathcal{L}_{layer}^{(i)} + \mathcal{L}_{pred}$$

Where each $\mathcal{L}_{layer}^{(i)}$ decomposes into:
- **Attention matrix loss**: MSE between teacher and student attention matrices $\mathbf{A}^T$, $\mathbf{A}^S$
- **Hidden state loss**: MSE between hidden states (with linear projection $W_h$ to align dimensions)

$$\mathcal{L}_{attn}^{(i)} = \frac{1}{h} \sum_{j=1}^h \text{MSE}(\mathbf{A}_j^S, \mathbf{A}_j^T)$$

Two-stage training:
1. **General distillation**: distil from BERT on large unlabeled corpus (Wikipedia + BookCorpus)
2. **Task-specific distillation**: distil on task data with fine-tuned BERT teacher

### MiniLM (Wang et al., 2020) — Self-attention relation distillation

Key insight: instead of matching the value of attention matrices, match the **self-attention relation** (value-to-value dot-products):

$$\mathcal{L}_{MiniLM} = \text{KL}\left( \frac{VV^T_S}{\sqrt{d_k}} \| \frac{VV^T_T}{\sqrt{d_k}} \right)$$

This is **architecture-independent**: student and teacher need not have the same hidden dimension, since only the last transformer layer is used. MiniLM achieves DistilBERT accuracy with 50% of the parameters.

### Comparison: NLP FM distillation methods

| Method | Teacher | Student | What is matched | GLUE score (avg) | Speed-up |
|---|---|---|---|---|---|
| BERT base | — | — | — | 82.5 | 1× |
| DistilBERT | BERT base | 6-layer, 768d | Softmax + last hidden | 77.0 | 1.6× |
| PKD-BERT | BERT base | 6-layer, 768d | Last 6 hidden states | 80.2 | 1.9× |
| TinyBERT-4 | BERT base | 4-layer, 312d | All attn + hidden | 82.6 | 7.5× |
| MiniLM-L6 | BERT large | 6-layer, 384d | Last self-attn relation | 84.0 | 5.3× |

---

## 6. Vision Foundation Model Distillation

### DINO / DINOv2 as teacher (Caron et al., 2021; Oquab et al., 2023)

DINOv2 is a ViT-L/14 trained with self-supervised DINO objective. Its features are remarkably transferable — even a k-NN classifier on DINOv2 features beats supervised ResNet-50 on ImageNet.

**Using DINOv2 as distillation teacher:**

```python
import torch
import torch.nn.functional as F

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dino.eval()

def dino_feature_loss(student_feat, images, temperature=0.1):
    with torch.no_grad():
        teacher_feat = dino(images)  # (B, 768)
    # L2-normalize both
    s = F.normalize(student_feat, dim=-1)
    t = F.normalize(teacher_feat, dim=-1)
    # InfoNCE-style or simple MSE
    return F.mse_loss(s, t)
```

### SAM (Segment Anything Model) distillation

SAM (ViT-H, 636M) can segment any object with a prompt. Distillation targets:
1. **Image encoder outputs** — SAM's 256-d image embeddings (from ViT-H backbone)
2. **Mask decoder outputs** — multi-scale mask logits

Efficient SAM (ICML 2024) distills SAM to a ViT-S/16 student:
- 20× smaller
- 20× faster
- <3 mIoU drop on COCO

### FD-CLIP: Feature Distillation from CLIP

CLIP (ViT-L/14) produces joint vision-language embeddings. Distilling its visual encoder:

$$\mathcal{L}_{FD\text{-}CLIP} = 1 - \frac{\mathbf{f}_S \cdot \mathbf{f}_T}{\|\mathbf{f}_S\| \|\mathbf{f}_T\|}$$

This cosine loss is **scale-invariant** — important because the student's embedding magnitudes may differ from CLIP's.

---

## 7. Cross-Modal Distillation (CLIP-style)

CLIP learns a shared embedding space for images and text. Cross-modal distillation uses the **paired modality** to provide supervision for the **target modality student**.

### CLIP-KD pattern

```
Text: "a photo of a cat"  →  CLIP Text Encoder  →  text embedding t
Image: [cat image]        →  CLIP Image Encoder →  teacher embedding f_T
                          →  Student (small CNN) →  student embedding f_S

Loss = 1 - cosine(f_S, f_T)  +  1 - cosine(f_S, t)  [optional: language alignment]
```

### Advantage: zero-shot transfer in the student

After CLIP-style distillation, the student's embedding space is aligned with CLIP's. To classify a new unseen class:

```python
# At inference — no retraining needed for new classes
new_class_embedding = clip_text_encoder("a photo of a {new_class}")
student_embedding = student_encoder(image)
score = cosine_similarity(student_embedding, new_class_embedding)
```

### DeiT — Distillation via Distillation Token

DeiT (Data-efficient Image Transformer, Touvron et al., 2021) adds a special `[distil]` token to the standard ViT `[cls]` token:

$$\mathcal{L}_{DeiT} = \frac{1}{2} \mathcal{L}_{CE}(y^{cls}, y) + \frac{1}{2} \mathcal{L}_{KD}(y^{distil}, y^T_{hard})$$

- The `[cls]` head is supervised with ground-truth labels
- The `[distil]` head is supervised with the **teacher's hard argmax** (not soft labels)
- At inference, both token predictions are averaged

Using a CNN teacher (RegNet-Y-16GF) rather than another ViT allows cross-architecture knowledge transfer: the CNN's inductive biases (locality, translation invariance) propagate to the ViT student via the distillation token.

---

## 8. Data-Free and Black-Box Teacher Distillation

### Black-box (API-only) distillation from LLMs

When you cannot access internal weights — only API outputs — logit-level distillation is impossible. Options:

| Method | Teacher output needed | Notes |
|---|---|---|
| **Sequence-level KD** | Generated text (greedy/beam) | Student learns on FM-generated text as training data |
| **GKD (Generalized KD)** | Token-level log-probs | Available on some APIs (GPT-4 logprobs=True) |
| **On-policy GKD** | Student generates, teacher scores | Prevents exposure bias |
| **Speculative decoding** | None (inference-time only) | Not distillation per se — FM verifies student tokens |

### Sequence-level KD (Kim & Rush, 2016)

Generate a new training corpus by running the teacher on the training inputs:

```
For each (x, y) in train_set:
    ŷ = teacher.generate(x)     # FM inference (expensive)
    new_dataset.append((x, ŷ))

student.train(new_dataset, standard_CE_loss)
```

This transfers the teacher's **generation style** to the student. Used extensively to compress GPT-4 knowledge into open-source models (Alpaca, WizardLM, etc.).

### GKD (Agarwal et al., 2023)

On-policy variant that addresses distribution shift:

$$\mathcal{L}_{GKD} = \mathbb{E}_{x \sim \mathcal{D}, \hat{y} \sim p_S(\cdot|x)} \left[ \text{KL}(p_T(\cdot|\hat{y},x) \| p_S(\cdot|\hat{y},x)) \right]$$

Key: the student **generates** $\hat{y}$, then the teacher scores it token by token. The student learns on sequences from its own distribution — preventing the teacher's easy-case bias that afflicts offline distillation.

### Data-Free KD from Foundation Models

When neither teacher data nor real images are available, synthesize samples using the teacher itself:

```python
# DeepInversion-style: optimize a random input to match FM's batch norm stats
x = torch.randn(B, C, H, W, requires_grad=True)
optimizer = torch.optim.Adam([x], lr=0.01)

for step in range(1000):
    logits = teacher(x)
    loss = (
        -F.cross_entropy(logits, target_class)       # classification loss
        + bn_regularization(teacher, x)               # match BN statistics
        + tv_loss(x)                                   # total variation (smoothness)
    )
    loss.backward(); optimizer.step()
```

---

## 9. Advantages — Detailed Analysis

### 9.1 Data efficiency

| Labels available | Baseline (scratch) | Supervised KD (small teacher) | FM distillation |
|---|---|---|---|
| 100% | 76.1% | 80.3% | **83.4%** |
| 10% | 59.2% | 71.4% | **80.1%** |
| 1% | 38.1% | 52.3% | **73.8%** |

(Approximate numbers from DINOv2 semi-supervised experiments on ImageNet-1K.)

At 1% labeled data, FM distillation cuts the performance gap by >50% vs supervised training.

### 9.2 OOD generalization

Foundation models are trained on diverse internet-scale data. Students inheriting FM features generalize better to distribution shifts:

- ImageNet-C (corrupted): +6% mean accuracy vs supervised student
- ImageNet-A (adversarial): +12%
- ImageNet-Sketch: +8%

### 9.3 Emergent capability transfer

Empirical finding: students distilled from instruction-tuned LLMs (GPT-4, Claude) develop **rudimentary in-context learning** even at 1B parameters, which students trained from scratch at 1B never exhibit. The soft labels contain the teacher's in-context reasoning structure.

---

## 10. Pitfalls and Cautions

| Pitfall | Symptom | Fix |
|---|---|---|
| **Capacity gap** | Student KD loss decreases but accuracy is worse than baseline | Use TAKD (teacher assistant) or progressive distillation |
| **Hallucination transfer** | Student confidently predicts wrong answers inherited from FM biases | Filter teacher outputs with confidence threshold; mix hard labels |
| **Terms of service violation** | Using GPT-4 API to generate training data for a competing model | Read API ToS carefully; OpenAI and Anthropic prohibit this for competing models |
| **Domain mismatch** | FM trained on internet images, student task is medical imaging | Fine-tune FM on domain first, then distill; or use domain-adapted FM |
| **Feature dimension mismatch** | Student hidden dim ≠ teacher hidden dim | Add linear projection adapter (trainable); freeze teacher side |
| **Layer correspondence** | Unclear which teacher layer to match with which student layer | Use progressive layer matching (TinyBERT strategy) or skip hidden-state matching, use only output |
| **Temperature sensitivity** | Small changes in T cause large accuracy swings | Sweep T ∈ {2, 4, 8, 16}; use logit standardization to decouple |
| **Training instability** | Loss NaN or divergence when FM logits have extreme values | Clip teacher logits; use gradient clipping; reduce LR |
| **Black-box API cost** | Teacher inference is expensive ($$) | Cache teacher outputs; distil in one offline pass |
| **Stale teacher** | FM teacher updated after distillation; student knowledge is outdated | Re-distil periodically or pin teacher version |

### On ToS violations — important note

Using a closed-source FM's outputs to train a competing commercial model is explicitly prohibited by most providers. **Research use within an organization is generally allowed**, but distributing the resulting student model or using it in a product may not be. Always verify:

- OpenAI: prohibits using outputs to train models that compete with OpenAI products
- Anthropic: similar restrictions on Claude outputs
- Meta LLaMA: community license permits research but restricts commercial use above 700M MAU

---

## 11. Practical Checklist

### Before starting

- [ ] Is the FM teacher **white-box** (access to weights) or **black-box** (API only)?
  - White-box: full feature-level distillation possible
  - Black-box: logit or sequence-level only; check ToS
- [ ] What is the **parameter ratio** teacher/student?
  - < 10×: standard KD or DIST
  - 10–100×: use teacher assistant (TAKD) or progressive distillation
  - > 100×: definitely TAKD; consider intermediate FM (e.g., 7B → 1B → 200M)
- [ ] Is there **domain mismatch** between FM pretraining data and your task?
  - Yes: fine-tune FM on domain data first, then distil from the fine-tuned version
  - No: distil from base FM or fine-tuned task version

### Loss recipe by task

| Task | Recommended FM distillation loss |
|---|---|
| Image classification | FM logit KD (DIST or Logit-Std) + feature MSE at last layer |
| Dense prediction (seg, depth) | FM feature MSE at multiple scales + task loss |
| NLP sequence classification | TinyBERT-style (attn + hidden) + softmax KD |
| LLM generation (white-box) | Token-level KD (forward KL on logits) + GKD |
| LLM generation (black-box) | Sequence-level KD (train on FM outputs) |

### Hyperparameters to sweep

```python
T         = [2, 4, 8, 16]           # KD temperature
alpha     = [0.1, 0.5, 0.9]         # weight of KD vs task loss
proj_dim  = [256, 512]              # adapter projection size
lr_scale  = [0.5, 1.0, 2.0]        # FM features may need different LR
```

---

## 12. References

- Bommasani et al. (2021). *On the Opportunities and Risks of Foundation Models.* Stanford HAI. https://arxiv.org/abs/2108.07258
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* NeurIPS 2019 Workshop. https://arxiv.org/abs/1910.01108
- Jiao et al. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding.* EMNLP 2020. https://arxiv.org/abs/1909.10351
- Wang et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression.* NeurIPS 2020. https://arxiv.org/abs/2002.10957
- Touvron et al. (2021). *Training data-efficient image transformers & distillation through attention.* ICML 2021. https://arxiv.org/abs/2012.12877
- Caron et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV 2021. https://arxiv.org/abs/2104.14294
- Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024. https://arxiv.org/abs/2304.07193
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021. https://arxiv.org/abs/2103.00020
- Mirzadeh et al. (2020). *Improved Knowledge Distillation via Teacher Assistant.* AAAI 2020. https://arxiv.org/abs/1902.03393
- Kim & Rush (2016). *Sequence-Level Knowledge Distillation.* EMNLP 2016. https://arxiv.org/abs/1606.07947
- Agarwal et al. (2023). *GKD: Generalized Knowledge Distillation for Auto-regressive Language Models.* https://arxiv.org/abs/2306.13649
- Xiong et al. (2024). *EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything.* CVPR 2024. https://arxiv.org/abs/2312.00863
- Huang et al. (2022). *DIST: Knowledge Distillation from A Stronger Teacher.* NeurIPS 2022. https://arxiv.org/abs/2205.10536
- Sun et al. (2024). *Logit Standardization in Knowledge Distillation.* CVPR 2024. https://arxiv.org/abs/2403.01427
