---
title: "LLM Fine-Tuning Methods — A Survey and Practical Map"
description: "A map of every major LLM fine-tuning approach — full fine-tuning, PEFT families, quantized fine-tuning, instruction tuning, and preference alignment — driven into a runnable comparison."
---

> A map of every major way to adapt a pretrained LLM: full fine-tuning, the three PEFT families (additive / reparameterization / selective), quantized fine-tuning, instruction tuning (SFT), and preference alignment (RLHF, DPO, GRPO and friends). Companion notebook: [finetuning_methods_survey.ipynb](https://github.com/morimori0456/ML_report/blob/main/llm/finetuning_methods_survey.ipynb) — six methods trained head-to-head on the same task on CPU, plus a from-scratch DPO implementation. GPU leg: [qlora_gpu_smoke_kaggle.ipynb](https://github.com/morimori0456/ML_report/blob/main/llm/qlora_gpu_smoke_kaggle.ipynb) — the same task fine-tuned with real QLoRA (Qwen2.5-1.5B-Instruct, NF4) on a Kaggle T4, with measured VRAM and all pipeline checks as asserts. For LoRA/QLoRA internals see [lora_qlora.md](lora_qlora.md); for an end-to-end domain project see [domain_finetune_driving.md](domain_finetune_driving.md).

"Fine-tuning" is not one technique — it is a family tree. Job postings say "experience with LLM fine-tuning"; what they mean spans everything from flipping a LoRA switch in a config file to running RLHF at scale. Knowing the whole map matters for two reasons: (1) method choice is a resource decision — the difference between full fine-tuning and LoRA is the difference between 8×A100 and one consumer GPU; (2) method choice is a *problem-type* decision — SFT teaches formats and skills, preference optimization teaches judgment, and using one where the other belongs is the most common failure in practice. This survey organizes the landscape, gives the math where it earns its keep, and drives every claim into a runnable comparison.

---

## Table of Contents
1. [The Landscape: Four Stages, Three Questions](#1-the-landscape-four-stages-three-questions)
2. [Full Fine-Tuning — the Baseline and Its Cost](#2-full-fine-tuning--the-baseline-and-its-cost)
3. [PEFT Family 1: Additive Methods (Adapters, Soft Prompts, IA3)](#3-peft-family-1-additive-methods-adapters-soft-prompts-ia3)
4. [PEFT Family 2: Reparameterization (LoRA and Its Descendants)](#4-peft-family-2-reparameterization-lora-and-its-descendants)
5. [PEFT Family 3: Selective Methods (BitFit, Layer Freezing)](#5-peft-family-3-selective-methods-bitfit-layer-freezing)
6. [Quantized Fine-Tuning (QLoRA and Beyond)](#6-quantized-fine-tuning-qlora-and-beyond)
7. [Instruction Tuning (SFT) — the Workhorse Stage](#7-instruction-tuning-sft--the-workhorse-stage)
8. [Preference Alignment — RLHF, DPO, and the Post-DPO Zoo](#8-preference-alignment--rlhf-dpo-and-the-post-dpo-zoo)
9. [Choosing a Method — a Decision Guide](#9-choosing-a-method--a-decision-guide)
10. [Tooling Ecosystem](#10-tooling-ecosystem)
11. [Common Pitfalls](#11-common-pitfalls)
12. [References](#12-references)

---

## 1. The Landscape: Four Stages, Three Questions

Modern LLM training is a pipeline; "fine-tuning" is everything after stage 1.

```
Stage 1: Pretraining            next-token prediction on trillions of tokens  →  base model
Stage 2: Continued pretraining  same objective, domain corpus (medicine, code, JP legal)
Stage 3: Instruction tuning     supervised (prompt, response) pairs           →  SFT model
Stage 4: Preference alignment   pairwise preferences / reward signals         →  aligned model
```

Any concrete fine-tuning project answers three nearly independent questions:

| Question | Options | Section |
|---|---|---|
| **What objective?** (which stage) | continued pretraining / SFT / preference optimization | §7, §8 |
| **Which parameters move?** | all (full FT) / added modules (additive) / low-rank deltas (LoRA) / a subset (selective) | §2–§5 |
| **What precision is the frozen base?** | fp32 / bf16 / 8-bit / 4-bit (QLoRA) | §6 |

These axes compose: "QLoRA SFT" = SFT objective + LoRA parameters + 4-bit base. "DPO with LoRA" = preference objective + LoRA parameters. Most confusion about fine-tuning dissolves once you see that the *objective* axis and the *parameter* axis are orthogonal.

### Key insight
> **PEFT methods are not competitors to SFT/DPO — they are how you afford them.** LoRA is not an alternative to instruction tuning; it is a way to run instruction tuning (or DPO, or continued pretraining) with 100–1000× fewer trainable parameters. Interview answers and design docs that treat "LoRA vs DPO" as a choice are mixing up the axes.

**Why this matters**: the notebook exploits this orthogonality directly — it holds the objective fixed (one SFT task) and sweeps the parameter axis (6 methods), then holds the parameters fixed (LoRA) and swaps the objective (SFT → DPO).

---

## 2. Full Fine-Tuning — the Baseline and Its Cost

Full fine-tuning (full FT) updates every weight with plain backprop. It is the quality ceiling that PEFT methods are measured against, and its cost structure explains why PEFT exists.

For a model with $N$ parameters trained with Adam in mixed precision, per-parameter memory is roughly:

$$
\underbrace{2N}_{\text{bf16 weights}} + \underbrace{4N}_{\text{fp32 master copy}} + \underbrace{4N + 4N}_{\text{Adam } m, v} + \underbrace{2N\ (\text{or } 4N)}_{\text{gradients}} \approx 16N \text{ bytes}
$$

A 7B model therefore wants ~112 GB before activations — multiple A100s with ZeRO/FSDP sharding, versus a single 24 GB card for QLoRA. (Full breakdown with activation memory: [lora_qlora.md §1](lora_qlora.md).)

| | Full FT | PEFT (LoRA-class) |
|---|---|---|
| Quality ceiling | highest (large data, large shift) | matches full FT on most instruction/domain tasks |
| Memory | ~16 bytes/param + activations | base model forward + tiny optimizer state |
| Catastrophic forgetting | severe without care | structurally bounded (base frozen) |
| Artifact per task | full checkpoint (14 GB for 7B bf16) | adapter file (tens of MB) |
| Multi-task serving | one model per task | one base + N adapters, hot-swappable |

**When full FT is still right**: continued pretraining on billions of domain tokens; changing the model's language or modality; small models (< 1B) where the cost argument disappears; squeezing the last points on a benchmark with abundant data. Frontier labs full-FT; most application teams should not.

**Why this matters**: every method in §3–§5 is an answer to "which 0.01–1% of directions in weight space do we allow to move?" — you can only evaluate those answers against the 100% baseline, which the notebook trains first.

---

## 3. PEFT Family 1: Additive Methods (Adapters, Soft Prompts, IA3)

Additive methods bolt **new** small modules onto a frozen base model and train only those.

### 3.1 Adapter layers (Houlsby 2019, Pfeiffer 2020)

The original PEFT method: insert a bottleneck MLP after attention and/or FFN sublayers:

$$
h \leftarrow h + f(hW_{\text{down}})W_{\text{up}}, \qquad W_{\text{down}} \in \mathbb{R}^{d \times r},\ W_{\text{up}} \in \mathbb{R}^{r \times d},\ r \ll d
$$

Trains ~1–4% of parameters with near-full-FT quality on NLU benchmarks. The catch: the adapter sits **in series** in the forward pass, adding inference latency that cannot be merged away. This is the main reason LoRA (parallel, mergeable) displaced adapters.

### 3.2 Soft prompts: Prompt Tuning, Prefix Tuning, P-Tuning

Instead of editing weights, prepend trainable *virtual tokens* — continuous vectors that never correspond to real vocabulary:

| Method | What is learned | Where it is injected | Params for GPT-2-scale |
|---|---|---|---|
| **Prompt Tuning** (Lester 2021) | $P \in \mathbb{R}^{k \times d}$ — $k$ embedding vectors | input embedding layer only | ~15K ($k{=}20$) |
| **Prefix Tuning** (Li & Liang 2021) | virtual **key/value vectors per layer** | every attention layer's KV cache | ~180K–590K |
| **P-Tuning v2** (Liu 2021) | per-layer prefixes (≈ prefix tuning generalized) | every layer | similar to prefix |

Prompt tuning is the most parameter-frugal method in existence, but it only modulates the input — with a small base model it has the least capacity to teach *new* behavior, and it is notoriously sensitive to initialization and learning rate (typical lr: 1e-2 to 3e-1, orders above LoRA's). Its quality gap versus full FT closes as the base model grows (the original paper's headline result: at 10B+ params, prompt tuning ≈ full FT on SuperGLUE).

Deployment quirk: soft prompts occupy $k$ positions of context at inference, and prefix tuning permanently reserves KV-cache slots.

### 3.3 IA3 (Liu 2022, "Infused Adapter by Inhibiting and Amplifying Inner Activations")

Learn per-channel **rescaling vectors** on keys, values, and FFN activations:

$$
k \leftarrow l_k \odot k, \qquad v \leftarrow l_v \odot v, \qquad h_{\text{ffn}} \leftarrow l_{\text{ff}} \odot h_{\text{ffn}}
$$

Only $l_k, l_v, l_{\text{ff}}$ (initialized to ones) are trained — even fewer parameters than LoRA, and like LoRA the vectors **merge into the base weights** at inference (elementwise multiply), so zero latency overhead. IA3 was built for few-shot task adaptation (the T-Few recipe) and is a strong low-budget baseline, but its expressivity ceiling is real: it can only rescale existing feature directions, never create new ones.

### Key insight
> **Additive methods differ in *where* they inject capacity, and that placement is their capacity.** Prompt tuning touches only the input (cheapest, weakest), prefix tuning touches every layer's attention (stronger, still frozen weights), adapters/IA3 touch activations directly. Caveat from the notebook: on a task well within the base model's reach, *every* method — even 15K-parameter prompt tuning — hits the exact-match ceiling; capacity differences surface in eval-loss margins and convergence speed there, and in accuracy only as tasks get harder (the prompt-tuning paper's scale result is the flip side of the same coin).

**Why this matters**: when a recruiter or paper says "PEFT," people picture LoRA — but adapters and soft prompts are what you'll meet in the pre-2021 literature and in multi-task serving systems (one soft prompt per tenant is far cheaper to hot-swap than one adapter per tenant).

---

## 4. PEFT Family 2: Reparameterization (LoRA and Its Descendants)

LoRA (Hu 2021) reparameterizes the weight *update* as a low-rank product, trained **in parallel** with the frozen weight:

$$
W' = W + \Delta W = W + \frac{\alpha}{r} BA, \qquad A \in \mathbb{R}^{r \times d_{\text{in}}},\ B \in \mathbb{R}^{d_{\text{out}} \times r}
$$

$B$ starts at zero (so training starts exactly at the base model), and after training $BA$ **merges into $W$** — zero inference overhead. This mergeability plus consistent near-full-FT quality made LoRA the de facto standard. Full internals — rank/α scaling, initialization, which modules to target, memory math — are in [lora_qlora.md](lora_qlora.md); here is the family at a glance:

| Variant | Idea | When it earns its keep |
|---|---|---|
| **LoRA** (2021) | low-rank delta, merge at inference | the default; start here |
| **QLoRA** (2023) | LoRA on a 4-bit NF4-quantized frozen base | single-GPU fine-tuning of 7B–70B (§6) |
| **DoRA** (2024) | decompose $W$ into magnitude × direction; LoRA on direction only | +1–3 points over LoRA at low rank, ~same cost |
| **AdaLoRA** (2023) | SVD-form deltas, prune rank per-module by importance | fixed param budget spread over many modules |
| **rsLoRA** (2023) | scale by $\alpha/\sqrt{r}$ instead of $\alpha/r$ | stability when sweeping to high ranks (r ≥ 64) |
| **LoRA+** (2024) | lr(B) ≫ lr(A) (~16×) | faster convergence, esp. large models |
| **VeRA** (2023) | shared frozen random $A,B$; train tiny per-layer scaling vectors | extreme parameter budgets (10× fewer than LoRA) |
| **PiSSA** (2024) | initialize $A,B$ from top singular vectors of $W$ | faster convergence than zero-init |

Practical defaults that survive contact with most projects: `r=8–16`, `α=2r`, target **all linear modules** (attention + MLP) rather than just `q,v`, lr `1e-4`–`3e-4` (10× full-FT's), dropout 0.05.

### Key insight
> **LoRA won because of deployment, not just training economics.** Adapters also train ~1% of params — but LoRA's delta merges into the base weights, giving *exactly zero* inference-time overhead, and unmerged adapters are tiny swappable artifacts (multi-tenant LoRA serving à la S-LoRA/vLLM). A method that saves training memory but taxes every inference forever loses in production.

**Why this matters**: in the notebook, LoRA at r=8 on all linear modules (~0.7% of params) matches full fine-tuning's eval loss to four decimals — the observation that made it the industry default.

---

## 5. PEFT Family 3: Selective Methods (BitFit, Layer Freezing)

Selective methods train a **subset of existing** parameters — nothing added, nothing reparameterized.

- **BitFit** (Ben Zaken 2021): train only bias terms (~0.05–0.1% of params). Shockingly competitive on classification-style tasks; weak for generation. Great as a *diagnostic*: if BitFit already solves your task, the task barely needed adaptation, and the knowledge was already in the base model.
- **Layer freezing / partial FT**: unfreeze the top-$k$ transformer blocks (plus, usually, the LM head). The classic transfer-learning move; a reasonable middle ground when you have moderate data and no PEFT tooling available.
- **Embedding-only tuning**: train input embeddings for new-token/vocabulary extension work (domain jargon, new language scripts) — usually *combined* with LoRA on the body.

| Family | Representative | Trainable % | Added inference cost | Mergeable |
|---|---|---|---|---|
| Additive – series | Adapters | 1–4% | yes (extra MLP) | no |
| Additive – soft prompt | Prompt/Prefix Tuning | 0.01–0.1% | yes (context slots) | no |
| Additive – rescaling | IA3 | ~0.03% | no | yes |
| Reparameterization | LoRA / DoRA | 0.1–1% | no | yes |
| Selective | BitFit | ~0.06% | no | (already in-place) |
| — | Full FT | 100% | no | — |

**Why this matters**: the three families answer "where does the model store what it learns?" differently — new modules, low-rank deltas, or existing slack capacity. The notebook's parameter-count-vs-accuracy plot is this table made empirical.

---

## 6. Quantized Fine-Tuning (QLoRA and Beyond)

Quantization changes the *storage precision of the frozen base*, and composes with any PEFT method (in practice: LoRA).

**QLoRA** (Dettmers 2023) = base weights in 4-bit **NF4** (a quantile-optimal 4-bit float for normally-distributed weights) + double quantization of the scale constants + paged optimizer states + LoRA adapters in bf16. Gradients flow *through* the dequantized weights into the adapters; the base never updates, so quantization error never accumulates. Result: 65B fine-tuning on a single 48 GB card, 7B on a free Colab T4, with quality matching 16-bit LoRA on instruction benchmarks. Full mechanics: [lora_qlora.md §4–5](lora_qlora.md); GPU recipe: [lora_qlora_finetune.ipynb](https://github.com/morimori0456/ML_report/blob/main/llm/lora_qlora_finetune.ipynb).

Ecosystem notes, current as of mid-2026:

- **QA-LoRA / GPTQ-LoRA**: quantization-aware variants so the *merged* model stays quantized (plain QLoRA merges back to 16-bit).
- **Unsloth**: hand-written Triton kernels around the QLoRA recipe; ~2× faster, ~60% less VRAM, same math — the default hobbyist/edge choice.
- The bf16-LoRA vs 4-bit-QLoRA decision is purely a VRAM decision: if bf16 base + activations fit, prefer plain LoRA (faster steps, no dequant overhead).

**Why this matters**: QLoRA is the reason "fine-tune a 7B model" stopped being a lab-only sentence. Hardware sizing tables for concrete model/GPU pairs: [domain_finetune_driving.md §9](domain_finetune_driving.md).

---

## 7. Instruction Tuning (SFT) — the Workhorse Stage

Supervised fine-tuning turns a base next-token predictor into an assistant by training on (prompt, response) pairs. Whatever parameter method you chose (§2–§5), the *objective mechanics* below decide whether SFT works.

**Loss masking (prompt masking).** Compute cross-entropy only on response tokens — set prompt-token labels to `-100` so they are skipped:

$$
\mathcal{L} = -\sum_{t \in \text{response}} \log p_\theta(y_t \mid y_{<t}, x)
$$

Without masking, the model spends its capacity learning to predict your *instructions* (an echo of pretraining) instead of learning to *answer* them. On short-response data the effect is dramatic; the companion driving-domain notebook demonstrates a masked-vs-unmasked ablation.

**Chat templates.** Instruction models are trained with special role markup (`<|user|>`, `<|assistant|>`, `<|im_start|>`…). The single most common SFT bug is a train/inference template mismatch — always format via `tokenizer.apply_chat_template`, never hand-rolled f-strings.

**EOS discipline.** Every training response must end with the EOS token, or the fine-tuned model never learns to stop.

**Packing.** Concatenate short examples into full-length sequences (separated by EOS) so no compute is wasted on padding — 2–5× throughput on short-example datasets. Correct packing masks cross-example attention (block-diagonal attention or position-id resets).

**Data quality beats data volume.** LIMA (Zhou 2023) hit strong instruction-following with 1,000 hand-curated examples; every practitioner survey since has replicated the direction. A few hundred *clean, diverse, deduplicated* examples with consistent formatting outperform 100K scraped ones. Data design methodology — coverage matrices, leakage-free splits, quality gates: [domain_finetune_driving.md §3](domain_finetune_driving.md).

### Key insight
> **SFT can only teach what a correct answer looks like — it cannot teach that one answer is better than another.** Every SFT token is treated as equally correct; the model has no notion of "this response is valid but worse." That gap — relative quality — is exactly what preference alignment (§8) exists to close, and why the two stages are run in sequence rather than either alone.

---

## 8. Preference Alignment — RLHF, DPO, and the Post-DPO Zoo

Preference methods optimize *relative judgments* — "response A > response B" — rather than absolute targets.

### 8.1 RLHF with PPO (the original recipe)

InstructGPT (Ouyang 2022) three-step pipeline:

1. SFT on demonstrations.
2. Train a **reward model** $r_\phi(x, y)$ on human pairwise preferences via the Bradley–Terry likelihood: $p(y_w \succ y_l) = \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))$.
3. Optimize the policy with PPO against the learned reward, minus a KL penalty tethering it to the SFT model:

$$
\max_\pi \; \mathbb{E}_{y \sim \pi} \left[ r_\phi(x, y) \right] - \beta \, \mathbb{D}_{\text{KL}}\!\left[ \pi(y|x) \,\|\, \pi_{\text{ref}}(y|x) \right]
$$

Powerful but operationally heavy: four models in memory (policy, reference, reward, value), online generation inside the training loop, and PPO's notorious hyperparameter sensitivity. The KL term is load-bearing — without it the policy **reward-hacks**: it finds degenerate outputs the reward model scores highly (repetition, sycophancy, length inflation).

### 8.2 DPO — Direct Preference Optimization (Rafailov 2023)

DPO's derivation is one of the cleanest results in the field: the KL-constrained RLHF objective has a closed-form optimal policy, and inverting it expresses the *implicit reward* as $\hat r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$. Substituting into Bradley–Terry eliminates the reward model entirely:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \! \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

Two models instead of four, no sampling loop, no reward model, trains like supervised learning — this is why DPO became the default alignment method for open models. $\beta$ (typically 0.1–0.5) plays the same tether role as the RLHF KL coefficient. The notebook implements this loss **from scratch in ~15 lines** and shows the reward margin separating chosen from rejected behavior.

### 8.3 The post-DPO zoo

| Method | One-line delta vs DPO | Data needed | Reference model |
|---|---|---|---|
| **IPO** (2023) | replaces log-sigmoid with a squared regression target — resists overfitting to preferences | pairs | yes |
| **KTO** (2024) | prospect-theory loss on *unpaired* good/bad labels | 👍/👎 only, no pairs | yes |
| **ORPO** (2024) | folds an odds-ratio penalty into SFT — one stage, no separate alignment run | pairs | **no** |
| **SimPO** (2024) | length-normalized implicit reward, no reference model | pairs | **no** |
| **GRPO** (2024, DeepSeek) | back to online RL, but replaces PPO's value network with group-relative advantages (sample $G$ responses, normalize rewards within the group) | verifiable reward function | yes (KL) |

### 8.4 RLVR — alignment's second life for *reasoning*

GRPO's real significance: with **verifiable rewards** (unit tests for code, exact answers for math — no learned reward model, no preference data), online RL trains chain-of-thought reasoning. DeepSeek-R1 (2025) showed long-form reasoning emerging from pure RLVR on a base model. This moved RL fine-tuning from "safety/style polish" to "capability training," and it is the most active fine-tuning frontier as of 2026.

### Key insight
> **The entire alignment zoo is variations on one tension: maximize a preference signal while staying close to a reference policy.** RLHF enforces closeness with an explicit KL penalty; DPO bakes it into the loss via $\pi_{\text{ref}}$; ORPO/SimPO approximate it with SFT anchoring or length normalization. When you see a new alignment paper, first ask: what is the preference signal, and what keeps the policy tethered?

**Why this matters**: "fine-tuning experience" in 2026 job descriptions increasingly means *this* — SFT+DPO pipelines and RLVR for domain reasoning — not just LoRA mechanics.

---

## 9. Choosing a Method — a Decision Guide

Answer the three §1 questions in order:

**Q1 — Objective (what is broken?)**

| Symptom | Stage to run |
|---|---|
| Model doesn't know domain language/facts at all | continued pretraining (then SFT) |
| Model knows the domain but won't follow format/instructions | SFT |
| SFT model follows instructions but makes poor choices between valid outputs (tone, safety, verbosity, reasoning quality) | preference alignment (DPO first; GRPO if rewards are verifiable) |
| Facts change weekly | **stop — use RAG**, not fine-tuning ([domain_finetune_driving.md §1](domain_finetune_driving.md)) |

**Q2 — Parameters (what hardware/data do you have?)**

| Situation | Method |
|---|---|
| Default: 100s–100Ks examples, one task | **LoRA** (r=8–16, all linear modules) |
| VRAM-starved for the model size you need | **QLoRA** (+ Unsloth) |
| Tiny data (< ~100 examples), quick adaptation | IA3 or prompt tuning; consider few-shot prompting instead |
| Many tenants/tasks sharing one deployed base | LoRA adapters (multi-adapter serving) or soft prompts |
| Billions of domain tokens, big distribution shift | full FT / continued pretraining with FSDP |
| Sanity-check "does this task need FT at all?" | BitFit probe |

**Q3 — Precision**: bf16 LoRA if it fits; NF4 QLoRA if it doesn't. Never quantize below 4-bit for training.

---

## 10. Tooling Ecosystem

| Tool | Layer | What it gives you | Reach for it when |
|---|---|---|---|
| **peft** (HF) | library | LoRA/DoRA/IA3/prompt/prefix as `get_peft_model(model, cfg)` | you write your own training loop (this repo's notebooks) |
| **trl** (HF) | library | `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, reward modeling | standard SFT/alignment with HF stack |
| **Unsloth** | kernels | 2× faster QLoRA, 60% less VRAM, drop-in | single-GPU budgets |
| **axolotl** | YAML config layer | full pipelines (SFT/DPO, packing, FSDP) with no code | reproducible recipes, quick experiments |
| **LLaMA-Factory** | YAML + WebUI | similar to axolotl, broad model zoo, GUI | low-code experimentation |
| **torchtune** | PyTorch-native recipes | hackable pure-PyTorch fine-tuning, no HF Trainer | you want to read/own every line |
| **NeMo / Megatron** | framework | tensor/pipeline parallel full FT | multi-node industrial training |

Skill-building path this repo follows: **peft + hand-written loops first** (you see the mechanics — this notebook), then trl (production ergonomics), then config layers (throughput).

---

## 11. Common Pitfalls

1. **Mixing up the axes** — "should I use LoRA or DPO?" is a category error; one is a parameter method, the other an objective (§1).
2. **Skipping prompt masking** — training loss looks fine, but capacity is spent predicting instructions; short-response tasks degrade badly (§7).
3. **Chat-template mismatch** between training and inference — the model was tuned on markup it never sees at inference. Always `apply_chat_template`.
4. **Missing EOS** on training targets — generations never terminate.
5. **LoRA lr copied from full FT** (or vice versa) — LoRA wants ~10× full-FT lr; prompt tuning wants ~100×. Reusing 5e-5 across methods silently cripples the additive methods (the notebook sweeps per-method lrs for exactly this reason).
6. **Judging fine-tuning by training loss** — evaluate the *task*: schema adherence, exact match, held-out combinations, judge scores ([domain_finetune_driving.md §7](domain_finetune_driving.md)).
7. **DPO on a non-SFT'd model** — DPO adjusts relative preferences between behaviors the model can already produce; run SFT first (§8).
8. **β too low in DPO** (or KL coefficient too low in RLHF) — the policy drifts far from the reference and degenerates: repetition, length explosion, reward hacking.
9. **Ignoring catastrophic forgetting** — even LoRA shifts general behavior at high rank/lr; keep a general-capability eval in the loop ([domain_finetune_driving.md §8](domain_finetune_driving.md)).
10. **Fine-tuning to inject volatile facts** — they go stale and small models hallucinate them; that's RAG's job.

---

## 12. References

- Houlsby et al., 2019. *Parameter-Efficient Transfer Learning for NLP* (Adapters). [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
- Li & Liang, 2021. *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
- Lester et al., 2021. *The Power of Scale for Parameter-Efficient Prompt Tuning*. [arXiv:2104.08691](https://arxiv.org/abs/2104.08691)
- Liu et al., 2021. *P-Tuning v2*. [arXiv:2110.07602](https://arxiv.org/abs/2110.07602)
- Ben Zaken et al., 2021. *BitFit: Simple Parameter-efficient Fine-tuning*. [arXiv:2106.10199](https://arxiv.org/abs/2106.10199)
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Liu et al., 2022. *Few-Shot PEFT is Better and Cheaper than In-Context Learning* (IA3 / T-Few). [arXiv:2205.05638](https://arxiv.org/abs/2205.05638)
- Dettmers et al., 2023. *QLoRA: Efficient Finetuning of Quantized LLMs*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Zhang et al., 2023. *AdaLoRA*. [arXiv:2303.10512](https://arxiv.org/abs/2303.10512) · Kalajdzievski, 2023. *rsLoRA*. [arXiv:2312.03732](https://arxiv.org/abs/2312.03732) · Liu et al., 2024. *DoRA*. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) · Kopiczko et al., 2023. *VeRA*. [arXiv:2310.11454](https://arxiv.org/abs/2310.11454) · Hayou et al., 2024. *LoRA+*. [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)
- Zhou et al., 2023. *LIMA: Less Is More for Alignment*. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- Ouyang et al., 2022. *Training language models to follow instructions with human feedback* (InstructGPT/RLHF). [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Rafailov et al., 2023. *Direct Preference Optimization*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Azar et al., 2023. *IPO*. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036) · Ethayarajh et al., 2024. *KTO*. [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) · Hong et al., 2024. *ORPO*. [arXiv:2403.07691](https://arxiv.org/abs/2403.07691) · Meng et al., 2024. *SimPO*. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
- Shao et al., 2024. *DeepSeekMath* (GRPO). [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) · DeepSeek-AI, 2025. *DeepSeek-R1*. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
