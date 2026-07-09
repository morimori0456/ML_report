# Domain-Specific LLM Fine-Tuning — LoRA in Practice on a Driving-Scenario Generator

> End-to-end guide for adapting a pretrained LLM to a narrow domain task with parameter-efficient fine-tuning (PEFT): data design, LoRA configuration, training, evaluation, and deployment. Companion notebook: [domain_finetune_driving.ipynb](domain_finetune_driving.ipynb) — a complete CPU-runnable pipeline that turns a small base model into an autonomous-driving scenario-description generator. For LoRA/QLoRA internals see [lora_qlora.md](lora_qlora.md); for the GPU/Colab QLoRA recipe see [lora_qlora_finetune.ipynb](lora_qlora_finetune.ipynb).

Prompting gets you far, but some tasks need the model's weights to move: strict output schemas, domain vocabulary the base model rarely saw, latency budgets that rule out long few-shot prompts, or on-prem constraints that rule out API models. Domain fine-tuning with LoRA is the highest-leverage version of this skill — a few hundred good examples and a few million trainable parameters routinely beat elaborate prompt engineering on narrow tasks. This report walks the full pipeline on a concrete, portfolio-ready case: generating structured driving-scenario descriptions of the kind used in scenario-based safety testing.

---

## Table of Contents
1. [Fine-Tune, Prompt, or RAG?](#1-fine-tune-prompt-or-rag)
2. [The Pipeline at a Glance](#2-the-pipeline-at-a-glance)
3. [Data Design — Where Projects Are Won or Lost](#3-data-design--where-projects-are-won-or-lost)
4. [The Case Study: Driving-Scenario Description Generation](#4-the-case-study-driving-scenario-description-generation)
5. [LoRA Configuration That Matters](#5-lora-configuration-that-matters)
6. [Training Mechanics](#6-training-mechanics)
7. [Evaluation — Beyond Eyeballing Generations](#7-evaluation--beyond-eyeballing-generations)
8. [Catastrophic Forgetting and How LoRA Contains It](#8-catastrophic-forgetting-and-how-lora-contains-it)
9. [Hardware Reality: What Runs Where](#9-hardware-reality-what-runs-where)
10. [Deployment: Adapters, Merging, Serving](#10-deployment-adapters-merging-serving)
11. [Common Pitfalls](#11-common-pitfalls)
12. [References](#12-references)

---

## 1. Fine-Tune, Prompt, or RAG?

The first engineering decision is whether to fine-tune at all.

| Approach | Changes | Best for | Fails when |
|---|---|---|---|
| **Prompting / few-shot** | nothing | general tasks the base model already does; fast iteration | strict schemas, domain jargon, long prompts blow the latency/token budget |
| **RAG** (Retrieval-Augmented Generation) | context only | injecting *facts* that change often (docs, tickets, specs) | the problem is *behavior/format*, not missing facts |
| **Fine-tuning (LoRA)** | weights (adapters) | teaching *behavior*: output schema, style, domain vocabulary, task procedure | facts churn daily (retrain treadmill); < ~50 usable examples |
| **Full fine-tuning** | all weights | large data + large distribution shift (new language, code models) | almost everything else — cost, forgetting, checkpoint sprawl |

### Key insight
> **Fine-tune for behavior, retrieve for knowledge.** The most common failure is fine-tuning to inject facts (they go stale, and small models hallucinate them) or building RAG to fix formatting (retrieval cannot change how the model writes). Schema adherence, tone, and domain phrasing are weight-level properties — that is LoRA's territory.

**Why this matters**: the decision table above is asked in nearly every ML system-design interview, and getting it wrong in production costs months. The case study below is deliberately a *behavior* problem — teaching a schema — so fine-tuning is the right tool.

---

## 2. The Pipeline at a Glance

```
1. Task definition      output schema, quality bar, eval metric   <- write this FIRST
2. Data                 collect or synthesize; split train/val/test by scenario, not by row
3. Base model           smallest model that can do the task; instruction-tuned if chat-style
4. PEFT config          LoRA r / alpha / target modules / dropout
5. Training             prompt-masked causal-LM loss, 1-5 epochs, cosine or constant LR
6. Evaluation           perplexity + task metrics (format adherence, field accuracy) + human/LLM judge
7. Deployment           keep adapters separate, or merge_and_unload() for a single artifact
```

Steps 1 and 7 are the ones most tutorials skip, and the ones that distinguish a portfolio project from a toy: define the metric before training, and show the deployment story.

---

## 3. Data Design — Where Projects Are Won or Lost

### 3.1 Format: prompt + completion with loss masking

For a causal LM, each example is one token sequence, but the loss should apply **only to the completion tokens**:

$$\mathcal{L} = -\frac{1}{|C|}\sum_{t \in C} \log p_\theta(x_t \mid x_{<t}), \qquad C = \text{completion token positions}$$

Masking the prompt (setting its labels to `-100` in PyTorch convention) prevents the model from spending capacity learning to reproduce your prompt boilerplate and measurably improves small-data fine-tuning. Chat models add a template layer (`tokenizer.apply_chat_template`) — using the *wrong* template silently degrades quality because the model sees delimiters it never saw in training.

### 3.2 Synthetic data is legitimate — with structure

When real domain data is scarce or unshareable, programmatic generation works well for *schema-teaching* tasks: define parameter vocabularies, sample combinations, and render deterministic gold outputs from templates with controlled linguistic variation. The notebook does exactly this. Rules of thumb:

- **Coverage beats volume**: 300–1,000 examples spanning the parameter space outperform 10,000 near-duplicates. LIMA (Zhou et al. 2023) showed 1,000 curated examples suffice for strong instruction-following.
- **Vary surface form, fix semantics**: multiple phrasings per template slot, so the model learns the schema rather than one string.
- **Hold out by scenario, not by row**: if train and test contain the same parameter combination in different phrasings, your eval is measuring memorization. Split on the underlying parameter tuples.
- If you generate data with a *stronger LLM* (the common industrial shortcut), you are doing sequence-level knowledge distillation — see [foundation_model_distillation.md](foundation_model_distillation.md) for the framing and the API terms-of-service cautions.

### 3.3 Quality gates

Deduplicate (exact + near-duplicate), validate every gold output against the schema with the same checker you will use at eval time, and cap example length — a handful of extreme-length outliers dominates padding cost and destabilizes small-batch training.

---

## 4. The Case Study: Driving-Scenario Description Generation

### 4.1 Why this task

Scenario-based testing is how autonomous-driving systems are validated: regulations and standards (UNECE, ISO 34502) require demonstrating behavior across a structured catalog of scenarios, typically authored in formats like ASAM OpenSCENARIO. Turning structured scenario parameters into precise natural-language descriptions (and eventually the reverse) is real toolchain work: test-report generation, scenario-catalog documentation, requirement traceability.

### 4.2 Task definition

Input: a structured parameter block. Output: a description following a fixed 3-sentence schema.

```
### Scenario
road: highway | lanes: 3 | weather: heavy rain | time: night
ego_speed: 100 km/h | event: cut-in | actor: truck | actor_speed: 80 km/h
### Description
The ego vehicle is driving at 100 km/h in the center lane of a 3-lane highway
at night under heavy rain. A truck traveling at 80 km/h initiates a cut-in
from the adjacent lane into the ego lane. The ego vehicle must detect the
maneuver and adjust speed to maintain a safe gap on the wet road surface.
```

The schema demands: (1) ego state + environment, (2) actor + event, (3) required ego response including a condition-dependent consequence (wet road, low visibility, ...). The eval checker verifies all schema fields appear — a deterministic, CI-friendly metric.

### 4.3 What the notebook demonstrates

A small base model (distilgpt2, 82M parameters — chosen so the whole pipeline runs on CPU in minutes) cannot produce this schema zero-shot. After LoRA fine-tuning on ~400 synthetic examples with prompt-masked loss, it produces schema-conformant descriptions for held-out parameter combinations. The point is not the model — it is that **the identical pipeline scales to Llama/Qwen on a GPU by changing two lines** (model name, target modules).

---

## 5. LoRA Configuration That Matters

LoRA reparameterizes each chosen weight as $W = W_0 + \frac{\alpha}{r} BA$ with $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$ trainable and $W_0$ frozen (full math in [lora_qlora.md](lora_qlora.md)). What actually moves the needle, in order:

| Knob | Effect | Practical default |
|---|---|---|
| **target_modules** | which layers get adapters | all attention projections; adding MLP projections helps more than raising r |
| **learning rate** | dominant hyperparameter | 1e-4 to 3e-4 (10-100x higher than full FT) |
| **r** (rank) | adapter capacity | 8-16; returns diminish fast beyond 32 for narrow tasks |
| **alpha** | effective scale α/r | set α = 2r and leave it |
| **dropout** | regularization | 0.05-0.1 for < 1k examples |

- GPT-2-family models fuse Q,K,V into one `c_attn` module — target that. Llama-family exposes `q_proj,k_proj,v_proj,o_proj` (+ `gate_proj,up_proj,down_proj` for MLP).
- Trainable-parameter budget: for the notebook's config, ~0.5% of total parameters. Print it (`model.print_trainable_parameters()`) in every run log — it is the first sanity check that the config bit.
- **QLoRA** = same adapters on a 4-bit NF4-quantized frozen base — memory drops ~4x, quality within noise of fp16 LoRA (Dettmers et al. 2023). It changes *where you can train* (24GB GPU for 7-13B), not *how you design the task*.

---

## 6. Training Mechanics

Small-data fine-tuning is closer to careful cooking than to pretraining:

- **Epochs**: 2-5. Watch validation loss per epoch — schema tasks converge fast, and epoch 10 is usually memorizing.
- **Batching**: pad within batch, mask padding in both attention and labels. Effective batch 8-32 via gradient accumulation if memory-bound.
- **Optimizer**: AdamW, weight decay 0.0-0.01 on adapters, linear or cosine decay with ~3% warmup. For CPU runs, plain constant LR is fine.
- **EOS discipline**: append the tokenizer's EOS to every completion, or the fine-tuned model will not learn to stop — the single most common "my generations ramble forever" bug.
- **Seed the run and log everything** (see [experiment_tracking.md](../experiment_tracking/experiment_tracking.md)): base model revision, data hash, LoRA config, loss curves.

---

## 7. Evaluation — Beyond Eyeballing Generations

Three layers, cheapest first:

1. **Held-out loss / perplexity** ($\mathrm{PPL} = e^{\mathcal{L}}$): tracks learning, catches overfitting (train ↓, val ↑), but does not measure generation quality — teacher-forced loss and free-running generation can diverge (exposure bias).
2. **Deterministic task metrics on generations**: for schema tasks, a rule-based checker — required fields present, values copied correctly from the input (no hallucinated speeds!), sentence count in range. This is the metric that belongs in CI. The notebook implements field-level checks and reports adherence before vs after fine-tuning.
3. **Judgment layer**: human review of a sample, or LLM-as-judge with a rubric for fluency/faithfulness. Use it to catch what rules cannot (awkward phrasing, subtle contradictions), not as the primary metric — judge models drift and have biases (position, verbosity).

### Key insight
> **Value-copy accuracy is the metric that matters in structured generation.** A model can be perfectly fluent, perfectly formatted, and quietly write "80 km/h" where the input said "100 km/h". Always include a check that extracts values from the generation and compares them to the source parameters — hallucinated specifics are the failure mode that kills domain deployments.

---

## 8. Catastrophic Forgetting and How LoRA Contains It

Full fine-tuning on a narrow distribution degrades general capabilities (catastrophic forgetting). LoRA structurally limits this: $W_0$ is frozen, and the update is rank-$r$, so the model cannot drift arbitrarily far. Additional containment when it still bites:

- **Mix in general data**: 5-10% generic instruction data alongside domain examples preserves general behavior.
- **Lower LR / fewer epochs** before reaching for bigger hammers.
- **Adapters are removable**: serving with adapters (rather than merging) lets you A/B the base model instantly — the ultimate forgetting insurance.
- Measure it: run a small general benchmark (or a fixed set of generic prompts) before and after, not just the domain metric.

---

## 9. Hardware Reality: What Runs Where

| Setup | What is feasible | Notes |
|---|---|---|
| **CPU only** (this repo's notebook) | LoRA on ≤ ~150M models, hundreds of examples, minutes | full pipeline practice; distilgpt2/GPT-2-small/TinyLlama-class |
| **1x consumer GPU (8-16GB)** | LoRA on 1-3B fp16/bf16; QLoRA on 7B | the sweet spot for personal portfolio work |
| **1x 24GB (3090/4090/L4)** | QLoRA on 7-13B comfortably | Colab Pro / cheap cloud territory; see [lora_qlora_finetune.ipynb](lora_qlora_finetune.ipynb) |
| **Multi-GPU / A100+** | full FT on small models, LoRA on 70B | needs accelerate/FSDP or DeepSpeed configs |

bitsandbytes (the 4-bit backbone of QLoRA) requires CUDA — on CPU-only machines, skip quantization and shrink the model instead. The transferable skill is the pipeline, not the parameter count.

---

## 10. Deployment: Adapters, Merging, Serving

Two end states:

```python
# A) merge: one standalone model artifact, zero inference overhead
merged = peft_model.merge_and_unload()
merged.save_pretrained("scenario-gen-v1")

# B) keep adapters separate: ~10-40MB file, hot-swappable per task
peft_model.save_pretrained("scenario-gen-adapter-v1")   # adapter_model.safetensors only
```

| | Merged | Separate adapters |
|---|---|---|
| Inference overhead | none | none if merged at load; small if dynamic |
| Artifact size | full model | megabytes |
| Multi-task serving | one copy per task | one base + N adapters (vLLM/LoRAX serve many adapters over one base) |
| Rollback / A-B | redeploy | swap adapter |

For a portfolio repo, ship the adapter + a reproduction script: it is small enough for git, and it proves you understand the deployment trade-off.

---

## 11. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Loss on prompt tokens | wasted capacity, prompt echoing | labels = -100 on prompt and padding positions |
| Missing EOS on completions | generations never stop | append `eos_token` to every training completion |
| Wrong / no chat template on an instruct model | mysteriously degraded quality | `tokenizer.apply_chat_template`; verify rendered strings by printing one |
| Train/test split by row on templated data | inflated metrics (memorization) | split by underlying parameter combination |
| Value hallucination | fluent output, wrong numbers | value-copy checker in eval; more coverage of numeric slots |
| r too large on tiny data | val loss rises after epoch 1 | r=8, dropout 0.1, fewer epochs |
| LR copied from full-FT recipes (1e-5) | adapter barely learns | LoRA wants 1e-4 to 3e-4 |
| Fine-tuning to inject facts | confident stale/hallucinated answers | RAG for facts; fine-tune for behavior (Section 1) |
| Evaluating only perplexity | great PPL, broken generations | generation-level checks (Section 7 layer 2) |
| bitsandbytes on CPU/aarch64 | import error | drop 4-bit, use a smaller model fp32/bf16 |

---

## 12. References

- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Zhou et al. *LIMA: Less Is More for Alignment*. NeurIPS 2023. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- Ouyang et al. *Training language models to follow instructions with human feedback*. NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Biderman et al. *LoRA Learns Less and Forgets Less*. TMLR 2024. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- ASAM OpenSCENARIO: https://www.asam.net/standards/detail/openscenario/
- ISO 34502:2022 *Road vehicles — Test scenarios for automated driving systems*: https://www.iso.org/standard/78951.html
- HF PEFT documentation: https://huggingface.co/docs/peft
