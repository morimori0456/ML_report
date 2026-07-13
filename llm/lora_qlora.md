---
title: "LoRA / QLoRA Complete Guide"
description: "A complete guide to LoRA and QLoRA parameter-efficient fine-tuning, from the low-rank update math to 4-bit NF4 quantization internals in PEFT and bitsandbytes."
---

> Code references:
> - `peft` (HuggingFace PEFT): `src/peft/tuners/lora/layer.py`, `src/peft/tuners/lora/model.py`, `src/peft/tuners/lora/config.py`
> - `bitsandbytes`: `bitsandbytes/nn/modules.py` (`Linear4bit`, `Params4bit`), `bitsandbytes/functional.py` (`quantize_4bit`, `dequantize_4bit`)
> - `transformers` (HuggingFace): `src/transformers/integrations/bitsandbytes.py`, `src/transformers/quantizers/quantizer_bnb_4bit.py`
> - Original papers: LoRA (Hu et al., 2021, arXiv:2106.09685) / QLoRA (Dettmers et al., 2023, arXiv:2305.14314)

---

## 0. How to read this document

LoRA and QLoRA are techniques for "fine-tuning large pre-trained models with less memory and computation."
- **LoRA** = a method that approximates weight updates with **low-rank matrices** (the core of the memory reduction)
- **QLoRA** = a method that freezes the base model quantized to **4-bit** and applies LoRA on top (further memory reduction)

The recommended reading order is **Full FT memory breakdown → LoRA → quantization → QLoRA**.
Each section ends with a "hands-on" point, so use `lora_qlora_demo.ipynb` (runs immediately with numpy only) and `lora_qlora_finetune.ipynb` (real training on GPU) together.

---

## 1. Why PEFT is needed — the memory problem with Full Fine-Tuning

GPU memory breakdown for full fine-tuning a 7B model (7 billion parameters) with **bfloat16 + Adam**:

| Item | Factor | Amount for 7B | Description |
|---|---|---|---|
| Model weights | 2 bytes/param | 14 GB | stored in bf16 |
| Gradients | 2 bytes/param | 14 GB | one per parameter |
| Adam: 1st moment `m` | 4 bytes/param | 28 GB | stored in fp32 |
| Adam: 2nd moment `v` | 4 bytes/param | 28 GB | stored in fp32 |
| (fp32 master weights for mixed precision) | 4 bytes/param | 28 GB | implementation-dependent |
| **Total (excluding activations)** | | **~84–112 GB** | |

Key insight: **the model weights are only 14 GB, but the optimizer state (Adam's m, v) is overwhelmingly large**.
Even a single A100 80GB cannot comfortably handle full fine-tuning of a 7B model; add activation memory and multiple GPUs become necessary.

> **The right perspective**: the bottleneck in fine-tuning is not "the size of the weights" but "the optimizer state and gradients attached to trainable parameters." Reducing these is the idea behind PEFT.

**PEFT (Parameter-Efficient Fine-Tuning)** = freeze the base weights and train only a tiny set of additional parameters.
If only 0.1–1% of parameters are trainable, gradients and optimizer state are needed only for those.

→ Hands-on: the "memory breakdown bar chart" cell in `demo`

---

## 2. The principle of LoRA — approximating weight updates with low-rank decomposition

### 2.1 Core idea

Fine-tuning means updating pre-trained weight `W₀ ∈ ℝ^{d×k}` to `W₀ + ΔW`.
LoRA's hypothesis (original paper): **ΔW has a low "intrinsic rank"** → it can be sufficiently approximated by a low-rank matrix.

```
ΔW = B · A

  B ∈ ℝ^{d×r}   (up projection side)
  A ∈ ℝ^{r×k}   (down projection side)
  r ≪ min(d, k)  (rank; typically r = 8, 16, 32, 64)
```

Original forward pass:
```
h = W₀ · x
```
Forward pass after applying LoRA:
```
h = W₀ · x + (α/r) · B · (A · x)
        └ frozen ┘   └─ trainable (only B, A) ─┘
```

- `W₀` is **frozen** (no gradient flows through it)
- Only `A` and `B` are trained
- `α` (lora_alpha) is a **scaling coefficient** that adjusts the effective learning rate

### 2.2 Parameter count comparison

For a single Linear layer with `d = k = 4096` (typical hidden dimension of a 7B model):

| Method | Parameter count | Value at 4096×4096 |
|---|---|---|
| Full | d × k | 16,777,216 (~16.8 M) |
| LoRA (r=8) | r × (d + k) | 65,536 (~65 K) |
| LoRA (r=16) | r × (d + k) | 131,072 (~131 K) |

With r=8, trainable parameters drop to just **0.39%**.

### 2.3 The subtlety of initialization: why ΔW = 0 at the start of training

```
A ~ N(0, σ²)  (Gaussian initialization, or Kaiming)
B = 0          (zero initialization)
→ ΔW = B·A = 0  (at the start of training)
```

Starting with `B = 0` means **the output at the very beginning of training is identical to the original model**.
This prevents "performance collapse at the start of fine-tuning" and allows stable departure from `W₀`.

> peft code: `reset_lora_parameters()` in `src/peft/tuners/lora/layer.py`.
> `lora_A` is initialized with `kaiming_uniform_`, `lora_B` with `zeros_`.

### 2.4 The meaning of α / r scaling

```
Effective scale = α / r
```
In practice, many users "fix `α` equal to `r` and preserve the scale when changing `r`" (e.g., r=8, α=16 → scale 2).
- Increasing α = amplifying the contribution of ΔW (similar to raising the learning rate)
- **rsLoRA** (rank-stabilized LoRA) uses `α/√r` to maintain stability at higher ranks

→ Hands-on: "low-rank approximation for matrix reconstruction" and "parameter count" cells in `demo`

---

## 3. Which layers to apply LoRA to

Specified via `target_modules`. The target is Linear layers in Transformer Attention/MLP.

| Target | Module name (LLaMA family) | Effect |
|---|---|---|
| Query/Value projections | `q_proj`, `v_proj` | Minimal configuration from the original paper. Good cost-efficiency |
| Full Attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Higher expressiveness |
| Including MLP | + `gate_proj`, `up_proj`, `down_proj` | Recommended by the QLoRA paper. Applying to all Linear layers is the standard |

Key finding from the QLoRA paper: **"applying LoRA to all Linear layers" yields the best performance** (better than Attention only).
When in doubt, use all Linear layers (`all-linear`). In peft, `target_modules="all-linear"` sets this automatically.

```python
from peft import LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",   # all Linear layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

---

## 4. Where LoRA's memory reduction comes from

When trainable parameters drop to 0.4%, everything tied to them can be eliminated:

| Item | Full FT (7B) | LoRA (7B, ~0.4%) |
|---|---|---|
| Model weights (frozen) | 14 GB | 14 GB (unchanged) |
| Gradients | 14 GB | **~0.06 GB** |
| Adam m, v | 56 GB | **~0.24 GB** |
| Total (excluding activations) | ~84 GB | **~14.3 GB** |

Note: **the weights themselves are not reduced**. What LoRA eliminates is the gradient and optimizer state.
→ This motivates the next step of "wanting to further reduce the 14 GB of weights," which leads to QLoRA (4-bit quantization).

→ Hands-on: "Full vs LoRA vs QLoRA memory comparison" cell in `demo`

---

## 5. Quantization basics — prerequisite knowledge for QLoRA

The "Q" in QLoRA stands for 4-bit quantization. First, understand quantization in general.

### 5.1 What is quantization

Compressing fp16 (2 bytes = 16 bits) weights to 4 bits → 1/4 the memory.
The basic idea is "rounding real numbers to discrete grid points":

```
Quantization:    q = round(x / scale)        scale = absmax / (2^(bits-1) - 1)
Dequantization:  x̂ = q × scale
```

Holding a `scale` (quantization constant) per block is called **block-wise quantization** (localizes the impact of outliers).

### 5.2 NF4 (NormalFloat4) — the first core of QLoRA

Ordinary int4 uses equally spaced grid points. However, **neural network weights are approximately normally distributed with mean 0**.
→ Instead of equal spacing, using **non-uniform grid points aligned with quantiles of the normal distribution** is information-theoretically optimal. This is NF4.

```
The 16 grid points of NF4 are derived from the quantiles of the standard normal distribution N(0,1) (a symmetric set of 16 values including 0).
Normalize the weight block to [-1, 1] using absmax → store the index (4 bits) of the nearest NF4 grid point.
```

- The paper claims "information-theoretically optimal" because, when the input is normally distributed, data falls uniformly into each bin.
- Higher accuracy than int4 for the same 4 bits.

> bitsandbytes code: `create_normal_map()` in `bitsandbytes/functional.py` generates the NF4 grid.
> `quantize_4bit(x, quant_type="nf4")` / `dequantize_4bit(...)`.

### 5.3 Double Quantization (DQ) — the second core

With block-wise quantization, each block needs a `scale` (fp32, 4 bytes). For block size 64:
```
Scale overhead = 32 bit / 64 params = 0.5 bit/param
```
This is non-negligible. **DQ "further quantizes the quantization constants themselves"** to compress them to 8 bits:
```
0.5 bit/param → approximately 0.127 bit/param (paper value)
→ savings of approximately 0.37 bits per parameter
```

### 5.4 Paged Optimizers — the third core

To prevent OOM from memory spikes during training (e.g., from long sequences),
NVIDIA Unified Memory is used to **page** the optimizer state between CPU and GPU as needed (like OS virtual memory).

→ Hands-on: "NF4 quantization simulation" and "int4 vs NF4 error comparison" cells in `demo`

---

## 6. QLoRA — putting it all together

### 6.1 Structure

```
                  ┌─────────────── 4bit (NF4, frozen) ────────────────┐
input x ─────────► dequantize W₀(4bit) to fp16/bf16 block-by-block, then matmul
   │              └────────────────────────────────────────────────────┘
   │                                                          │
   └──► LoRA: A(16bit) ─► B(16bit) ──(α/r)──────────────────► (+)──► h
              └─ trainable (fp16/bf16) ─┘
```

Key points:
1. **Base weight W₀ is stored in NF4 (frozen)** → weight storage shrinks from 14 GB to ~3.5 GB
2. During forward/backward computation, only the required blocks are **dequantized to fp16** for matmul (computation precision is 16-bit)
3. **Gradients flow only to LoRA's A and B** (W₀ is frozen so no gradients needed)
4. LoRA parameters are in 16-bit; optimizer is paged

### 6.2 Memory comparison (7B model)

| Method | Weights | Gradients | Adam state | Total (excl. activations) | Required GPU |
|---|---|---|---|---|---|
| Full FT (bf16) | 14 GB | 14 GB | 56 GB | **~84 GB** | A100 80GB × multiple |
| LoRA (bf16) | 14 GB | 0.06 GB | 0.24 GB | **~14.3 GB** | RTX 4090 24GB |
| **QLoRA (NF4)** | 3.5 GB | 0.06 GB | 0.24 GB | **~3.8 GB** + activations | **RTX 3060 12GB / Colab T4** |

Representative achievement from the QLoRA paper: **fine-tuning a 65B model on a single 48 GB GPU** (Guanaco).
This was a revolutionary improvement — something that required hundreds of GB running on a single card.

### 6.3 Does accuracy suffer?

The QLoRA paper's claim: **achieves performance equivalent to 16-bit full fine-tuning using a 4-bit base + LoRA**.
Reasons:
- NF4 is optimized for the weight distribution
- The computation (matmul) itself is done in 16-bit (only storage is 4-bit)
- LoRA absorbs the quantization error through learning

---

## 7. PEFT / bitsandbytes code analysis

### 7.1 4-bit loading (transformers + bitsandbytes)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # use NF4 (§5.2)
    bnb_4bit_use_double_quant=True,     # Double Quantization (§5.3)
    bnb_4bit_compute_dtype=torch.bfloat16,  # dtype for computation (§6.1-2)
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", quantization_config=bnb_config, device_map="auto",
)
```

Internal flow (`transformers/integrations/bitsandbytes.py`):
- `nn.Linear` is replaced with `bnb.nn.Linear4bit`
- Weights are NF4-quantized as `Params4bit` and moved to GPU
- During forward, `Linear4bit.forward()` calls `dequantize_4bit` → matmul

### 7.2 Applying LoRA (peft)

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)  # prepare quantized model for training
#  └ cast layernorm to fp32, add requires_grad to input, enable gradient checkpointing, etc.

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules="all-linear",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~10M || all params: ~1.2B || trainable%: ~0.8%
```

Schematic excerpt of `Linear.forward()` from `peft/tuners/lora/layer.py`:
```python
result = self.base_layer(x)                 # frozen (4-bit) base
for active_adapter in self.active_adapters:
    lora_A = self.lora_A[active_adapter]
    lora_B = self.lora_B[active_adapter]
    scaling = self.scaling[active_adapter]  # = alpha / r
    x_ = dropout(x)
    result = result + lora_B(lora_A(x_)) * scaling
return result
```

---

## 8. Practical hyperparameter guide

| Parameter | Recommended starting value | Effect and tips |
|---|---|---|
| `r` (rank) | 16 (8–64) | Higher = more expressiveness but higher memory and overfitting risk. Start with 16 |
| `lora_alpha` | 2× r (=32) | Scale α/r. Common practice to keep it linked to r |
| `lora_dropout` | 0.05–0.1 | Prevents overfitting. Use higher values for smaller datasets |
| `target_modules` | `all-linear` | QLoRA paper recommendation. All Linear layers work best |
| `learning_rate` | 1e-4–2e-4 | **One order of magnitude higher** than Full FT is fine (fewer parameters) |
| `bias` | `"none"` | Bias is usually not trained |
| optimizer | `paged_adamw_8bit` | Standard for QLoRA. 8-bit Adam + paging |

Rules of thumb:
- **Link r and alpha** (alpha = 2r) to avoid needing to re-tune lr when changing r
- For small datasets (hundreds to thousands of samples), use smaller r and larger dropout
- When things don't work, "widening target_modules" often helps more than "increasing r"

---

## 9. Inference after training — merging and adapter switching

### 9.1 merge (for production deployment)

```python
merged = model.merge_and_unload()   # compute W₀ + (α/r)BA and bake into a single set of weights
merged.save_pretrained("./merged-model")
```
- `merge_and_unload()` materializes `W = W₀ + scaling·B·A` → the LoRA branch disappears, resulting in **zero additional inference latency**
- Caution: **you cannot merge directly into a 4-bit-quantized base** (the safe approach is to reload in fp16 and then merge)

### 9.2 Adapter swapping (multi-task)

```python
model.load_adapter("./adapter-task-A", adapter_name="A")
model.load_adapter("./adapter-task-B", adapter_name="B")
model.set_adapter("A")   # switch at runtime
```
One large base + multiple small adapters enables cheap multi-task operation.
This is one of LoRA's strong practical advantages (serving systems like S-LoRA scale this up).

---

## 10. LoRA variants and extensions (2024–)

| Method | Summary | When to use |
|---|---|---|
| **QLoRA** | 4-bit NF4 + LoRA | Memory-first priority. The star of this document |
| **DoRA** | Decompose weights into "magnitude" and "direction," then apply LoRA | Higher quality than LoRA. Enable with `use_dora=True` in peft |
| **rsLoRA** | Uses α/√r as the scale | Stable at high rank (r≥64) |
| **LoRA+** | Different learning rates for A and B (higher for B) | For faster convergence |
| **VeRA** | Shared random matrices + small scaling vectors | Ultimate parameter efficiency |
| **PiSSA** | Initialize A and B from principal singular components | Better convergence and performance |

The recommended order is: master QLoRA completely, then try DoRA next.

---

## 11. Pitfalls and practical tips

1. **Do not forget `prepare_model_for_kbit_training`** — required before applying LoRA to a quantized model. Forgetting it causes no gradient flow and no training progress.
2. **`compute_dtype` should be bf16** (Ampere and later). Use fp16 for older GPUs like T4.
3. **Use with gradient checkpointing** to further reduce activation memory (trade-off with speed).
4. **Learning rate should be higher than Full FT** (1e-4–2e-4). Too low and nothing happens.
5. **Incorrect `target_modules` specification** often results in "trainable% is extremely small or 0" → always verify with `print_trainable_parameters()`.
6. **Do not merge directly into a 4-bit base** — merge using an fp16 base. Merging while quantized degrades accuracy.
7. **Remember to disable dropout during evaluation** (`model.eval()`).
8. **Only the adapter is saved** (tens of MB). The base is managed separately. Reproduction requires both the base model name and the adapter.

---

## 12. Summary — at a glance

```
Full FT:   weights 14 + gradients 14 + Adam 56 = 84 GB   (optimizer state is the main culprit)
   │
   ├─ LoRA:   training reduced to 0.4% → gradients and Adam nearly vanish → 14.3 GB
   │            (but 14 GB of weights remain)
   │
   └─ QLoRA:  weights quantized to NF4 4-bit → 3.5 GB + LoRA → 3.8 GB
                · NF4: grid at quantiles of the normal distribution (information-theoretically optimal)
                · Double Quant: quantize the quantization constants as well
                · Paged Optimizer: prevents OOM
                · computation in 16-bit / storage in 4-bit / gradients only for LoRA
```

**Learning roadmap**:
1. Experience "low-rank decomposition," "memory calculation," and "NF4 quantization" with numpy in `lora_qlora_demo.ipynb` (runs right now in this environment)
2. Actually fine-tune a small LLM with QLoRA on Colab/GPU in `lora_qlora_finetune.ipynb`
3. Change r/alpha/target_modules and observe the differences in behavior
4. Extend to DoRA and rsLoRA

---

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314
- Dettmers et al., "8-bit Optimizers via Block-wise Quantization", arXiv:2110.02861
- Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", arXiv:2402.09353
- HuggingFace PEFT docs: https://huggingface.co/docs/peft
