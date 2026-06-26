# Advanced Knowledge Distillation — Practical Techniques for Production

> This guide is the implementation companion to [knowledge_distillation.md](knowledge_distillation.md) and
> [distillation_methods_survey.md](distillation_methods_survey.md). Those documents establish *what* each method
> does. This document asks *why classic KD breaks in practice*, shows the exact failure modes with
> concrete diagnostics, and provides **copy-paste-ready code** for the techniques that fix them.
> After reading this you should be able to drop modern KD into any existing training loop.

---

## Table of Contents

1. [What's Wrong with Classic KD](#1-whats-wrong-with-classic-kd)
2. [DIST — Correlation-Based Logit Distillation (NeurIPS 2022)](#2-dist--correlation-based-logit-distillation-neurips-2022)
3. [Logit Standardization in KD (CVPR 2024)](#3-logit-standardization-in-kd-cvpr-2024)
4. [CTKD — Curriculum Temperature (AAAI 2023)](#4-ctkd--curriculum-temperature-aaai-2023)
5. [SimKD — Reuse the Teacher's Classifier (CVPR 2022)](#5-simkd--reuse-the-teachers-classifier-cvpr-2022)
6. [Hook-Based Feature Extraction — Production Pattern](#6-hook-based-feature-extraction--production-pattern)
7. [Multi-Loss Weight Tuning](#7-multi-loss-weight-tuning)
8. [Debugging Distillation — What to Monitor](#8-debugging-distillation--what-to-monitor)
9. [Production Drop-In Recipe](#9-production-drop-in-recipe)
10. [References](#10-references)

---

## 1. What's Wrong with Classic KD

Classic Hinton KD minimizes:

$$\mathcal{L}_\text{KD} = (1 - \alpha)\,\mathcal{L}_\text{CE}(y, \sigma(z_s)) + \alpha T^2\,\text{KL}\!\left(\sigma\!\left(\tfrac{z_t}{T}\right) \Big\|\, \sigma\!\left(\tfrac{z_s}{T}\right)\right)$$

where $z_t, z_s \in \mathbb{R}^C$ are teacher and student logits, $T$ is the temperature, and $\alpha$ controls the mixing ratio. This formulation has **three distinct failure modes** that each demand a different fix.

### Failure Mode 1 — Temperature Sensitivity (Scale Problem)

KL divergence between softmax distributions is **not invariant to affine shifts** in the logit space. If the teacher produces logits with high variance (e.g., typical for a large ResNet with large weight norms) and the student produces low-variance logits early in training, the soft targets seen by the KL loss are very different in sharpness even at the same $T$.

Concretely, if teacher logit variance $\sigma_t^2 \gg \sigma_s^2$, then $\sigma(z_t / T)$ is still quite peaked while $\sigma(z_s / T)$ is diffuse. The KL loss is dominated by the single highest-probability class, and the inter-class ranking information ("the cat looks 15% like a dog and 8% like a fox") is lost.

This makes the choice of $T$ highly sensitive to *which specific teacher-student pair* you are using, not just the task difficulty.

### Failure Mode 2 — Over-Confidence Suppression (Signal Loss)

When $T$ is raised to soften the teacher's targets, the soft distribution becomes increasingly uniform. In the limit $T \to \infty$, $\sigma(z_t / T) \to \text{Uniform}(C)$ — the soft target carries zero discriminative information.

This means the practitioner faces an unavoidable trade-off:

| Temperature | Target sharpness | Signal quality |
|---|---|---|
| $T = 1$ | Very peaked (near one-hot) | Mimics hard labels — small gradient toward non-top classes |
| $T = 4$ | Moderate | Good balance — but depends on logit scale |
| $T = 20$ | Near-uniform | Gradient vanishes — effectively random soft supervision |

There is no universally safe value of $T$. The CVPR 2024 logit standardization paper showed empirically that across 12 teacher-student pairs and 3 datasets, the optimal $T$ varies from 1.5 to 8.0. Any fixed choice is a compromise.

### Failure Mode 3 — Label Contamination

The $\mathcal{L}_\text{CE}$ term uses the hard one-hot label $y$. On the same sample, the soft KD target says "class 3 has 12% probability" while the hard CE target says "class 3 has 0% probability." These two signals are in direct conflict for every non-target class.

The student receives contradictory gradient signals: CE pushes the student toward a delta distribution on the true class; KD pushes it toward a broad distribution that overlaps with plausible confusers. The mixing weight $\alpha$ blunts but does not resolve this conflict — it just determines which signal wins on average.

DKD (covered in `distillation_methods_survey.md`) partially addresses this by decoupling target-class and non-target-class components. But the underlying tension remains when both losses are summed.

### Summary: Three Failure Modes, Three Fixes

| Failure mode | Root cause | Fix |
|---|---|---|
| Scale sensitivity | Teacher/student logit magnitudes differ | Logit standardization (§3) |
| Signal washed out at high T | Softmax collapses near-uniform | DIST: match correlation, not probabilities (§2) |
| Hard label conflicts with soft target | CE uses one-hot; KD uses soft | Curriculum T that is high early and low late (§4) |

---

## 2. DIST — Correlation-Based Logit Distillation (NeurIPS 2022)

### Core Idea

Instead of matching the *values* of soft-label distributions, DIST matches the **Pearson correlation** between logit vectors. Correlation is invariant to both scale (multiply all logits by a constant) and shift (add a constant to all logits), which eliminates the logit-magnitude problem entirely.

The loss is defined on two levels:

- **Intra-sample** ($\rho_\text{intra}$): correlation between the $C$ logit values for *one sample* — captures which classes the teacher ranks as similar.
- **Inter-sample** ($\rho_\text{inter}$): correlation between the $B$ logit values for *one class* across the batch — captures which samples the teacher considers similarly difficult.

$$\mathcal{L}_\text{DIST} = \beta \cdot (1 - \rho_\text{intra}(z_s, z_t)) + \gamma \cdot (1 - \rho_\text{inter}(z_s, z_t))$$

where $\rho$ is the Pearson correlation coefficient. Both terms are in $[0, 2]$ (since $\rho \in [-1, 1]$) and zero when the student perfectly replicates the teacher's ranking.

### Why Correlation Works Where KL Does Not

Consider a teacher with logits $z_t = [10, 1, 0.1]$ and a student with $z_s = [2, 0.2, 0.02]$ for the same sample. The student has reproduced the teacher's *ranking* perfectly — class 0 wins by a large margin, class 1 is second, class 2 is third. With $T = 4$:

- $\text{KL}(\sigma(z_t / 4) \| \sigma(z_s / 4))$ is large because the distributions differ in sharpness.
- $\rho(z_t, z_s) = 1.0$ — the student gets zero loss.

This is the right behavior: the student has learned everything the teacher knows about class ordering; the logit magnitude difference is an artifact of different network weight scales, not a difference in knowledge.

### When to Use DIST

- Teacher and student have different backbone families (e.g., ViT teacher → CNN student) where logit scales routinely differ by a factor of 3–5×.
- After DKD has plateaued and you want a complementary signal.
- Any setting where you cannot confidently tune $T$.

### Implementation

```python
import torch
import torch.nn.functional as F


def pearson_correlation(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Pearson correlation along a given dimension.

    Args:
        x, y: tensors of same shape
        dim:  dimension along which to compute correlation

    Returns:
        correlation values, shape = x.shape with `dim` removed
    """
    x = x - x.mean(dim=dim, keepdim=True)
    y = y - y.mean(dim=dim, keepdim=True)
    eps = 1e-8
    return (x * y).sum(dim=dim) / (
        x.norm(dim=dim) * y.norm(dim=dim) + eps
    )


def dist_loss(
    student_logits: torch.Tensor,   # [B, C]
    teacher_logits: torch.Tensor,   # [B, C]
    beta: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """DIST loss: Huang et al., NeurIPS 2022 (arXiv:2205.10536).

    beta  controls intra-sample (inter-class) correlation.
    gamma controls inter-sample (intra-class) correlation.
    """
    # Intra-sample: correlate over the C class dimension for each sample
    rho_intra = pearson_correlation(student_logits, teacher_logits, dim=-1)  # [B]

    # Inter-sample: correlate over the B batch dimension for each class
    rho_inter = pearson_correlation(
        student_logits.T, teacher_logits.T, dim=-1
    )  # [C]

    loss_intra = (1.0 - rho_intra).mean()
    loss_inter = (1.0 - rho_inter).mean()

    return beta * loss_intra + gamma * loss_inter
```

**Why this matters**: DIST removes the need to tune $T$ at all for logit matching. It is a near drop-in replacement for the KD term and consistently outperforms vanilla KL-based KD on tasks where teacher/student capacity gaps are large.

---

## 3. Logit Standardization in KD (CVPR 2024)

### The Problem

Even with a well-chosen temperature, the raw KL-based KD loss behaves differently depending on the logit scale. Teacher logit standard deviation typically grows over pretraining (weight norms increase), while a freshly initialized student starts with small logits. This mismatch is worst at the beginning of training — exactly when the student most needs clear guidance.

Measured on ResNet-110 teacher / ResNet-20 student on CIFAR-100: at epoch 1, $\sigma(z_t) \approx 2.8$ while $\sigma(z_s) \approx 0.4$. With $T = 4$, the effective temperature for the teacher is $4 / 2.8 \approx 1.4$ (still quite peaked) while for the student it is $4 / 0.4 = 10$ (nearly uniform). They are not in the same softness regime at all.

### The Fix — Z-Score Before Softmax

Normalize the logits to zero mean and unit variance **before** computing the softmax:

$$\hat{z} = \frac{z - \bar{z}}{\text{std}(z) + \epsilon}$$

where $\bar{z}$ and $\text{std}(z)$ are computed over the class dimension $C$ for each sample independently. Then apply temperature and softmax to $\hat{z}$.

After standardization, both teacher and student logits have the same scale entering the KL computation, so $T$ has the same effect on both. The choice of $T$ now only controls "how soft do you want the targets" — not "how do I compensate for different weight norms."

### Implementation

```python
def standardize_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score logits along the class dimension (per sample).

    This makes KD scale/shift-invariant and removes the need to tune T
    differently for different teacher-student pairs.
    """
    mean = logits.mean(dim=-1, keepdim=True)
    std  = logits.std(dim=-1, keepdim=True)
    return (logits - mean) / (std + eps)


def kd_loss_with_standardization(
    student_logits: torch.Tensor,   # [B, C]
    teacher_logits: torch.Tensor,   # [B, C]
    temperature: float = 4.0,
) -> torch.Tensor:
    """KD loss with logit standardization (Sun et al., CVPR 2024, arXiv:2403.01427)."""
    s_norm = standardize_logits(student_logits)
    t_norm = standardize_logits(teacher_logits)

    soft_student = F.log_softmax(s_norm / temperature, dim=-1)
    soft_teacher = F.softmax(t_norm / temperature, dim=-1)

    return F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
```

You can apply this standardization to **any** logit-based KD variant: vanilla KD, DKD, DIST, or CTKD. It is a single two-line preprocessing step with negligible compute cost.

**Why this matters**: Logit standardization decouples temperature semantics from logit magnitude, making $T$ a genuine hyperparameter about supervision softness rather than a hidden normalization factor. Ablations in the paper show it improves accuracy by 0.3–0.8 pp across diverse teacher-student pairs with no other changes.

---

## 4. CTKD — Curriculum Temperature (AAAI 2023)

### The Static Temperature Problem

A fixed temperature $T$ throughout training is a compromise between two conflicting needs:

- **Early training**: the student is far from the teacher; very peaked targets (low $T$) produce large gradients that can destabilize training. High $T$ gives smoother targets that are easier to chase.
- **Late training**: the student has improved; now the teacher's soft distribution contains fine-grained inter-class information that is only visible at low $T$. High $T$ washes it out.

A fixed $T$ cannot be optimal at both stages. CTKD observes that this is a curriculum problem: start easy (soft targets), get progressively harder (peaked targets), analogous to curriculum learning on data difficulty.

### CTKD Schedule

Anneal $T$ from $T_\text{max}$ to $T_\text{min}$ using a cosine schedule synchronized with the learning rate schedule:

$$T(e) = T_\text{min} + \tfrac{1}{2}(T_\text{max} - T_\text{min})\!\left(1 + \cos\!\left(\frac{\pi \cdot e}{E}\right)\right)$$

where $e$ is the current epoch and $E$ is the total number of epochs. When LR is decayed at epoch $E/2$ (common in step schedules), CTKD can also checkpoint the temperature schedule to match.

Recommended defaults: $T_\text{max} = 8$, $T_\text{min} = 1$ (or $T_\text{min} = 4$ for very large capacity gaps).

### Implementation

```python
import math


class CurriculumTemperature:
    """Cosine-annealing temperature schedule for CTKD.

    Li et al., AAAI 2023 (arXiv:2211.06177).

    Usage:
        ct = CurriculumTemperature(T_max=8.0, T_min=1.0, total_epochs=200)
        for epoch in range(200):
            T = ct.get(epoch)
            loss = kd_loss(student_logits, teacher_logits, temperature=T)
    """

    def __init__(self, T_max: float = 8.0, T_min: float = 1.0, total_epochs: int = 200):
        self.T_max = T_max
        self.T_min = T_min
        self.E = total_epochs

    def get(self, epoch: int) -> float:
        """Return the temperature for the given epoch (0-indexed)."""
        cos_val = math.cos(math.pi * epoch / self.E)
        return self.T_min + 0.5 * (self.T_max - self.T_min) * (1 + cos_val)


# Integration with training loop
def train_one_epoch(model_s, model_t, loader, optimizer, epoch, ct_schedule):
    model_t.eval()
    model_s.train()
    T = ct_schedule.get(epoch)

    for images, labels in loader:
        with torch.no_grad():
            logits_t = model_t(images)
        logits_s = model_s(images)

        loss_ce = F.cross_entropy(logits_s, labels)
        loss_kd = kd_loss_with_standardization(logits_s, logits_t, temperature=T)
        loss = 0.1 * loss_ce + 0.9 * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Why this matters**: CTKD requires zero additional parameters and negligible implementation effort, yet closes 30–50% of the remaining gap between DKD and more expensive feature-based methods on standard benchmarks. Combining CTKD with logit standardization gives the best pure-logit baseline before adding any feature loss.

---

## 5. SimKD — Reuse the Teacher's Classifier (CVPR 2022)

### The Unusual Idea

Every KD recipe so far keeps the student's classifier head and only transfers knowledge about intermediate representations or output distributions. SimKD asks: *why not reuse the teacher's head directly?*

At inference time, SimKD routes the student's penultimate feature through the teacher's frozen linear classifier:

```
Standard KD:  input → student backbone → [student head] → logits
SimKD:        input → student backbone → adapter → [teacher head (frozen)] → logits
```

The adapter is a small projection layer (typically 1×1 conv or linear) that maps the student's feature dimension to the teacher's feature dimension. During training, the student learns to produce features that — after adaptation — produce the same activations the teacher head expects. The teacher head is never updated.

### Why It Works

The student classifier diverging from the teacher's decision boundaries is a major source of performance loss in distillation. Even when feature-level alignment is good, the student's head may carve out different hyperplanes. SimKD eliminates this divergence by definition: at inference, the same head that produced the teacher's excellent calibration is applied to the (adapted) student features.

This works because the teacher's classifier, once trained, is a good linear readout of whatever the teacher has learned to encode. If the student can reproduce those encodings (after adaptation), it inherits the classifier "for free."

### Training and Inference Protocol

**Training**: freeze teacher entirely, train student backbone + adapter.

$$\mathcal{L}_\text{SimKD} = \left\| f_\text{adapter}(h_s) - h_t \right\|_2^2$$

where $h_s, h_t$ are the penultimate features (after global pooling).

**Inference**: remove the adapter and the student's own head. Deploy: student backbone → teacher head. The teacher head is small (one linear layer), so inference cost is dominated by the student backbone — exactly what you want.

### Implementation

```python
class SimKDAdapter(torch.nn.Module):
    """Projection adapter for SimKD feature alignment.

    Chen et al., CVPR 2022 (arXiv:2203.09372).
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(student_dim, teacher_dim, bias=False)
        # Small init to avoid large feature mismatch loss at step 0
        torch.nn.init.xavier_uniform_(self.proj.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def simkd_training_step(
    student_backbone,
    teacher_backbone,
    teacher_head,
    adapter: SimKDAdapter,
    images: torch.Tensor,
):
    """Single forward pass for SimKD training."""
    with torch.no_grad():
        feat_t = teacher_backbone(images)   # [B, D_t]

    feat_s = student_backbone(images)       # [B, D_s]
    feat_s_proj = adapter(feat_s)           # [B, D_t]

    loss = F.mse_loss(feat_s_proj, feat_t)
    return loss


def simkd_inference(student_backbone, teacher_head, images):
    """Deploy without adapter: student backbone → teacher head."""
    feat_s = student_backbone(images)
    return teacher_head(feat_s)   # requires D_s == D_t after stripping adapter
```

### Constraints and When to Use

| Condition | Required |
|---|---|
| Task is fixed | Yes — teacher and student solve the same $C$-class problem |
| Teacher is well-calibrated | Strongly recommended — you inherit its classification boundaries |
| Adapter can be removed post-training | Yes — save only backbone + teacher head for deployment |
| Teacher head dimension matches student feature dim | Must project to match, or use teacher's GAP output dim directly |

SimKD is ideal when you have a well-trained teacher on a fixed benchmark and want maximum accuracy with minimum implementation complexity. It is less suitable for fine-tuning on downstream tasks where the teacher was pretrained on a different label space.

**Why this matters**: SimKD consistently outperforms most feature distillation methods on CIFAR-100 and ImageNet despite using only a single MSE loss — because it solves the classifier divergence problem structurally rather than trying to minimize it with additional loss terms.

---

## 6. Hook-Based Feature Extraction — Production Pattern

### The Problem with Modifying Model Code

Most tutorials implement feature distillation by editing `forward()` to return intermediate activations. This is fragile: it couples your distillation logic to the model source code, breaks when you swap models, and requires maintaining forked copies of every architecture you distill.

The clean production pattern uses PyTorch's hook system to extract features **without touching model code**.

### Hook Types

| Hook type | When it fires | Use case |
|---|---|---|
| `register_forward_hook` | After a module's `forward()` returns | Capture output features (most common) |
| `register_forward_pre_hook` | Before a module's `forward()` is called | Capture or modify input features |
| `register_full_backward_hook` | After backward through a module | Gradient-based analysis; rarely needed for KD |

### FeatureExtractor Context Manager

```python
from typing import Dict, List
import torch
import torch.nn as nn


class FeatureExtractor:
    """Context manager that extracts named intermediate features via hooks.

    Usage (teacher, no grad):
        with FeatureExtractor(teacher, {"layer3": teacher.layer3}) as extractor:
            with torch.no_grad():
                _ = teacher(images)
            feat_t = extractor.features["layer3"]

    Usage (student, with grad):
        with FeatureExtractor(student, {"layer3": student.layer3}) as extractor:
            logits_s = student(images)
            feat_s = extractor.features["layer3"]
            loss = at_loss(feat_s, feat_t) + ce_loss
            loss.backward()
    """

    def __init__(self, model: nn.Module, layers: Dict[str, nn.Module]):
        """
        Args:
            model:  the model whose layers you want to tap (used only for clarity)
            layers: dict of {name: module} — the modules to hook
        """
        self.layers = layers
        self.features: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHook] = []

    def __enter__(self):
        for name, module in self.layers.items():
            # Capture name in closure
            def make_hook(n):
                def hook(module, input, output):
                    self.features[n] = output
                return hook
            handle = module.register_forward_hook(make_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
```

### Pattern: Teacher (No Grad) vs Student (With Grad)

```python
# Define which layers to tap (model-specific, no source edits needed)
teacher_layers = {
    "stage2": teacher.layer2,
    "stage3": teacher.layer3,
}
student_layers = {
    "stage2": student.layer2,
    "stage3": student.layer3,
}

for images, labels in loader:
    # --- Teacher forward (frozen, no grad) ---
    with FeatureExtractor(teacher, teacher_layers) as t_ext:
        with torch.no_grad():
            logits_t = teacher(images)
        feat_t = {k: v.detach() for k, v in t_ext.features.items()}

    # --- Student forward (trainable, with grad) ---
    with FeatureExtractor(student, student_layers) as s_ext:
        logits_s = student(images)
        feat_s = s_ext.features   # tensors still have grad_fn

    # --- Compute losses ---
    loss_ce = F.cross_entropy(logits_s, labels)
    loss_kd = dist_loss(logits_s, logits_t)
    loss_at = sum(
        at_loss(feat_s[k], feat_t[k]) for k in feat_t
    )
    loss = 0.1 * loss_ce + 0.9 * loss_kd + 50.0 * loss_at

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Multi-Layer Extraction for ReviewKD-Style Distillation

For methods that aggregate features across multiple stages (ReviewKD, FitNets at multiple hint layers), simply add more entries to the `layers` dict. The context manager handles arbitrarily many hooks:

```python
teacher_layers = {f"stage{i}": getattr(teacher, f"layer{i}") for i in range(1, 5)}
student_layers = {f"stage{i}": getattr(student, f"layer{i}") for i in range(1, 5)}
```

All hooks are removed at `__exit__` even if an exception occurs, so there is no risk of stale hooks accumulating across iterations.

**Why this matters**: The hook-based pattern lets you apply any feature distillation method to *any* PyTorch model without forking its source. It also makes it trivial to swap which layers are distilled — a critical flexibility when you do not know a priori which layers are most informative for a given teacher-student pair.

---

## 7. Multi-Loss Weight Tuning

### The Combined Loss

Most practical distillation setups combine three terms:

$$\mathcal{L} = w_\text{ce} \cdot \mathcal{L}_\text{CE} + w_\text{kd} \cdot \mathcal{L}_\text{KD} + w_\text{feat} \cdot \mathcal{L}_\text{feat}$$

Tuning three weights manually is expensive. Two principled alternatives exist.

### Approach 1 — Loss Ratio Monitoring (Free)

At epoch 1, log the ratio $\mathcal{L}_\text{KD} / \mathcal{L}_\text{CE}$. If this ratio is much larger than 1, the KD term dominates before the student has learned anything — gradient signal from CE is drowned out, which hurts convergence. Target a ratio of approximately 1–3 at epoch 1.

This is a cheap diagnostic, not a full tuning strategy, but it catches the most common misconfiguration (feature loss scale too large) immediately.

### Approach 2 — Uncertainty Weighting (Kendall et al., 2018)

Learn a scalar log-variance $\log \sigma_i^2$ for each loss term. The total loss becomes:

$$\mathcal{L}_\text{total} = \sum_i \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i$$

The $1 / (2\sigma_i^2)$ term automatically down-weights losses with high variance (tasks the model is uncertain about), and the $\log \sigma_i$ regularizer prevents the model from setting all variances to infinity.

```python
class UncertaintyWeightedLoss(torch.nn.Module):
    """Learnable multi-task loss weighting.

    Kendall et al., 2018 (arXiv:1705.07115).
    Learns log(sigma^2) for each loss term; stable to initialize at 0.
    """

    def __init__(self, n_losses: int = 3):
        super().__init__()
        # log(sigma^2), initialized to 0 → sigma=1 → weight=0.5
        self.log_vars = torch.nn.Parameter(torch.zeros(n_losses))

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        assert len(losses) == self.log_vars.shape[0]
        total = torch.tensor(0.0, device=losses[0].device)
        for loss, log_var in zip(losses, self.log_vars):
            precision = torch.exp(-log_var)          # 1 / sigma^2
            total = total + 0.5 * precision * loss + 0.5 * log_var
        return total


# Usage:
uw_loss = UncertaintyWeightedLoss(n_losses=3).to(device)
# Add uw_loss.parameters() to optimizer param groups
optimizer = torch.optim.SGD(
    list(student.parameters()) + list(uw_loss.parameters()),
    lr=0.1, momentum=0.9, weight_decay=1e-4
)

# In the training step:
loss = uw_loss(loss_ce, loss_kd, loss_feat)
```

### Practical Weight Heuristics

If you do not want learned weights, these ranges are a reliable starting point:

| Setting | $w_\text{ce}$ | $w_\text{kd}$ | $w_\text{feat}$ | Notes |
|---|---|---|---|---|
| Pure logit KD | 0.1 | 0.9 | 0 | Alpha=0.9 is the standard Hinton default |
| Logit + AT | 0.1 | 0.7 | 50–1000 | AT loss is very small per-element; needs large weight |
| Logit + CRD | 0.1 | 0.6 | 0.8 | CRD is contrastive; already normalized to similar scale as KD |
| Dense prediction (FGD) | 0.5 | 0.3 | 1.0 | CE matters more; labels are pixel-dense |
| LLM token-level KD | 0 | 1.0 | 0 | Hard labels often dropped entirely |

**Why this matters**: The most common reason a distillation experiment "doesn't work" is wrong loss weighting rather than wrong method choice. A KD loss 100× larger than CE will prevent the student from learning to classify at all. The ratio diagnostic at epoch 1 catches this in minutes, not after a full training run.

---

## 8. Debugging Distillation — What to Monitor

Distillation training has more moving parts than standard supervised training. Tracking four metrics catches 90% of failure modes early.

### Metric 1 — KD / CE Loss Ratio at Epoch 1

```python
ratio = loss_kd.item() / (loss_ce.item() + 1e-8)
# Target: 0.5 – 3.0
# If ratio >> 3: reduce w_kd or increase w_ce
# If ratio << 0.5: KD is ignored; increase w_kd or check T
```

### Metric 2 — Accuracy Gap Over Time

Plot `teacher_acc - student_acc` per epoch. It should narrow monotonically. If it widens after some epoch $K$:
- The student has overfit to the teacher's soft labels and started ignoring true labels.
- Fix: reduce `w_kd`, increase `w_ce`, or reduce `alpha` in the mixing formula.

### Metric 3 — CKA at the Distilled Layer

Centered Kernel Alignment (CKA) measures representation similarity independent of linear transformation. Track it throughout training; it should increase toward 1.0 at the distilled layer.

```python
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two feature matrices [N, D].

    Returns a scalar in [0, 1]; 1 means identical (up to rotation/scale).
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    numerator   = (XtY * XtY.T).sum()
    denominator = ((XtX * XtX.T).sum() * (YtY * YtY.T).sum()).sqrt()
    return (numerator / (denominator + 1e-8)).item()

# Log per epoch:
cka = linear_cka(feat_s.flatten(1), feat_t.flatten(1))
# If CKA is flat or declining: feature loss weight is too small,
# or you are tapping the wrong layer.
```

### Metric 4 — Expected Calibration Error (ECE)

Feature distillation can degrade calibration when `w_feat` is too high: the student learns to produce teacher-like features but its logit magnitudes are distorted. Track ECE alongside accuracy; if ECE worsens while accuracy improves, reduce `w_feat`.

### Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Student accuracy worse than scratch baseline | KD overwhelms CE; student stops using labels | Reduce `w_kd` to ≤ 0.5; check loss ratio |
| Feature KD gives no improvement over logit KD | Feature loss scale is too small | Check `w_feat`; AT needs ~100×, CRD needs ~1× |
| Good early performance, then late collapse | Teacher left in `train()` mode — dropout/BN shifts | Always call `teacher.eval()` before training loop |
| Loss is `nan` from first batch | Adapter weight init too large; logit norm explosion | Use `gain=0.1` in Xavier init; check for log(0) in KL |
| Student matches teacher on train set, poor generalization | Temperature too high throughout; student memorizes soft labels | Use CTKD to anneal T down over training |
| ECE worsens despite accuracy gain | Feature loss is distorting logit scale | Reduce `w_feat`; add logit standardization |

### DistillationMonitor

```python
class DistillationMonitor:
    """Logs KD health metrics to a dict (compatible with wandb.log / tb SummaryWriter)."""

    def __init__(self):
        self.history = []

    @torch.no_grad()
    def log(
        self,
        epoch: int,
        loss_ce: float,
        loss_kd: float,
        feat_s: torch.Tensor,   # [B, D] flattened student features
        feat_t: torch.Tensor,   # [B, D] flattened teacher features
        logits_s: torch.Tensor, # [B, C]
        labels: torch.Tensor,   # [B]
    ) -> dict:
        ratio = loss_kd / (loss_ce + 1e-8)
        cka   = linear_cka(feat_s, feat_t)
        acc   = (logits_s.argmax(-1) == labels).float().mean().item()

        record = dict(epoch=epoch, kd_ce_ratio=ratio, cka=cka, student_acc=acc)
        self.history.append(record)

        if ratio > 5.0:
            print(f"[KD WARNING] Epoch {epoch}: KD/CE ratio={ratio:.2f} >> 1 — consider reducing w_kd")
        if len(self.history) > 1 and cka < self.history[-2]["cka"] - 0.02:
            print(f"[KD WARNING] Epoch {epoch}: CKA dropped {self.history[-2]['cka']:.3f}→{cka:.3f}")

        return record
```

**Why this matters**: Most failed distillation runs are caused by silent misconfiguration — the loss runs without error, but one term is 100× larger than intended, or the teacher is accidentally left in training mode. These four metrics surface those issues within the first epoch, saving multiple full training runs.

---

## 9. Production Drop-In Recipe

This section provides a single, self-contained training loop template that incorporates all of the above techniques. Copy it, replace the model constructors with your own, and it should work.

### Full Training Loop Template

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ── Assumes you have defined somewhere: ──────────────────────────────────────
#   teacher:   nn.Module  (pretrained, kept frozen)
#   student:   nn.Module  (to be trained)
#   train_loader: DataLoader
#   val_loader:   DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def train_with_kd(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    lr: float = 0.1,
    # Logit distillation
    use_standardize: bool = True,
    use_ctkd: bool = True,
    T_max: float = 8.0,
    T_min: float = 1.0,
    # Feature distillation (optional)
    use_feature_kd: bool = False,
    teacher_feat_layer: nn.Module = None,   # e.g. teacher.layer3
    student_feat_layer: nn.Module = None,   # e.g. student.layer3
    w_feat: float = 50.0,
    # Loss weights
    w_ce: float = 0.1,
    w_kd: float = 0.9,
    device: str = "cuda",
    log_to_wandb: bool = False,
):
    # ── Freeze teacher ────────────────────────────────────────────────────────
    teacher.to(device).eval()
    teacher.requires_grad_(False)

    # ── Student optimizer ─────────────────────────────────────────────────────
    student.to(device)
    optimizer = torch.optim.SGD(
        student.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Curriculum temperature ────────────────────────────────────────────────
    ct = CurriculumTemperature(T_max=T_max, T_min=T_min, total_epochs=epochs)

    # ── Feature layers ────────────────────────────────────────────────────────
    if use_feature_kd:
        assert teacher_feat_layer is not None and student_feat_layer is not None
        # Adapter: maps student feat dim → teacher feat dim at first batch
        # (defer construction until we know dims)
        adapter = None

    monitor = DistillationMonitor()

    for epoch in range(epochs):
        student.train()
        T = ct.get(epoch) if use_ctkd else 4.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # ── Teacher forward ───────────────────────────────────────────────
            if use_feature_kd:
                t_layers = {"feat": teacher_feat_layer}
                s_layers = {"feat": student_feat_layer}

                with FeatureExtractor(teacher, t_layers) as t_ext:
                    with torch.no_grad():
                        logits_t = teacher(images)
                    feat_t = t_ext.features["feat"].detach()

                with FeatureExtractor(student, s_layers) as s_ext:
                    logits_s = student(images)
                    feat_s = s_ext.features["feat"]

                # Build adapter on first batch once we know dims
                if adapter is None:
                    d_s = feat_s.flatten(1).shape[-1]
                    d_t = feat_t.flatten(1).shape[-1]
                    adapter = nn.Linear(d_s, d_t, bias=False).to(device)
                    nn.init.xavier_uniform_(adapter.weight, gain=0.1)
                    optimizer.add_param_group({"params": adapter.parameters()})
                    print(f"Adapter: {d_s} → {d_t}")

                loss_feat = F.mse_loss(
                    adapter(feat_s.flatten(1)), feat_t.flatten(1)
                )
            else:
                with torch.no_grad():
                    logits_t = teacher(images)
                logits_s = student(images)
                loss_feat = torch.tensor(0.0, device=device)

            # ── Logit distillation ────────────────────────────────────────────
            loss_ce = F.cross_entropy(logits_s, labels)

            if use_standardize:
                loss_kd = kd_loss_with_standardization(logits_s, logits_t, temperature=T)
            else:
                s_soft = F.log_softmax(logits_s / T, dim=-1)
                t_soft = F.softmax(logits_t / T, dim=-1)
                loss_kd = F.kl_div(s_soft, t_soft, reduction="batchmean") * (T ** 2)

            loss = w_ce * loss_ce + w_kd * loss_kd + w_feat * loss_feat

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ── Log metrics at first batch of epoch ──────────────────────────
            if batch_idx == 0:
                with torch.no_grad():
                    feat_log_s = feat_s.flatten(1).detach() if use_feature_kd else logits_s.detach()
                    feat_log_t = feat_t.flatten(1).detach() if use_feature_kd else logits_t.detach()
                record = monitor.log(
                    epoch, loss_ce.item(), loss_kd.item(),
                    feat_log_s, feat_log_t, logits_s.detach(), labels
                )
                if log_to_wandb:
                    import wandb
                    wandb.log({"epoch": epoch, "T": T, **record})

        scheduler.step()

    # ── Save student only (no adapter) ───────────────────────────────────────
    # Strip adapter from state — deploy only student backbone + its own head
    checkpoint = {
        "student_state_dict": student.state_dict(),
        "epoch": epochs,
    }
    return student, checkpoint
```

### Checklist — Before Calling It Done

Before declaring a distillation run complete, verify all of the following:

- [ ] `teacher.eval()` was called before training and never changed during the run
- [ ] `teacher.requires_grad_(False)` was set — confirm via `sum(p.requires_grad for p in teacher.parameters()) == 0`
- [ ] Any adapter layer is **excluded** from the saved student checkpoint
- [ ] KD/CE loss ratio was logged at epoch 1 and was in the range 0.5–5.0
- [ ] Student accuracy was compared against a scratch baseline (same student, no teacher) — KD should improve by at least 0.5 pp; if not, the teacher may have too small a capacity gap to add signal
- [ ] Final CKA between distilled layers is higher at the end of training than at epoch 1
- [ ] ECE was checked if calibration matters for the downstream use case

---

## 10. References

| Method | Paper | arXiv |
|---|---|---|
| DIST | Huang et al., "Knowledge Distillation from A Stronger Teacher," NeurIPS 2022 | [arXiv:2205.10536](https://arxiv.org/abs/2205.10536) |
| Logit Standardization | Sun et al., "Logit Standardization in Knowledge Distillation," CVPR 2024 | [arXiv:2403.01427](https://arxiv.org/abs/2403.01427) |
| CTKD | Li et al., "Curriculum Temperature for Knowledge Distillation," AAAI 2023 | [arXiv:2211.06177](https://arxiv.org/abs/2211.06177) |
| SimKD | Chen et al., "Revisiting Knowledge Distillation for Autoregressive Language Models," CVPR 2022 | [arXiv:2203.09372](https://arxiv.org/abs/2203.09372) |
| Uncertainty weighting | Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses," CVPR 2018 | [arXiv:1705.07115](https://arxiv.org/abs/1705.07115) |
| CKA | Kornblith et al., "Similarity of Neural Network Representations Revisited," ICML 2019 | [arXiv:1905.00414](https://arxiv.org/abs/1905.00414) |
| DKD | Zhao et al., "Decoupled Knowledge Distillation," CVPR 2022 | [arXiv:2203.08679](https://arxiv.org/abs/2203.08679) |

See also:
- [knowledge_distillation.md](knowledge_distillation.md) — FitNets, AT, FSP, NST, PKT, RKD, CRD, OFD, ReviewKD
- [distillation_methods_survey.md](distillation_methods_survey.md) — DKD, TAKD, online/self distillation, data-free, FGD, TinyBERT, MiniLLM
- [feature_distillation_why.md](feature_distillation_why.md) — CKA analysis of why features beat logits
