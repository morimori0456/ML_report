---
title: "Weight EMA — Exponential Moving Average of Model Weights"
description: "How keeping an exponential moving average shadow copy of model weights delivers cheap, reliable accuracy gains at evaluation time."
---

> Keep a slowly-moving **shadow copy** of the model weights, updated as a running exponential
> average of the training weights. At evaluation you use the shadow copy, not the raw weights.
> It is one of the cheapest, most reliable "free accuracy" tricks in deep learning.

For a runnable demonstration (variance reduction on a toy problem + a real MLP that EMA improves),
see [weight_ema_demo.ipynb](https://github.com/morimori0456/ML_report/blob/main/ema/weight_ema_demo.ipynb).

---

## Table of Contents
1. [The Update Rule](#1-the-update-rule)
2. [Why It Helps (Three Views)](#2-why-it-helps-three-views)
3. [Decay ↔ Averaging Window](#3-decay--averaging-window)
4. [Bias and Warmup](#4-bias-and-warmup)
5. [EMA vs SWA vs Polyak–Ruppert](#5-ema-vs-swa-vs-polyakruppert)
6. [Where EMA Is Essential](#6-where-ema-is-essential)
7. [Implementation Details That Matter](#7-implementation-details-that-matter)
8. [Common Pitfalls](#8-common-pitfalls)
9. [References](#9-references)

---

## 1. The Update Rule

Let `θ_t` be the model weights after optimizer step `t`, and `v_t` the EMA ("shadow") weights.
After every step:

$$ v_t = \beta \, v_{t-1} + (1 - \beta)\, \theta_t $$

- `β` is the **decay** (typ. 0.99, 0.999, 0.9999). Higher β = slower, smoother shadow.
- `v` is **not** part of the forward/backward pass — it never produces gradients. It is updated
  *after* the optimizer step from the freshly-updated `θ`.
- At eval/deploy you **swap in `v`** (and usually keep the raw `θ` for continued training).

Equivalent "how far does it lag" form with `α = 1 - β`:

$$ v_t = v_{t-1} + (1-\beta)\,(\theta_t - v_{t-1}) $$

This is exactly the **Polyak (soft) update** used for RL target networks (there written
`v ← τθ + (1-τ)v`, `τ = 1-β`).

Expanding the recursion shows what `v_t` actually is — a geometrically-weighted average of the
**entire weight trajectory**, most weight on recent steps:

$$ v_t = (1-\beta)\sum_{i=0}^{t} \beta^{\,t-i}\, \theta_i $$

---

## 2. Why It Helps (Three Views)

**(a) Variance reduction.** SGD with a finite learning rate doesn't converge to a point — it
*bounces around* the minimum with a stationary variance set by the LR and gradient noise. Averaging
those iterates cancels the zero-mean noise. For iterates with variance `σ²`, the EMA has variance

$$ \mathrm{Var}(v) = \sigma^2 \cdot \frac{1-\beta}{1+\beta} $$

so `β = 0.99` shrinks the standard deviation by ~14×. You get the "center" of the cloud instead of a
random sample from it.

**(b) Flatter / wider minima.** Averaging in **weight space** lands you in the *interior* of a
low-loss basin rather than on a jagged wall of it. Wide, flat minima generalize better — this is the
intuition shared with SWA. The averaged point often has **lower test loss than any single iterate**,
even when its train loss is slightly higher.

**(c) An implicit ensemble.** `v` blends many nearby models from the trajectory. Like an ensemble it
reduces variance, but at the cost of **one** extra weight copy and **zero** extra forward passes at
inference (unlike a real ensemble).

---

## 3. Decay ↔ Averaging Window

The geometric weights have a "center of mass" — the effective number of recent steps being averaged:

$$ N \approx \frac{1}{1-\beta} $$

| β (decay) | Effective window N | Feel |
|---|---|---|
| 0.9 | ~10 steps | very responsive, light smoothing |
| 0.99 | ~100 steps | typical for short / small runs |
| 0.999 | ~1,000 steps | typical classification default |
| 0.9999 | ~10,000 steps | diffusion / very long runs |

Key consequence: **β must be matched to run length.** `β = 0.9999` on a 500-step run averages
almost nothing but the initialization — the shadow never catches up. Pick β so `N` is a meaningful
fraction of total steps (or use a warmup, §4). Some libraries instead specify the window/`num_updates`
and derive β.

---

## 4. Bias and Warmup

Early on, `v` is dragged toward its initialization (usually `v_0 = θ_0`), so for small `t` the
average is **biased toward the initial weights**. Two standard fixes:

**Bias correction** (same as Adam's moments):

$$ \hat{v}_t = \frac{v_t}{1 - \beta^{\,t}} $$

**Warmup decay** (used by timm / the original TF EMA) — ramp β up from ~0:

$$ \beta_t = \min\!\left(\beta,\; \frac{1 + t}{10 + t}\right) $$

At `t=0` this is `1/10` (tracks the model almost directly); it approaches the target β as training
proceeds. This avoids the init bias without an explicit correction term and is the common choice for
weight EMA.

---

## 5. EMA vs SWA vs Polyak–Ruppert

All three average weights; they differ in *which* weights and *how*.

| Method | Weighting | LR schedule | Notes |
|---|---|---|---|
| **EMA** | exponential (recent-heavy) | any | online, one extra copy; the default |
| **SWA** (Stochastic Weight Averaging) | **uniform** over a window | high **constant/cyclic** LR in the averaging phase | snapshots every epoch; recompute BN stats at the end |
| **Polyak–Ruppert** | uniform over **all** iterates | decaying LR | classical convex-optimization result; optimal asymptotic rate |

Practical differences: **SWA** deliberately uses a *high* LR so snapshots are spread across the basin,
then averages uniformly — and you must **recompute BatchNorm statistics** afterward (the averaged
weights never "saw" data). **EMA** is fully online, weights recent steps more, and works with your
normal LR schedule. In practice EMA is the easier drop-in; SWA can edge it out near the end of
training with the right cyclic LR.

---

## 6. Where EMA Is Essential

EMA is "nice to have" for classification but **load-bearing** in several areas:

- **Diffusion / score models** — EMA of the U-Net/DiT weights is *critical*; samples from the raw
  weights are visibly worse. β ≈ 0.9999 (+) is standard, and the EMA weights are what gets released.
- **Self-supervised learning** — the **momentum encoder** in MoCo and the **target network** in BYOL
  / DINO are EMAs of the online network (`β` ramped 0.99 → 1.0). The slow target prevents collapse.
- **Semi-supervised** — **Mean Teacher**: the teacher is an EMA of the student; its predictions on
  unlabeled data are the consistency target.
- **RL** — **target networks** (DQN/DDPG/SAC) are Polyak EMAs of the online network; the slow target
  stabilizes the bootstrapped TD target.
- **GANs** — EMA of the generator weights markedly improves and stabilizes sample quality (StyleGAN).
- **Supervised SOTA recipes** — timm/torchvision "A1/A2" recipes ship `ModelEmaV2`; it's a routine
  fraction-of-a-percent boost and a smoother validation curve.

---

## 7. Implementation Details That Matter

A correct `ModelEMA` is short, but several details bite:

1. **Update *after* the optimizer step**, once per step, on the just-updated weights.
2. **Average parameters; copy or average buffers carefully.** BatchNorm `running_mean/var` are
   **buffers**, not parameters. Most implementations *EMA the parameters and just copy the latest
   buffers* (or EMA them too). Forgetting buffers entirely → the EMA model uses init-time BN stats
   and evaluates terribly. (See the demo.)
3. **No grad.** Do the update under `torch.no_grad()`; `v` requires no gradient.
4. **Keep `v` on the same device**; for memory you can hold it on CPU and sync at eval (slower).
5. **Checkpoint the EMA separately** (`state_dict` of the shadow) and **resume it** — losing it on a
   restart resets the averaging window. Save both raw and EMA.
6. **Distributed**: update the EMA on each rank from its *already-synchronized* weights (after
   DDP all-reduce the weights are identical across ranks, so the EMA stays consistent). Don't
   all-reduce the EMA itself.
7. **Update frequency**: every step is standard; updating every `k` steps just rescales the effective
   window by `k` (cheaper, occasionally used).
8. **Evaluate with `v`, train with `θ`.** A common pattern: swap `v`→model for validation, then swap
   the raw weights back.

Minimal PyTorch (warmup decay, parameters + buffers):

```python
class ModelEMA:
    def __init__(self, model, decay=0.999, warmup=True):
        self.decay, self.warmup, self.t = decay, warmup, 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.t += 1
        d = min(self.decay, (1 + self.t) / (10 + self.t)) if self.warmup else self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:        # params + float buffers (e.g. BN stats)
                s.mul_(d).add_(v.detach(), alpha=1 - d)
            else:                                # int buffers (e.g. num_batches_tracked)
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
```

---

## 8. Common Pitfalls

1. **β too high for a short run** — the shadow never catches up to the model; EMA looks "stuck" near
   init. Match `N≈1/(1-β)` to run length, or use warmup (§3–4).
2. **Forgetting BatchNorm buffers** — EMA-ing only `parameters()` leaves stale/garbage BN stats →
   terrible eval. EMA or copy the buffers (§7.2).
3. **Evaluating the raw weights** — you did all the averaging and then validated `θ` instead of `v`.
   Swap `v` in for eval.
4. **Not checkpointing the EMA** — a preemption/restart silently restarts the averaging window;
   save and resume the shadow (§7.5).
5. **Updating before the optimizer step** (averaging stale weights) or **more/less than once per
   step** without adjusting β.
6. **Expecting gains with a tiny LR / already-converged model** — if SGD isn't bouncing, there's
   little variance to average away. EMA shines with higher/noisier LR and on long runs.
7. **All-reducing the EMA in DDP** — unnecessary and can desync; update locally from synced weights.

---

## 9. References

- Polyak & Juditsky — *Acceleration of stochastic approximation by averaging*, 1992 (Polyak–Ruppert).
- Izmailov et al. — *Averaging Weights Leads to Wider Optima and Better Generalization* (**SWA**), UAI 2018. [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
- Tarvainen & Valpola — *Mean Teachers* (EMA teacher, semi-supervised), NeurIPS 2017. [arXiv:1703.01780](https://arxiv.org/abs/1703.01780)
- He et al. — *Momentum Contrast (MoCo)* (momentum encoder = EMA), CVPR 2020. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
- Grill et al. — *BYOL* (EMA target network), NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
- Ho et al. — *Denoising Diffusion Probabilistic Models* (EMA weights for sampling), NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- timm `ModelEmaV2`/`ModelEmaV3` — https://github.com/huggingface/pytorch-image-models
- PyTorch `torch.optim.swa_utils.AveragedModel` (supports EMA averaging) — https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging
