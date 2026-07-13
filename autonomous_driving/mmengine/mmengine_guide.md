---
title: "mmEngine Complete Guide — OpenMMLab's Unified Training Engine"
description: "A guide to mmEngine, the training framework underlying mmDetection, mmDetection3D, and DriveTransformer, with a runnable comparison against raw PyTorch."
---

> mmEngine: [open-mmlab/mmengine](https://github.com/open-mmlab/mmengine) — the foundational training framework powering mmDetection, mmDetection3D, and DriveTransformer.

For a runnable hands-on demo comparing raw PyTorch vs mmEngine, see [mmengine_demo.ipynb](mmengine_demo.ipynb).

---

## Table of Contents
1. [What Problem Does mmEngine Solve?](#1-what-problem-does-mmengine-solve)
2. [Core Concepts: Five Pillars](#2-core-concepts-five-pillars)
3. [Runner — The Training Loop Abstraction](#3-runner--the-training-loop-abstraction)
4. [Registry — Component Swapping Without Code Changes](#4-registry--component-swapping-without-code-changes)
5. [Config — Hierarchical, Inheritable Configuration](#5-config--hierarchical-inheritable-configuration)
6. [Hook System — Callbacks Without Boilerplate](#6-hook-system--callbacks-without-boilerplate)
7. [Evaluator and Metrics](#7-evaluator-and-metrics)
8. [How DriveTransformer Uses mmEngine](#8-how-drivetransformer-uses-mmengine)
9. [Common Pitfalls](#9-common-pitfalls)

---

## 1. What Problem Does mmEngine Solve?

Every PyTorch training script reinvents the same boilerplate:

```python
# What every researcher writes from scratch — repeatedly
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
    
    # Manually added logging
    # Manually added checkpoint saving
    # Manually added evaluation
    # Manually added LR scheduling
    # Manually added distributed sync
```

This works for one project. But when you have 10 papers to reproduce, each with subtle loop differences (gradient clipping, mixed precision, EMA, warmup), the cost compounds.

**mmEngine's answer**: standardize the loop into a `Runner`, and handle everything else through a `Hook` system. The researcher writes only the parts that are unique to their paper.

---

## 2. Core Concepts: Five Pillars

| Pillar | Class | What It Replaces |
|---|---|---|
| **Runner** | `mmengine.runner.Runner` | Manual `for epoch / for batch` loop |
| **Registry** | `mmengine.Registry` | Manual `if model_name == 'X': ...` factories |
| **Config** | `mmengine.Config` | `argparse` + hardcoded dicts + YAML |
| **Hook** | `mmengine.hook.Hook` | Scattered `if epoch % N == 0:` callbacks |
| **Evaluator** | `mmengine.evaluator.Evaluator` | Manual metric accumulation |

---

## 3. Runner — The Training Loop Abstraction

`Runner` is the central object. Pass it a config dict (or Config object) and call `.train()`.

```python
from mmengine.runner import Runner

runner = Runner(
    model=dict(type='MyModel'),           # built via Registry
    work_dir='./work_dirs/exp1',
    train_dataloader=dict(
        dataset=dict(type='MyDataset'),
        batch_size=32,
        num_workers=4,
    ),
    train_cfg=dict(by_epoch=True, max_epochs=10),
    optim_wrapper=dict(optimizer=dict(type='AdamW', lr=1e-3)),
    default_hooks=dict(
        checkpoint=dict(type='CheckpointHook', interval=1),
        logger=dict(type='LoggerHook', interval=50),
    ),
)
runner.train()
```

What you get automatically:
- Per-iteration and per-epoch logging (loss, lr, eta)
- Checkpoint saving and best-model tracking
- Resume from checkpoint (`runner.resume('latest.pth')`)
- Distributed training (change `launcher='pytorch'` — zero code change)
- Mixed precision (`optim_wrapper=dict(type='AmpOptimWrapper', ...)`)
- Gradient clipping (`optim_wrapper=dict(clip_grad=dict(max_norm=35))`)

### Training loop internals

```
Runner.train()
  └─ EpochBasedTrainLoop / IterBasedTrainLoop
       ├─ before_train_epoch() → calls all registered hooks
       ├─ for batch in dataloader:
       │    ├─ before_train_iter()
       │    ├─ model.train_step(data, optim_wrapper)
       │    │    └─ loss = model(data)
       │    │       optim_wrapper.update_params(loss)
       │    └─ after_train_iter()
       └─ after_train_epoch()
```

The model only needs to implement `forward()` returning a loss dict. Everything else is handled.

---

## 4. Registry — Component Swapping Without Code Changes

Registry is a global lookup table: `name_string → class`.

```python
from mmengine.model import BaseModel
from mmengine import MODELS

@MODELS.register_module()
class MyDetector(BaseModel):
    def forward(self, inputs, data_samples, mode='tensor'):
        ...

# Build from config dict — no import needed at call site
model = MODELS.build(dict(type='MyDetector', num_classes=80))
```

Why this matters for research:
- Swap backbone: change `type='ResNet50'` → `type='EVA02'` in config, no code changes
- Grid search over architectures: config files only, single codebase
- DriveTransformer ships 4 model sizes (tiny/small/base/large) as 4 config files, same code

---

## 5. Config — Hierarchical, Inheritable Configuration

```python
# configs/my_exp.py
_base_ = ['./base_model.py', './dataset_nuscenes.py']  # inherit two bases

# Override only what changes
model = dict(
    decoder=dict(num_layers=12, hidden_dim=768)  # Large config
)
train_cfg = dict(max_epochs=24)
```

```python
from mmengine import Config
cfg = Config.fromfile('configs/my_exp.py')
cfg.model.decoder.hidden_dim  # → 768
```

`_base_` inheritance is a tree merge: child values override parent values recursively. This is how DriveTransformer's `drivetransformer_large.py` inherits from `drivetransformer_base.py` and overrides only `D=768, L=12`.

---

## 6. Hook System — Callbacks Without Boilerplate

Hooks intercept 14 points in the training loop:

```
before_run → before_train → before_train_epoch → before_train_iter
→ after_train_iter → after_train_epoch → before_val_epoch → ...
→ after_val_epoch → after_train → after_run
```

Built-in hooks (selected):

| Hook | What It Does |
|---|---|
| `CheckpointHook` | Save every N epochs; save best by metric |
| `LoggerHook` | Print/TensorBoard/WandB logging |
| `ParamSchedulerHook` | Step LR schedulers at right timing |
| `EarlyStoppingHook` | Stop when metric plateaus |
| `EMAHook` | Exponential moving average of weights |
| `SyncBuffersHook` | Sync BN running stats in DDP |
| `VisualizationHook` | Draw predictions and log as images |

Custom hook:
```python
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class MyGradientMonitorHook(Hook):
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        for name, p in runner.model.named_parameters():
            if p.grad is not None and p.grad.norm() > 100:
                runner.logger.warning(f'Large gradient: {name}')
```

---

## 7. Evaluator and Metrics

```python
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

@METRICS.register_module()
class MyAccuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        # Called per batch — accumulate into self.results
        pred = [s['pred_label'] for s in data_samples]
        gt   = [s['gt_label']   for s in data_samples]
        self.results.append({'pred': pred, 'gt': gt})

    def compute_metrics(self, results):
        # Called once per epoch — aggregate and return dict
        all_pred = sum([r['pred'] for r in results], [])
        all_gt   = sum([r['gt']   for r in results], [])
        acc = sum(p == g for p, g in zip(all_pred, all_gt)) / len(all_gt)
        return dict(accuracy=acc)
```

Pass to Runner as `val_evaluator=dict(type='MyAccuracy')`. Runner handles the loop and logging.

---

## 8. How DriveTransformer Uses mmEngine

| Component | mmEngine feature used |
|---|---|
| Model `DriveTransformerE2E` | `@MODELS.register_module()`, inherits `BaseModel` |
| Config inheritance | `_base_ = ['drivetransformer_base.py']`, Large overrides `D/L` |
| Training loop | `Runner` with `IterBasedTrainLoop` (iter-based, not epoch) |
| Optimizer | `AdamW` + `AmpOptimWrapper` (FP16 mixed precision) |
| LR schedule | `CosineAnnealingLR` via `ParamSchedulerHook` |
| Checkpointing | `CheckpointHook(save_best='NDS')` — saves best nuScenes Detection Score |
| Logging | `LoggerHook` → TensorBoard + wandb |
| Distributed | 8× A100; `launcher='pytorch'`, zero model code changes |
| Evaluation | Custom `NuScenesMetric` and `Bench2DriveMetric` |

The Large model (~646M params) trained for 24 epochs on 8×A100 — none of the distributed/AMP/checkpoint code lives in the model itself.

---

## 9. Common Pitfalls

1. **`model.forward()` signature must match `mode` argument** — mmEngine calls `forward(inputs, data_samples, mode='loss'|'predict'|'tensor')`. The mode switch is mandatory for `BaseModel` subclasses.

2. **`data_preprocessor` is part of the model** — mmEngine's `BaseModel` has a `data_preprocessor` attribute that normalizes inputs. Forgetting to configure it leaves normalization out.

3. **Hook `priority` matters** — If two hooks listen at the same point, `priority` (int or string like `'NORMAL'`, `'HIGH'`) determines order. `CheckpointHook` runs after `LoggerHook` by default.

4. **`work_dir` collision in sweep experiments** — Always parameterize `work_dir` per run (e.g., `f'work_dirs/{cfg.model.type}_{lr}'`), otherwise checkpoints overwrite each other.

5. **Custom components must be imported before `Runner.build()`** — Registry is populated at import time. If your custom `@MODELS.register_module()` class isn't imported, `build(dict(type='MyModel'))` raises `KeyError`.

---

## References

- [mmEngine docs](https://mmengine.readthedocs.io/)
- [mmEngine GitHub](https://github.com/open-mmlab/mmengine)
- DriveTransformer config: `adzoo/drivetransformer/configs/drivetransformer_large.py`
- Related: [DriveTransformer Guide](../drive_transformer/drive_transformer.md)
