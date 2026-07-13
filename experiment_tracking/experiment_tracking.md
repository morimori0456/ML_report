---
title: "Experiment Tracking — TensorBoard & Weights & Biases (wandb)"
description: "A practical comparison of TensorBoard and Weights & Biases for logging, comparing, and reproducing training runs."
---

> Two tools for the same job: record what happened during training (metrics, hyperparameters,
> media, system stats) so you can **compare runs, debug, and reproduce**. TensorBoard is the
> local, file-based standard; wandb is a hosted, collaborative platform. Most teams use both.

For a runnable demo that logs to **both** from one training loop (and reads the logs back
inline), see [experiment_tracking_demo.ipynb](https://github.com/morimori0456/ML_report/blob/main/experiment_tracking/experiment_tracking_demo.ipynb).

---

## Table of Contents
1. [Why Track at All](#1-why-track-at-all)
2. [TensorBoard — the Local Standard](#2-tensorboard--the-local-standard)
3. [What You Can Log (and How to Read It)](#3-what-you-can-log-and-how-to-read-it)
4. [Weights & Biases — the Hosted Platform](#4-weights--biases--the-hosted-platform)
5. [wandb Beyond Scalars: Sweeps, Artifacts, Tables](#5-wandb-beyond-scalars-sweeps-artifacts-tables)
6. [Offline / Air-gapped Logging](#6-offline--air-gapped-logging)
7. [Framework Integrations](#7-framework-integrations)
8. [TensorBoard vs wandb — When to Use Which](#8-tensorboard-vs-wandb--when-to-use-which)
9. [Common Pitfalls](#9-common-pitfalls)
10. [References](#10-references)

---

## 1. Why Track at All

`print(loss)` does not scale. Once you have more than a handful of runs you need to answer:
"which hyperparameters gave the best val accuracy?", "did run B diverge or just get unlucky?",
"what data/code produced this checkpoint?". Experiment tracking records, per **run**:

- **scalars over time** — loss, metrics, learning rate, grad norm
- **hyperparameters / config** — so runs are comparable and searchable
- **media** — sample predictions, confusion matrices, attention maps, audio
- **distributions** — weight/gradient histograms (to catch dead/exploding layers)
- **system** — GPU/CPU util, memory, throughput (samples/s)
- **lineage** — code version, dataset/checkpoint artifacts (reproducibility)

---

## 2. TensorBoard — the Local Standard

TensorBoard reads **event files** that your training writes to a log directory, and serves a local
web UI. No account, no network, fully offline. In PyTorch the writer is `SummaryWriter`:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/exp1")           # one dir per run

for step, batch in enumerate(loader):
    loss = train_step(batch)
    writer.add_scalar("train/loss", loss, step)        # tag, value, global_step
writer.add_hparams({"lr": 1e-3, "bs": 32}, {"hparam/val_acc": acc})
writer.close()
```

View it:

```bash
tensorboard --logdir runs        # then open http://localhost:6006
# remote box: ssh -L 6006:localhost:6006 user@server   (tunnel the port)
```

Key idea: **one subdirectory per run** under `--logdir`. TensorBoard auto-discovers them and
overlays their curves so you can compare. The event files are append-only protobufs; you can also
**parse them programmatically** (see §3) without the web UI.

---

## 3. What You Can Log (and How to Read It)

| `SummaryWriter` method | Logs | Use |
|---|---|---|
| `add_scalar(tag, val, step)` | one number over time | loss, accuracy, lr |
| `add_scalars(tag, {a:.., b:..}, step)` | several curves together | train vs val |
| `add_histogram(tag, tensor, step)` | a distribution per step | weights/gradients health |
| `add_image / add_images` | image(s) | sample preds, confusion matrix, feature maps |
| `add_figure` | a matplotlib figure | any custom plot |
| `add_hparams(hparam_dict, metric_dict)` | a row in the HParams table | sweep comparison |
| `add_pr_curve` | precision-recall curve | classifier thresholds |
| `add_embedding` | high-dim vectors → projector | t-SNE/UMAP of features |
| `add_graph(model, input)` | the model compute graph | architecture sanity |
| `add_text` | markdown/text | notes, sample generations |

**Reading events back without the UI** (handy in notebooks / CI):

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("runs/exp1"); ea.Reload()
print(ea.Tags()["scalars"])                 # ['train/loss', 'val/acc', ...]
steps = [e.step  for e in ea.Scalars("val/acc")]
vals  = [e.value for e in ea.Scalars("val/acc")]
```

or with **`tbparse`** → a tidy pandas DataFrame:

```python
from tbparse import SummaryReader
df = SummaryReader("runs", extra_columns={"dir_name"}).scalars   # tag, step, value, run
```

Other tools that read the same event files: `add_histogram` → the Histograms/Distributions tabs;
the **PyTorch Profiler** writes a TensorBoard trace (`tensorboard --logdir` + the profiler plugin)
to find data-loading vs compute bottlenecks.

---

## 4. Weights & Biases — the Hosted Platform

wandb logs to a hosted (or self-hosted) backend and gives you a web dashboard, run comparison,
team sharing, and a database you can query. The core loop mirrors TensorBoard but is one object:

```python
import wandb
run = wandb.init(project="digits", name="exp1",
                 config={"lr": 1e-3, "bs": 32, "model": "mlp"})   # config = hyperparameters
for step, batch in enumerate(loader):
    loss = train_step(batch)
    wandb.log({"train/loss": loss, "lr": sched.get_last_lr()[0]}, step=step)
wandb.log({"val/acc": acc})
wandb.log({"preds": wandb.Image(fig)})        # media: Image, Table, Audio, Html, Plotly, Molecule
run.finish()
```

What you get beyond TensorBoard out of the box:
- **`config`** is first-class and searchable — filter/group/sort runs by any hyperparameter.
- **system metrics** (GPU util/mem/power, CPU, disk, network) logged automatically.
- **run comparison UI**: parallel-coordinates plots, scatter of hparam → metric, grouped runs.
- **collaboration**: share a URL; everyone sees the same live dashboards; **Reports** for writeups.
- **alerts** (`wandb.alert(...)`) → Slack/email when a run diverges or finishes.

---

## 5. wandb Beyond Scalars: Sweeps, Artifacts, Tables

This is where wandb earns its place over plain TensorBoard.

**Sweeps** — hyperparameter search orchestrated by wandb. Define the search, let agents run it:

```yaml
# sweep.yaml
program: train.py
method: bayes                 # grid | random | bayes
metric: {name: val/acc, goal: maximize}
parameters:
  lr:        {distribution: log_uniform_values, min: 1e-4, max: 1e-1}
  batch_size:{values: [32, 64, 128]}
  dropout:   {min: 0.0, max: 0.5}
```
```bash
wandb sweep sweep.yaml          # -> prints a SWEEP_ID
wandb agent <SWEEP_ID>          # run one or many agents (across machines) to execute trials
```
The agent injects sampled hyperparameters into `wandb.config`; Bayesian search uses past trials to
propose the next. (On Slurm: launch N `wandb agent` tasks as a job array — see the infra guide.)

**Artifacts** — versioned datasets/models/checkpoints with lineage:

```python
art = wandb.Artifact("model", type="model"); art.add_file("ckpt.pt")
run.log_artifact(art)                    # versioned: model:v0, v1, ...
# later / elsewhere:
art = run.use_artifact("model:latest"); path = art.download()
```
This records *which run produced which checkpoint from which dataset* — the reproducibility chain.

**Tables** — log rich per-example data (inputs, predictions, scores) and explore/filter in the UI:

```python
t = wandb.Table(columns=["image", "pred", "label", "conf"])
t.add_data(wandb.Image(x), pred, label, conf)
wandb.log({"samples": t})
```

---

## 6. Offline / Air-gapped Logging

Clusters often have no internet on compute nodes. wandb supports it:

```bash
export WANDB_MODE=offline        # or wandb.init(mode="offline"); writes to ./wandb/offline-run-*
# ... training runs, logs to local files, no network ...
wandb sync wandb/offline-run-*   # upload later from a login node with internet
```
Modes: `online` (default), `offline` (local, sync later), `disabled` (no-op, for unit tests).
TensorBoard is offline by nature. wandb can also **mirror TensorBoard**: `wandb.init(sync_tensorboard=True)`
auto-imports anything written via `SummaryWriter`. For full data control there is **self-hosted
wandb** (W&B Server) on-prem.

---

## 7. Framework Integrations

You rarely call the loggers by hand; frameworks wire them in:

| Framework | TensorBoard | wandb |
|---|---|---|
| **PyTorch Lightning** | `TensorBoardLogger` | `WandbLogger` |
| **HuggingFace Trainer** | `report_to="tensorboard"` | `report_to="wandb"` (set `WANDB_PROJECT`) |
| **Keras/TF** | `tf.keras.callbacks.TensorBoard` | `WandbMetricsLogger` callback |
| **mmengine** | `TensorboardVisBackend` | `WandbVisBackend` (see the mmEngine guide) |
| **Ultralytics/YOLO, fastai, …** | built-in | built-in |

Typically: pick the logger, pass your hyperparameters as `config`, and the callback logs loss/metrics
each step plus checkpoints as artifacts — no manual `add_scalar`.

---

## 8. TensorBoard vs wandb — When to Use Which

| Aspect | TensorBoard | Weights & Biases |
|---|---|---|
| Hosting | local files, self-served | hosted SaaS (or self-host) |
| Account / network | none | account; offline mode available |
| Hyperparameter search | manual (HParams view only) | **Sweeps** (grid/random/bayes, distributed) |
| Run comparison | overlay curves | rich: parallel-coords, group/filter by config |
| Collaboration | share files / a server | URLs, teams, **Reports** |
| Artifacts / lineage | none built-in | **versioned artifacts** |
| Media / tables | images, embeddings, PR curves | images, **interactive Tables**, audio, 3D |
| System metrics | no | automatic |
| Cost | free | free tier; paid for teams/private at scale |
| Best for | quick local debugging, profiler traces, privacy | team projects, sweeps, long-term tracking |

Rule of thumb: **TensorBoard for fast local iteration and the PyTorch profiler; wandb when you run
sweeps, work in a team, or need artifact lineage.** They are not mutually exclusive —
`sync_tensorboard=True` lets you keep `SummaryWriter` calls and get the wandb dashboard too.

---

## 9. Common Pitfalls

1. **Inconsistent global step.** Logging some metrics by epoch and others by iteration to the same
   x-axis makes curves misalign. Pick one step convention (usually global iteration) and pass it
   explicitly.
2. **One writer/run shared across processes in DDP.** In multi-GPU training, **log only from rank 0**
   — otherwise N processes write duplicate/garbled events (or N wandb runs). Guard with
   `if rank == 0:`.
3. **Logging every step at high frequency.** Flooding scalars (and especially histograms/images)
   slows training and bloats logs. Log scalars every K steps; media rarely.
4. **Forgetting `writer.close()` / `run.finish()`.** Buffered events may not flush; the run shows
   as "running"/crashed.
5. **Histograms/images every step.** Expensive and huge. Throttle hard.
6. **Not setting `config`/hparams.** Without logged hyperparameters you can't compare or search runs
   later — the whole point.
7. **Secrets in logs.** Don't log API keys/PII; `WANDB_API_KEY` belongs in env, not in code/config.
8. **Same `log_dir` for multiple runs (TB)** → curves stack into one confusing run. One dir per run.

---

## 10. References

- TensorBoard — https://www.tensorflow.org/tensorboard ; PyTorch `SummaryWriter` — https://pytorch.org/docs/stable/tensorboard.html
- PyTorch Profiler + TensorBoard — https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
- Weights & Biases docs — https://docs.wandb.ai/ ; Sweeps — https://docs.wandb.ai/guides/sweeps ; Artifacts — https://docs.wandb.ai/guides/artifacts
- `tbparse` (event files → pandas) — https://github.com/j3soon/tbparse
- Related in this repo: [ML Training Infrastructure](../infrastructure/ml_training_infrastructure.md)
  (sweeps as Slurm job arrays, monitoring), [mmEngine guide](../autonomous_driving/mmengine/mmengine_guide.md)
  (vis backends).
