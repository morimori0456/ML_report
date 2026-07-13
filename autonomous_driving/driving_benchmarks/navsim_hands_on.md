---
title: "Running NAVSIM Yourself — Lightweight CPU-only PDMS"
description: "A verified, reproducible recipe for running NAVSIM and computing a real PDM Score on a machine with no GPU."
---

A practical, reproducible recipe for actually running **NAVSIM** and getting a **real PDM Score**
on a machine **without an NVIDIA GPU** — by evaluating non-sensor agents on the classic
single-stage metric. Companion to [driving_benchmarks.md](driving_benchmarks.md) (which explains
what PDMS *is*).

> **Verified** on an Intel Core Ultra 7 155H (x86_64, 16C/22T, 62 GB RAM, **no NVIDIA GPU**),
> CPU-only, ~3 GB of data, ~90 s per evaluation.

---

## What this gets you (and what it doesn't)

| Goal | Feasible CPU-only? |
|---|---|
| Set up NAVSIM + nuPlan-devkit | ✅ |
| Evaluate **non-sensor** agents (constant-velocity, human/log-replay) → **real PDMS** | ✅ |
| Build the metric cache (simulation ground truth) | ✅ |
| Train / evaluate **vision** baselines (TransFuser) | ❌ needs CUDA — use a real NVIDIA GPU |

The trick: the official `mini` sensor blobs are **151 GB** (camera+lidar), but the **PDMS metric is
computed from the logs (maps + agent boxes + ego), not the images**. Agents that declare
`SensorConfig.build_no_sensors()` (constant-velocity, human) never touch the sensors — so we
download only **maps (1.4 GB) + mini metadata (1 GB)** and skip the 151 GB entirely.

---

## 0. Check your machine

```bash
uname -m                 # x86_64
nproc; free -h           # cores / RAM
nvidia-smi               # GPU? (this recipe assumes none → CPU)
```

NAVSIM needs its **own** Python 3.9 conda environment (separate from any repo venv). nuPlan-devkit
pins old deps, so don't try to reuse a 3.11/3.12 environment.

---

## 1. Install Miniconda (if absent)

```bash
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
# accept channel ToS (otherwise `conda env create` errors out)
$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$HOME/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 2. Clone NAVSIM and create the env

```bash
mkdir -p ~/navsim_ws && cd ~/navsim_ws
git clone https://github.com/autonomousvision/navsim.git
cd navsim
$HOME/miniconda3/bin/conda env create -f environment.yml     # Python 3.9 + torch 2.0.1 (CPU) + nuplan-devkit
$HOME/miniconda3/envs/navsim/bin/pip install -e .
```

> If the pip step gets interrupted, just re-run `pip install -r requirements.txt` then
> `pip install -e .` — pip is idempotent and fills the gaps.

## 3. Download ONLY maps + mini logs (skip the 151 GB sensors)

```bash
cd ~/navsim_ws/dataset      # mkdir -p first
# maps (~1.4 GB)
wget -q https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
unzip -q nuplan-maps-v1.1.zip && rm nuplan-maps-v1.1.zip && mv nuplan-maps-v1.0 maps
# mini metadata / logs (~1 GB) — NOTE: this is the metadata only, NOT the sensor loops
wget -q https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_mini.tgz
tar -xzf openscene_metadata_mini.tgz && rm openscene_metadata_mini.tgz
mv openscene-v1.1/meta_datas mini_navsim_logs && rm -r openscene-v1.1
# navsim expects ${OPENSCENE_DATA_ROOT}/navsim_logs/<data_split>
mkdir -p navsim_logs && ln -s ../mini_navsim_logs/mini navsim_logs/mini
```

(We deliberately **omit** the two `for split in {0..31}` sensor loops from the official
`download_mini.sh`.)

## 4. Environment variables

```bash
cat > ~/navsim_ws/env.sh <<'EOF'
export NAVSIM_DEVKIT_ROOT=$HOME/navsim_ws/navsim
export OPENSCENE_DATA_ROOT=$HOME/navsim_ws/dataset
export NUPLAN_MAPS_ROOT=$HOME/navsim_ws/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=$HOME/navsim_ws/exp
EOF
mkdir -p ~/navsim_ws/exp
source ~/navsim_ws/env.sh
```

## 5. Build the metric cache (maps + logs, no sensors)

```bash
PY=$HOME/miniconda3/envs/navsim/bin/python
$PY $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=navmini metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache
# -> "Completed dataset caching! All 396 features and targets were cached successfully."
```

(The `Failed to establish connection to the metrics exporter agent` lines are harmless Ray
telemetry warnings.)

---

## 6. The gotcha: v2.0.0's `run_pdm_score.py` is two-stage only

The current NAVSIM (v2.0.0, CoRL'25) `run_pdm_score.py` is hardwired for the **two-stage**
pseudo-simulation benchmark. On a single-stage split like `navmini` it crashes:

```
TypeError: 'NoneType' object is not iterable
  ... scene_loader.reactive_tokens_stage_two  ->  None
```

It also requires the synthetic-scene pickles we don't have. To get the **classic NAVSIM v1
single-stage PDMS** we use a small custom script, [run_pdm_singlestage.py](https://github.com/morimori0456/ML_report/blob/main/autonomous_driving/driving_benchmarks/run_pdm_singlestage.py),
which reuses `navsim.evaluate.pdm_score.pdm_score` with **non-reactive (log-replay)** background
traffic and only the original scenes:

```bash
cp run_pdm_singlestage.py $NAVSIM_DEVKIT_ROOT/navsim/planning/script/
source ~/navsim_ws/env.sh
$PY $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_singlestage.py \
    train_test_split=navmini agent=constant_velocity_agent \
    experiment_name=cv_agent metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

Swap `agent=human_agent` (replays the ground-truth ego trajectory) for an upper-bound reference.

---

## 7. Verified results (navmini, 396 scenarios, CPU, ~90 s)

| Agent | PDMS | NC | DAC | DDC | TLC | EP | TTC | LK | HC |
|---|---|---|---|---|---|---|---|---|---|
| **Human** (replays GT ego) | **0.914** | 1.00 | 1.00 | 1.00 | 1.00 | 0.886 | 1.00 | 1.00 | 0.98 |
| **Constant Velocity** | **0.308** | 0.668 | 0.641 | 0.903 | 0.987 | 0.791 | 0.652 | 0.902 | 0.98 |

This is the PDMS structure from [driving_benchmarks.md](driving_benchmarks.md) made concrete:
- Constant-velocity driving straight **collides** (NC 0.67) and **leaves the drivable area**
  (DAC 0.64) often → the multiplicative safety gates crush the final score to **0.31**.
- The human replay passes every safety gate (NC/DAC/TTC = 1.0) → **0.91**; it isn't a perfect
  1.0 mainly because Ego-Progress (0.89) is measured against the aggressive PDM reference.

---

## 8. Notes / next steps

- **More scenarios**: swap `train_test_split=navmini` → `navtest` (needs the `navtest` logs;
  sensors still optional for non-sensor agents). Build its metric cache first.
- **Vision agents (TransFuser, LTF)**: require the 151 GB sensor blobs **and** a CUDA GPU for any
  reasonable speed. On an iGPU/CPU-only box this is impractical — run it on a real NVIDIA GPU
  (e.g., a Jetson AGX Thor / a CUDA workstation) instead.
- **ego-status MLP agent**: non-sensor, but needs a trained checkpoint to be meaningful
  (untrained = random).
