# ML Report

A repository of survey reports on machine learning and autonomous driving papers and implementations.

---

## Setup (uv shared environment)

The shared environment for running all notebooks is managed with [uv](https://docs.astral.sh/uv/).

```bash
# If uv is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the core environment (6 notebooks, no GPU required)
uv sync

# Launch Jupyter
uv run jupyter lab
```

`uv sync` reconstructs `.venv/` from `pyproject.toml` / `uv.lock` (numpy, matplotlib, scipy, opencv-python, jupyterlab).

### Transformer + mmEngine notebooks

Minimal pure-PyTorch and mmEngine demos. **Run on CPU torch** (no GPU required).

```bash
uv sync --extra transformer   # adds torch + mmengine + scikit-learn (CPU)
```

### GPU fine-tuning notebook (`llm/lora_qlora_finetune.ipynb`)

This is the only notebook that requires a CUDA GPU (QLoRA / bitsandbytes). Install the additional dependencies in a CUDA environment or on Colab.

```bash
uv sync --extra llm-gpu   # torch / transformers / peft / trl / bitsandbytes / datasets / accelerate
```

> bitsandbytes requires CUDA and will fail to import on CPU / aarch64. It is intentionally separated from the core dependencies.

| notebook | required dependencies |
|---|---|
| `llm/kv_cache_demo.ipynb` | core only |
| `llm/lora_qlora_demo.ipynb` | core only |
| `autonomous_driving/camera_calibration/extrinsic_calibration_demo.ipynb` | core only |
| `autonomous_driving/camera_calibration/extrinsic_calibration_opencv.ipynb` | core only (opencv); uses real chessboard images in `data/chessboard/` (auto-downloaded if absent) |
| `autonomous_driving/VAD/vad_dataloader_demo.ipynb` | core only |
| `autonomous_driving/VAD/nuscenes_coordinate_transform.ipynb` | core only |
| `autonomous_driving/drive_transformer/drive_transformer_demo.ipynb` | `--extra transformer` (CPU torch) |
| `autonomous_driving/mmengine/mmengine_demo.ipynb` | `--extra transformer` (CPU torch + mmengine) |
| `distillation/knowledge_distillation_demo.ipynb` | `--extra transformer` (CPU torch + scikit-learn) |
| `llm/lora_qlora_finetune.ipynb` | `--extra llm-gpu` (CUDA GPU) |

---

## Directory structure

```
ML_report/
├── distillation/
│   ├── knowledge_distillation.md         # KD complete guide (response/feature/relation; FitNets, AT, FSP, NST, PKT, RKD, CRD, OFD, ReviewKD)
│   └── knowledge_distillation_demo.ipynb # logit-KD vs FitNets vs Attention Transfer on a small CNN (transfer-set regime)
├── llm/
│   ├── kv_cache.md             # KV Cache complete guide (transformers / vLLM code analysis)
│   ├── kv_cache_demo.ipynb     # KV Cache demo (numpy)
│   ├── lora_qlora.md           # LoRA / QLoRA complete guide (principles, NF4 quantization, PEFT code analysis)
│   ├── lora_qlora_demo.ipynb   # Conceptual demo (numpy only, no GPU)
│   └── lora_qlora_finetune.ipynb # Real QLoRA fine-tuning (Colab/GPU, PEFT/trl/bitsandbytes)
└── autonomous_driving/
    ├── localization_tech.md    # Localization technology survey (sensor fusion overview)
    ├── camera_calibration/     # Camera extrinsic calibration
    │   ├── extrinsic_calibration.md            # Complete guide ([R|t], PnP, epipolar, rectification)
    │   ├── extrinsic_calibration_demo.ipynb    # Conceptual demo (numpy only, no GPU)
    │   └── extrinsic_calibration_opencv.ipynb  # OpenCV in practice (calibrate/solvePnP/stereoRectify)
    ├── drive_transformer/      # DriveTransformer (ICLR 2025, E2E autonomous driving)
    │   ├── drive_transformer.md                # Complete guide (task parallelism, sparse representation, streaming)
    │   └── drive_transformer_demo.ipynb        # Minimal PyTorch implementation (3 attention types, FIFO, 6-mode planning)
    ├── mmengine/               # mmEngine training framework (OpenMMLab)
    │   ├── mmengine_guide.md                   # Complete guide (Runner, Registry, Config, Hook, Evaluator)
    │   └── mmengine_demo.ipynb                 # Raw PyTorch vs mmEngine side-by-side (Runner, CheckpointHook, BinaryAccuracy)
    ├── driving_benchmarks/     # Modern AD benchmarks & evaluation metrics
    │   └── driving_benchmarks.md               # NAVSIM (PDMS), ROADWork, Impromptu VLA, Alpamayo-R1 — what each metric measures
    └── VAD/                   # VAD (Vectorized Scene Representation)
        ├── dataloader.md       # nuScenes dataloader implementation guide
        ├── nuscenes_dataset.md # nuScenes dataset detailed guide (with ego_pose positioning notes)
        └── ego_trajectory.md   # Ego trajectory (gt_ego_his/fut_trajs) computation logic
```

---

## Report list

### Model Compression

| Title | Topics | Link |
|---|---|---|
| Knowledge Distillation (feature-focused) | Response/feature/relation families; logit KD, FitNets hints+regressor, Attention Transfer, FSP/NST/PKT/RKD/CRD/OFD/ReviewKD; adapters for dim mismatch, transforms, loss weighting | [knowledge_distillation.md](distillation/knowledge_distillation.md) + [logit-KD vs FitNets vs AT demo](distillation/knowledge_distillation_demo.ipynb) |

### LLM

| Title | Topics | Link |
|---|---|---|
| KV Cache Complete Guide | Principles, memory calculation, PagedAttention, Prefix Caching, MLA, quantization (transformers/vLLM code analysis) | [llm/kv_cache.md](llm/kv_cache.md) |
| LoRA / QLoRA Complete Guide | Low-rank decomposition, α/r scaling, NF4 quantization, Double Quant, memory calculation, PEFT/bitsandbytes code analysis, DoRA and other variants | [llm/lora_qlora.md](llm/lora_qlora.md) + [conceptual demo](llm/lora_qlora_demo.ipynb) / [real fine-tuning](llm/lora_qlora_finetune.ipynb) |

### Autonomous Driving (common technology)

| Title | Topics | Link |
|---|---|---|
| Localization Technology Survey | KF/EKF, NDT, SLAM, VIO, DL-based positioning, sensor fusion overview | [autonomous_driving/localization_tech.md](autonomous_driving/localization_tech.md) |
| Camera Extrinsic Calibration Complete Guide | Coordinate systems, [R\|t], projection P=K[R\|t], PnP/DLT, epipolar geometry, stereo rectification, disparity→depth, camera-LiDAR | [extrinsic_calibration.md](autonomous_driving/camera_calibration/extrinsic_calibration.md) + [conceptual demo](autonomous_driving/camera_calibration/extrinsic_calibration_demo.ipynb) / [OpenCV in practice](autonomous_driving/camera_calibration/extrinsic_calibration_opencv.ipynb) |
| DriveTransformer Complete Guide | Unified Transformer-based E2E autonomous driving, task parallelism (Self-Attn), sparse representation (BEV-free Sensor Cross-Attn), streaming FIFO (Temporal Cross-Attn), 6-mode planning WTA | [drive_transformer.md](autonomous_driving/drive_transformer/drive_transformer.md) + [minimal implementation demo](autonomous_driving/drive_transformer/drive_transformer_demo.ipynb) |
| mmEngine Complete Guide | Runner, Registry, Config (_base_ inheritance), Hook system, Evaluator/Metric; raw PyTorch vs mmEngine side-by-side on synthetic 2-class data | [mmengine_guide.md](autonomous_driving/mmengine/mmengine_guide.md) + [side-by-side demo](autonomous_driving/mmengine/mmengine_demo.ipynb) |
| Modern AD Benchmarks & Metrics | NAVSIM PDMS/EPDMS (gated weighted score), ROADWork work-zone tasks (AP/1-NED/SPICE/AE%), Impromptu VLA (nuScenes L2 / NeuroNCAP / diagnostic QA), Alpamayo-R1 (open/closed-loop + reasoning-quality) | [driving_benchmarks.md](autonomous_driving/driving_benchmarks/driving_benchmarks.md) |

### Autonomous Driving (VAD)

| Title | Topics | Link |
|---|---|---|
| VAD Dataloader Implementation Guide | nuScenes-format data loading, HD map generation, temporal queue | [autonomous_driving/VAD/dataloader.md](autonomous_driving/VAD/dataloader.md) |
| nuScenes Dataset Detailed Guide | Sensor configuration, data hierarchy, annotations, maps, ego_pose positioning accuracy | [autonomous_driving/VAD/nuscenes_dataset.md](autonomous_driving/VAD/nuscenes_dataset.md) |
| Ego Trajectory Computation Logic Guide | gt_ego_his_trajs / gt_ego_fut_trajs coordinate transforms, sequential differences, model usage | [autonomous_driving/VAD/ego_trajectory.md](autonomous_driving/VAD/ego_trajectory.md) |

---

## Contribution guidelines

- Create a top-level directory for each topic (e.g., `nlp/`, `generative/`)
- Create a subdirectory for each paper or implementation
- Use snake_case filenames that concisely describe the report content
