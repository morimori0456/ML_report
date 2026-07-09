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
| `distillation/feature_distillation_why.ipynb` | `--extra transformer` (CPU torch + scikit-learn) |
| `distillation/advanced_kd_practical.ipynb` | `--extra transformer` (CPU torch + scikit-learn) |
| `autonomous_driving/VAD/nuscenes_coordinate_transform.ipynb` | core only |
| `autonomous_driving/drive_transformer/drive_transformer_demo.ipynb` | `--extra transformer` (CPU torch) |
| `autonomous_driving/mmengine/mmengine_demo.ipynb` | `--extra transformer` (CPU torch + mmengine) |
| `distillation/knowledge_distillation_demo.ipynb` | `--extra transformer` (CPU torch + scikit-learn) |
| `agentic_engineering/loop_engineering_demo.ipynb` | core only |
| `agentic_engineering/loop_design_playbook_demo.ipynb` | core only |
| `distillation/foundation_model_distillation_demo.ipynb` | `--extra transformer` (CPU torch + scikit-learn) |
| `distillation/multi_teacher_distillation_demo.ipynb` | `--extra transformer` (CPU torch + scikit-learn) |
| `distillation/self_distillation_demo.ipynb` | `--extra transformer` (CPU torch + scikit-learn) |
| `ema/weight_ema_demo.ipynb` | `--extra transformer` (CPU torch + scikit-learn) |
| `experiment_tracking/experiment_tracking_demo.ipynb` | `--extra transformer` (CPU torch + tensorboard + wandb + tbparse) |
| `llm/lora_qlora_finetune.ipynb` | `--extra llm-gpu` (CUDA GPU) |

---

## Directory structure

```
ML_report/
├── infrastructure/
│   ├── ml_training_infrastructure.md     # Training-platform guide: Slurm (srun/sbatch), GRES, NCCL/IB, parallel storage, containers, K8s, monitoring
│   └── examples/                          # Slurm sbatch templates (single-node, multi-node, Pyxis container, sweep array)
├── experiment_tracking/
│   ├── experiment_tracking.md            # TensorBoard & wandb guide (logging, sweeps, artifacts, offline, integrations, comparison)
│   └── experiment_tracking_demo.ipynb    # Log to both from one loop; read TB events back with tbparse; inspect wandb offline run
├── ema/
│   ├── weight_ema.md                     # Weight EMA guide (update rule, variance reduction, decay↔window, warmup, EMA vs SWA, where it's essential)
│   └── weight_ema_demo.ipynb             # EMA hands-on: toy variance reduction, decay/window, MLP curve, BatchNorm-buffer pitfall
├── distillation/
│   ├── knowledge_distillation.md         # KD complete guide (response/feature/relation; FitNets, AT, FSP, NST, PKT, RKD, CRD, OFD, ReviewKD)
│   ├── knowledge_distillation_demo.ipynb # logit-KD vs FitNets vs Attention Transfer on a small CNN (transfer-set regime)
│   ├── feature_distillation_why.ipynb    # Why intermediate features beat logits: 5 methods, CKA, t-SNE, attention maps
│   ├── advanced_kd_practical.md          # Modern KD for production: DIST, Logit Standardisation, CTKD, SimKD, hooks, debugging
│   ├── advanced_kd_practical.ipynb       # Benchmark 6 methods + FeatureExtractor + DistillationMonitor hands-on
│   ├── distillation_methods_survey.md    # Broader survey: DKD, TAKD, online/self/data-free, detection/segmentation, NLP/LLM (TinyBERT, MiniLM, MiniLLM, GKD)
│   ├── foundation_model_distillation.md  # Foundation model distillation: FM definition, BERT/CLIP/DINO methods, capacity gap, TAKD, cross-modal, black-box API
│   ├── foundation_model_distillation_demo.ipynb  # FM teacher richness, data efficiency, capacity gap, TAKD, CLIP-style cosine distillation, feature adapter
│   ├── multi_teacher_distillation.md     # Multi-teacher KD: aggregation taxonomy (uniform/weighted/adaptive/gradient-space), CA-MKD, AEKD, AM-RADIO/Theia FM agglomeration
│   ├── multi_teacher_distillation_demo.ipynb  # 3 diverse teachers, error-correlation analysis, 5-student comparison (scratch/single/uniform/entropy/CA-MKD), clone-pool negative control
│   ├── self_distillation.md              # Self-distillation deep dive: BAN/BYOT/EMA/DINO variants, 3 theories (Mobahi regularization, multi-view ensembling, instance-specific smoothing), LLM self-improvement links
│   └── self_distillation_demo.ipynb      # Channel experiments under label noise (soft-KL vs pseudo-labels), BAN chain saturation, warm-start probe, Mobahi kernel-regression collapse
├── llm/
│   ├── kv_cache.md             # KV Cache complete guide (transformers / vLLM code analysis)
│   ├── kv_cache_demo.ipynb     # KV Cache demo (numpy)
│   ├── lora_qlora.md           # LoRA / QLoRA complete guide (principles, NF4 quantization, PEFT code analysis)
│   ├── lora_qlora_demo.ipynb   # Conceptual demo (numpy only, no GPU)
│   └── lora_qlora_finetune.ipynb # Real QLoRA fine-tuning (Colab/GPU, PEFT/trl/bitsandbytes)
├── agentic_engineering/
│   ├── loop_engineering.md               # Loop Engineering guide: automations, worktrees, skills, MCP, sub-agents, persistent state
│   ├── loop_engineering_demo.ipynb       # Runnable demos: LoopScheduler, WorktreeManager, SkillLoader, DualAgentVerifier, LoopMemory
│   ├── loop_design_playbook.md           # Real-world loop case studies (Ralph Wiggum, Bun port, Anthropic teams) -> design decisions, verifier theory, ops runbook
│   └── loop_design_playbook_demo.ipynb   # Simulations: retry math, budget sizing, verifier false-pass rates, cache-cliff cost, ops-metrics dashboard
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
    │   ├── driving_benchmarks.md               # NAVSIM (PDMS), ROADWork, Impromptu VLA, Alpamayo-R1 — what each metric measures
    │   ├── navsim_hands_on.md                  # Verified CPU-only recipe to run NAVSIM & get real PDMS (no GPU)
    │   └── run_pdm_singlestage.py              # Custom single-stage PDMS script (v2.0.0 run_pdm_score is two-stage only)
    └── VAD/                   # VAD (Vectorized Scene Representation)
        ├── dataloader.md       # nuScenes dataloader implementation guide
        ├── nuscenes_dataset.md # nuScenes dataset detailed guide (with ego_pose positioning notes)
        └── ego_trajectory.md   # Ego trajectory (gt_ego_his/fut_trajs) computation logic
```

---

## Report list

### Agentic Engineering

| Title | Topics | Link |
|---|---|---|
| Loop Engineering | Paradigm shift from prompting to system design; 6 loop components (automations, worktrees, skills, MCP connectors, sub-agents, persistent state); 3 architecture patterns (triage, writer+reviewer, full autonomous); verification/comprehension/cognitive-surrender debts; 8-step implementation roadmap | [loop_engineering.md](agentic_engineering/loop_engineering.md) + [hands-on demo](agentic_engineering/loop_engineering_demo.ipynb) |
| Loop Design Playbook | Real-world cases (Ralph Wiggum loop, Bun 750k-line Rust port, Anthropic internal teams, /goal + routines); loop taxonomy (5 mechanisms); 5 design decisions with retry-math budget sizing; verifier error model (false-pass compounding); loop portfolio worked example; operations runbook (cron/flock/kill switches, cache-cliff cost control); weekly ops metrics | [loop_design_playbook.md](agentic_engineering/loop_design_playbook.md) + [simulations](agentic_engineering/loop_design_playbook_demo.ipynb) |

### Infrastructure / MLOps

| Title | Topics | Link |
|---|---|---|
| ML Training Infrastructure | Slurm (srun/sbatch/salloc, GRES), srun×torchrun distributed launch, NCCL/InfiniBand, parallel storage (Lustre/GPFS/BeeGFS) & data-loading, Enroot+Pyxis/Apptainer, checkpoint/preemption/elastic, Kubernetes (Volcano/Kubeflow), DCGM monitoring, cluster provisioning | [ml_training_infrastructure.md](infrastructure/ml_training_infrastructure.md) + [sbatch templates](infrastructure/examples/) |
| Experiment Tracking (TensorBoard & wandb) | SummaryWriter logging (scalars/histograms/images/hparams) & reading events back (tbparse), wandb runs/config/system metrics, sweeps, artifacts, tables, offline mode + `wandb sync`, framework integrations, comparison & pitfalls | [experiment_tracking.md](experiment_tracking/experiment_tracking.md) + [dual-logging demo](experiment_tracking/experiment_tracking_demo.ipynb) |

### Training Techniques

| Title | Topics | Link |
|---|---|---|
| Weight EMA | Update rule & Polyak view, variance reduction, decay↔window (N≈1/(1-β)), bias/warmup, EMA vs SWA vs Polyak–Ruppert, uses (diffusion/MoCo/BYOL/Mean-Teacher/RL), BatchNorm-buffer pitfall | [weight_ema.md](ema/weight_ema.md) + [hands-on demo](ema/weight_ema_demo.ipynb) |

### Model Compression

| Title | Topics | Link |
|---|---|---|
| Knowledge Distillation (feature-focused) | Response/feature/relation families; logit KD, FitNets hints+regressor, Attention Transfer, FSP/NST/PKT/RKD/CRD/OFD/ReviewKD; adapters for dim mismatch, transforms, loss weighting | [knowledge_distillation.md](distillation/knowledge_distillation.md) + [logit-KD vs FitNets vs AT demo](distillation/knowledge_distillation_demo.ipynb) |
| Feature Distillation — Why Intermediate Features Beat Logits | Information bottleneck (10 vs 4,096 dims), gradient-path analysis, 5-method comparison (Scratch / Hinton KD / DKD / FitNets / AT), CKA representation alignment, t-SNE & attention-map visualisation | [feature_distillation_why.md](distillation/feature_distillation_why.md) + [hands-on demo](distillation/feature_distillation_why.ipynb) |
| Advanced KD — Practical Techniques for Production | Three failure modes of classic KD (scale, washout, label conflict); DIST (NeurIPS 2022, Pearson correlation); Logit Standardisation (CVPR 2024); CTKD curriculum temperature (AAAI 2023); SimKD (CVPR 2022); hook-based feature extractor; multi-loss uncertainty weighting; DistillationMonitor; production drop-in recipe | [advanced_kd_practical.md](distillation/advanced_kd_practical.md) + [benchmark demo](distillation/advanced_kd_practical.ipynb) |
| Distillation Methods — Broader Survey | Better logit losses (DKD/TCKD-NCKD, WSLD/NKD), capacity gap (TAKD), offline/online/self schemes (DML, BAN, BYOT), data-free (DeepInversion, DAFL), detection (FGD/FGFI/LD), segmentation (CWD/SKD), NLP/LLM (DistilBERT, TinyBERT, MiniLM, seq-level KD, MiniLLM reverse-KL, GKD on-policy) | [distillation_methods_survey.md](distillation/distillation_methods_survey.md) |
| Foundation Model Distillation | What is a foundation model (scale, emergence, generality); FM families (LLM/VFM/multimodal); why FM teachers produce richer soft labels; capacity gap & TAKD fix; DistilBERT/TinyBERT/MiniLM NLP distillation; DINOv2/SAM/CLIP-KD vision distillation; cross-modal DeiT distillation token; black-box/API-only (seq-level KD, GKD); ToS cautions; practical checklist | [foundation_model_distillation.md](distillation/foundation_model_distillation.md) + [demo](distillation/foundation_model_distillation_demo.ipynb) |
| Multi-Teacher Distillation | Ensemble compression vs complementary expertise vs backbone unification; aggregation taxonomy (uniform/accuracy-weighted/entropy EBKD/CA-MKD/learned gating AMTML-KD); AEKD gradient-space conflict resolution; RL teacher selection; feature-level adapters; AM-RADIO & Theia foundation-model agglomeration; teacher-diversity rules; production recipe & pitfalls | [multi_teacher_distillation.md](distillation/multi_teacher_distillation.md) + [demo](distillation/multi_teacher_distillation_demo.ipynb) |
| Self-Distillation | Same-architecture distillation paradox; 4 variants (born-again BAN, in-network BYOT, EMA teachers, DINO); 3 theories (Mobahi regularization amplification, Allen-Zhu multi-view ensembling, instance-specific label smoothing); gain-channel analysis (dark knowledge vs argmax denoising); LLM self-improvement (STaR, SDFT, model collapse); recipes and warm-start channel probe | [self_distillation.md](distillation/self_distillation.md) + [demo](distillation/self_distillation_demo.ipynb) |

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
| NAVSIM Hands-On (CPU-only) | Verified recipe to install NAVSIM and compute real PDMS without a GPU; lightweight data (skip 151 GB sensors), custom single-stage script; CV 0.308 vs Human 0.914 on navmini | [navsim_hands_on.md](autonomous_driving/driving_benchmarks/navsim_hands_on.md) |

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

### Adding a new report with Claude Code

A project-level Claude skill automates the full workflow (md → ipynb → execute → README update → push).
In Claude Code, run:

```
/add-report
```

The skill walks through topic, category, dependency level, then handles file generation,
notebook execution, README updates, and git push. See `.claude/commands/add-report.md` for the full spec.
