# Building ML Training Infrastructure — Tools & Know-How

> A practical map of what it takes to run multi-GPU / multi-node deep-learning training:
> the **scheduler** (Slurm — `srun`/`sbatch`), **storage** (NAS/NFS → parallel filesystems),
> the **interconnect** (InfiniBand/NCCL), **containers/environments**, the **distributed-launch**
> glue, **orchestration** (Kubernetes), and **monitoring**. Focus is on the decisions and
> pitfalls, not just the command list.

Runnable templates live in [examples/](examples/) (single-node, multi-node, container, sweep).

---

## Table of Contents
1. [The Layers of a Training Platform](#1-the-layers-of-a-training-platform)
2. [Slurm — the Scheduler (srun / sbatch / salloc)](#2-slurm--the-scheduler-srun--sbatch--salloc)
3. [Requesting GPUs, CPUs, Memory (GRES)](#3-requesting-gpus-cpus-memory-gres)
4. [Distributed Launch: srun × torchrun](#4-distributed-launch-srun--torchrun)
5. [The Interconnect: NCCL & InfiniBand](#5-the-interconnect-nccl--infiniband)
6. [Storage: NAS → NFS → Parallel FS → Object](#6-storage-nas--nfs--parallel-fs--object)
7. [The Data-Loading Bottleneck (know-how)](#7-the-data-loading-bottleneck-know-how)
8. [Containers & Environments](#8-containers--environments)
9. [Checkpointing, Preemption, Fault Tolerance](#9-checkpointing-preemption-fault-tolerance)
10. [Kubernetes — the Alternative Stack](#10-kubernetes--the-alternative-stack)
11. [Experiment Tracking & Cluster Monitoring](#11-experiment-tracking--cluster-monitoring)
12. [Provisioning a Cluster (bare-metal & cloud)](#12-provisioning-a-cluster-bare-metal--cloud)
13. [Command Cheat-Sheet](#13-command-cheat-sheet)
14. [Common Pitfalls](#14-common-pitfalls)

---

## 1. The Layers of a Training Platform

```
┌──────────────────────────────────────────────────────────────┐
│  Experiment / orchestration : W&B, MLflow, Hydra, sweeps      │
├──────────────────────────────────────────────────────────────┤
│  Scheduler                  : Slurm  (or Kubernetes + Volcano)│
│  Distributed launch         : torchrun / accelerate / deepspeed│
├──────────────────────────────────────────────────────────────┤
│  Runtime / env              : Enroot+Pyxis, Apptainer, conda  │
│  Collectives                : NCCL over InfiniBand/RoCE       │
├──────────────────────────────────────────────────────────────┤
│  Storage                    : Lustre/GPFS (datasets) + NVMe   │
│                               scratch + NFS (home) + S3 (cold)│
├──────────────────────────────────────────────────────────────┤
│  Hardware                   : GPU nodes, IB fabric, login node│
└──────────────────────────────────────────────────────────────┘
```

Two dominant paradigms:
- **HPC style** → **Slurm** + parallel filesystem + Enroot/Apptainer. Dominant in research labs
  and supercomputers. Batch-oriented, gang scheduling, very efficient for large training.
- **Cloud-native style** → **Kubernetes** + Kubeflow/Volcano + object storage. Better for serving,
  multi-tenant, autoscaling. §10.

Most AI research clusters (incl. NVIDIA DGX SuperPOD) are **Slurm-first**, so this guide leads with it.

---

## 2. Slurm — the Scheduler (srun / sbatch / salloc)

Slurm allocates nodes/GPUs to jobs and queues them by priority. Three entrypoints:

| Command | Use |
|---|---|
| **`sbatch script.sh`** | Submit a **batch** job (the normal way). Returns a job id, runs later. |
| **`srun ...`** | Run a command **inside** an allocation (launches tasks across nodes). Also runs interactively. |
| **`salloc`** | Grab an **interactive** allocation (a shell with nodes reserved) for debugging. |

Inspection / control:

| Command | Use |
|---|---|
| `squeue -u $USER` | your queued/running jobs |
| `sinfo` | partitions & node states (idle/alloc/down) |
| `scontrol show job <id>` | full job detail (why pending, node list) |
| `scancel <id>` | kill a job (`scancel -u $USER` kills all yours) |
| `sacct -j <id> --format=...` | accounting: elapsed, MaxRSS, state, exit code |
| `sprio`, `sshare` | priority / fair-share breakdown |

A minimal `sbatch` header (directives are `#SBATCH` comments):

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu          # which queue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0                  # 0 = all memory on the node
#SBATCH --time=24:00:00          # walltime limit (HH:MM:SS)
#SBATCH --output=logs/%x_%j.out  # %x=jobname %j=jobid
#SBATCH --requeue                # allow requeue on preemption
```

Key mental model:
- **The batch script runs only on the first node.** To run something on *all* allocated nodes you
  must use `srun` (it fans the command out to every task/node).
- `--ntasks` / `--ntasks-per-node` decide how many copies `srun` launches. For PyTorch you
  usually want **1 task per node** and let `torchrun` spawn the per-GPU workers (§4).

Useful extras: **job arrays** for sweeps (`#SBATCH --array=0-15%4` = 16 jobs, 4 at a time),
**dependencies** (`sbatch --dependency=afterok:<id> next.sh`), **QOS/partitions** for priority tiers.

---

## 3. Requesting GPUs, CPUs, Memory (GRES)

GPUs are a **GRES** (generic resource). Modern Slurm:

```bash
#SBATCH --gpus-per-node=8          # 8 GPUs per node
# or finer:
#SBATCH --gres=gpu:a100:8          # 8 GPUs of type a100
#SBATCH --gpus-per-task=1          # bind 1 GPU per task
```

Right-size the rest to the node:
- **`--cpus-per-task`**: dataloader workers live here. Rule of thumb **8–12 CPU cores per GPU**.
  Too few → GPUs starve waiting for `DataLoader`.
- **`--mem=0`** grabs all node RAM (needed for big dataset caching / pinned buffers).
- **`--exclusive`** reserves the whole node (avoid noisy neighbors for benchmarking).

Inside the job, Slurm sets `CUDA_VISIBLE_DEVICES` for you — **don't hard-code GPU ids**.

---

## 4. Distributed Launch: srun × torchrun

The canonical multi-node PyTorch pattern: **`srun` launches one `torchrun` per node**, and each
`torchrun` spawns one worker per local GPU using a rendezvous backend.

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # ONE torchrun per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=8 \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py --config configs/big.yaml
```

What each piece does:
- `MASTER_ADDR` = first node's hostname (the rendezvous host); all ranks dial in there.
- `--nproc_per_node=8` → 8 ranks/node × 4 nodes = **32 global ranks** (one per GPU).
- `c10d` rendezvous means **no separate etcd needed**; it also enables **elastic** restarts.
- In `train.py` you read `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from env (torchrun sets them) and call
  `torch.distributed.init_process_group("nccl")`, then `torch.cuda.set_device(LOCAL_RANK)`.

Two common alternatives:
- **`srun` launches every rank directly** (`--ntasks-per-node=8`, no torchrun) and you derive
  `RANK=$SLURM_PROCID`, `LOCAL_RANK=$SLURM_LOCALID`, `WORLD_SIZE=$SLURM_NTASKS`. Fewer moving parts,
  but you lose torchrun's elastic restart.
- **`accelerate`** / **`deepspeed`** / **`srun --mpi=pmix`** wrappers — same idea, different glue.

Parallelism strategies you'll be launching (the framework, not the scheduler):
- **DDP** (data parallel) — default; replicate model, all-reduce gradients.
- **FSDP / ZeRO** (sharded data parallel) — shard params/grads/optimizer states across GPUs to fit
  large models. (DeepSpeed ZeRO-1/2/3, PyTorch FSDP.)
- **TP / PP** (tensor / pipeline parallel) — split a single model across GPUs (Megatron-LM) when one
  layer/model doesn't fit. Usually combined ("3D parallelism") for LLMs.

---

## 5. The Interconnect: NCCL & InfiniBand

Gradient all-reduce traffic dominates multi-node training, so the **fabric** matters as much as the
GPUs. **NCCL** is NVIDIA's collective library; it auto-detects topology but you steer it with env vars.

| Fabric | What it is |
|---|---|
| **NVLink / NVSwitch** | intra-node GPU↔GPU (hundreds of GB/s) |
| **InfiniBand (IB)** | inter-node RDMA fabric (HDR/NDR 200–400 Gb/s); the HPC default |
| **RoCE** | RDMA over Ethernet — IB-like semantics on Ethernet |
| **GPUDirect RDMA** | NIC reads/writes GPU memory directly, bypassing the CPU |

NCCL knobs you will actually set:

```bash
export NCCL_DEBUG=INFO              # print the topology/rings it chose (first thing to check)
export NCCL_SOCKET_IFNAME=eth0      # which NIC for the bootstrap/control plane
export NCCL_IB_HCA=mlx5             # which IB HCAs to use
export NCCL_IB_DISABLE=0            # 1 to force TCP (debug only — slow)
export NCCL_P2P_LEVEL=NVL           # prefer NVLink for intra-node P2P
```

Know-how: if multi-node is *much* slower than single-node, 90% of the time it's NCCL **falling back
to TCP** because it couldn't pick the IB interface — `NCCL_DEBUG=INFO` will show
`via NET/Socket` instead of `via NET/IB`. Fix `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA`.

---

## 6. Storage: NAS → NFS → Parallel FS → Object

Different data, different store. A real cluster layers several:

| Tier | Tech | For |
|---|---|---|
| **Home / code** | **NFS** (NAS) | small, POSIX, convenient; **not** for training reads |
| **Datasets (hot)** | **Lustre, GPFS/Spectrum Scale, BeeGFS, WekaFS, VAST** | parallel, high-throughput reads from many nodes |
| **Local scratch** | node **NVMe** (`/scratch`, `/local`) | stage a shard here for fastest random reads |
| **Cold / archive** | **S3 / MinIO / object** | cheap, durable; stream or pre-stage |

Why a plain **NAS/NFS share will bottleneck training**: NFS is a single server with limited IOPS and
no client-side striping. Point 256 GPUs' dataloaders at one NFS export and you serialize on it — GPU
utilization tanks. Datasets belong on a **parallel filesystem** (data striped across many storage
servers, read in parallel) or **staged to local NVMe**.

Parallel FS in one line each:
- **Lustre** — the HPC workhorse; metadata server (MDS) + many object storage targets (OSTs); you
  *stripe* big files across OSTs for aggregate bandwidth.
- **GPFS / IBM Spectrum Scale** — similar, strong on metadata & enterprise features.
- **BeeGFS** — lighter to deploy, popular in mid-size GPU clusters.
- **WekaFS / VAST** — NVMe-first flash filesystems marketed for AI; very high small-file IOPS
  (great for many-small-files datasets like ImageNet/nuScenes crops).

---

## 7. The Data-Loading Bottleneck (know-how)

GPUs are fast; feeding them is the hard part. Symptoms: low/oscillating GPU utilization, `DataLoader`
workers pinned at 100% CPU, epoch time dominated by I/O.

Tactics, cheapest first:
1. **More `num_workers`** + `pin_memory=True` + `persistent_workers=True` + `prefetch_factor`.
2. **Kill small-file random I/O** — millions of tiny files murder any filesystem. Pack into
   **sharded archives**: **WebDataset** (`.tar` shards, sequential reads), **FFCV**, **MosaicML
   Streaming**, TFRecord, LMDB, or Parquet. Sequential reads are 10–100× friendlier than random.
3. **Stage to local NVMe** at job start (`rsync`/`cp` the shard the node needs to `/scratch`), then
   read locally. Best random-read latency.
4. **Stream from object store** (WebDataset/Mosaic Streaming pull `.tar` shards from S3) when the set
   is too big to stage — overlap download with compute.
5. **Decode on GPU** (NVIDIA **DALI**) to offload JPEG decode/augment off the CPU.
6. **Cache** the decoded/resized dataset once; reuse across epochs and jobs.

Rule of thumb: provision **8–12 CPU cores + a few GB/s of read bandwidth per GPU**. If a GPU needs
~1–2 GB/s of samples, 8 GPUs need ~10 GB/s — only a parallel FS or local NVMe delivers that.

---

## 8. Containers & Environments

Reproducibility = pin the whole stack (CUDA, cuDNN, NCCL, framework). Options:

| Tool | Where it fits |
|---|---|
| **Enroot + Pyxis** | the **Slurm-native** way: run OCI/Docker images as unprivileged Slurm tasks via `srun --container-image=...`. The DGX/NGC default. |
| **Apptainer (Singularity)** | rootless containers for HPC; `.sif` images; no daemon, runs as the user. |
| **Docker** | dev boxes & Kubernetes; usually **not** allowed directly on shared HPC (root daemon). |
| **conda / mamba / uv** | lightweight env without containers; fine for single-tenant boxes (see the NAVSIM recipe in this repo). |
| **Lmod `module load`** | HPC environment modules (`module load cuda/12.4 nccl`) to pick toolchains. |

Pyxis/Enroot in a Slurm job (no Docker daemon needed):

```bash
srun --container-image=nvcr.io/nvidia/pytorch:24.05-py3 \
     --container-mounts=/lustre/data:/data,/lustre/$USER:/work \
     --container-workdir=/work \
     python train.py
```

Pull base images from **NGC** (`nvcr.io/nvidia/pytorch:*`) — they ship matched CUDA/cuDNN/NCCL/Apex
and save days of dependency hell.

---

## 9. Checkpointing, Preemption, Fault Tolerance

At scale, **nodes fail and jobs get preempted** — design for it, don't hope against it.

- **Checkpoint frequently** (model + optimizer + scheduler + RNG + step) to the parallel FS; keep the
  last N. For big models use **sharded/distributed checkpoints** (each rank writes its shard;
  PyTorch DCP, DeepSpeed) so saving doesn't serialize through rank 0.
- **Requeue on preemption**: `#SBATCH --requeue`, trap `SIGTERM`/`SIGUSR1` (Slurm sends a warning
  signal `--signal=USR1@90` before the kill), flush a checkpoint, and resume from `latest` on restart.
- **Elastic / fault-tolerant training**: `torchrun` with `c10d` rendezvous + `--max-restarts` can
  survive worker failures and continue with the surviving nodes (torch elastic).
- **Idempotent resume**: always relaunch the *same* sbatch; the script detects `latest.ckpt` and
  continues. Time-limited jobs that auto-requeue let you train past any single walltime cap.

---

## 10. Kubernetes — the Alternative Stack

When you need multi-tenant, autoscaling, or to share infra with inference/serving, **Kubernetes**
replaces Slurm:

| Slurm world | Kubernetes world |
|---|---|
| `sbatch` job | `Job` / `PyTorchJob` (Training Operator) CRD |
| Slurm scheduler | **Volcano** / **Kueue** (gang scheduling — all pods or none) |
| Enroot/Pyxis | native container runtime |
| GRES `gpu` | **NVIDIA device plugin** + GPU Operator |
| Pipelines/sweeps | **Kubeflow** Pipelines / Katib |

Key gotcha: vanilla K8s scheduling is per-pod, which **deadlocks gang jobs** (some workers start,
others can't, all wait). You *must* add **gang scheduling** (Volcano or Kueue) for distributed
training. NCCL/IB on K8s needs the **GPU Operator** + **Network Operator** (RDMA/SR-IOV).
Rule of thumb: **Slurm for training-heavy research clusters, K8s when training shares a platform
with production serving.**

---

## 11. Experiment Tracking & Cluster Monitoring

Two different jobs: track *experiments*, and watch *hardware*.

**Experiments** — Weights & Biases, MLflow, TensorBoard, Neptune, Aim. Log loss/metrics/configs,
diff runs, store artifacts/checkpoints. Pair with **Hydra** for config sweeps + **W&B Sweeps** /
Slurm job arrays for the grid.

**Hardware / cluster health**:
- **`nvidia-smi`**, **`nvtop`**, **`dcgmi`** (NVIDIA **DCGM**) for live GPU util/mem/power/throttle.
- **DCGM-Exporter → Prometheus → Grafana** = the standard GPU-cluster dashboard (per-GPU SM util,
  memory, ECC errors, NVLink/IB traffic, power).
- **Node Exporter / Ganglia** for CPU/RAM/disk/network.
- Watch for: **low SM utilization** (data-loading bound), **thermal throttling** (`SM clocks`
  dropping), **ECC/Xid errors** (failing GPU), **IB error counters** (bad cable/port).

---

## 12. Provisioning a Cluster (bare-metal & cloud)

You rarely build Slurm by hand. Use a stack:

**Bare-metal / on-prem**
- **NVIDIA Base Command Manager (BCM, ex-Bright)** — provisions DGX/SuperPOD: OS imaging, Slurm,
  networking, monitoring. The "official" DGX path.
- **DeepOps** (NVIDIA, Ansible) — open-source playbooks to stand up Slurm **or** K8s on GPU nodes.
- **Determined AI**, **Run:ai** — schedulers/platforms layered on top for sharing & quotas.
- Ingredients you still own: a **provisioning/imaging** tool (Warewulf, MAAS, Foreman), a
  **config-management** tool (Ansible), the **IB fabric** (Subnet Manager), and the **storage**.

**Cloud**
- **AWS ParallelCluster** (managed Slurm), **SageMaker HyperPod** (resilient managed Slurm/K8s for
  large training), **Azure CycleCloud**, **GCP** (Slurm via Cluster Toolkit, or GKE).
- Cloud gives EFA (AWS's RDMA), elastic capacity, and S3 as the dataset tier — at a price premium.

**The minimal viable setups** (what most people actually start with):
- **1 node, N GPUs** → no scheduler needed: `torchrun --standalone --nproc_per_node=N train.py`.
- **A handful of nodes** → DeepOps/Slurm or just `pdsh` + `torchrun` with a static host list.
- **Shared team cluster** → Slurm + Enroot/Pyxis + a parallel FS (BeeGFS is the easiest to deploy).

---

## 13. Command Cheat-Sheet

```bash
# submit / interactive
sbatch train.sbatch                 # submit batch job
salloc -N1 --gpus=8 --time=2:00:00  # interactive allocation, then `srun --pty bash`
srun --jobid=<id> --pty bash        # shell on an existing allocation (to debug)

# inspect
squeue -u $USER                     # my jobs
squeue --start -j <id>              # estimated start time of a pending job
sinfo -N -l                         # node-level state
scontrol show job <id>              # why pending? node list, reason
sacct -j <id> --format=JobID,JobName,Elapsed,MaxRSS,State,ExitCode

# control
scancel <id>            ;  scancel -u $USER          # cancel one / all
scontrol hold <id>      ;  scontrol release <id>     # pause/resume in queue
scontrol requeue <id>                                 # requeue now

# GPU health
nvidia-smi ; nvtop ; dcgmi discovery -l ; dcgmi diag -r 1
```

---

## 14. Common Pitfalls

1. **Pointing dataloaders at NFS/NAS.** It serializes; GPUs starve. Datasets → parallel FS or local
   NVMe; pack small files into shards (§6–7).
2. **NCCL silently on TCP.** Multi-node crawls. `NCCL_DEBUG=INFO`; ensure it says `NET/IB`, fix
   `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA` (§5).
3. **Forgetting `srun`.** The batch script body runs **only on node 0**; without `srun` your "multi-node"
   job uses one node (§2).
4. **Too few `--cpus-per-task`.** Dataloader-bound; low GPU util. Budget 8–12 cores/GPU (§3).
5. **No checkpoint / no `--requeue`.** A preemption or node failure throws away a week of training (§9).
6. **Hard-coding GPU ids / `CUDA_VISIBLE_DEVICES`.** Slurm sets it; overriding breaks binding (§3).
7. **Gang-scheduling missing on K8s.** Distributed pods deadlock — add Volcano/Kueue (§10).
8. **Mismatched CUDA/NCCL across nodes.** Use one container image (NGC) everywhere (§8).
9. **Walltime too long / too short.** Too long → won't schedule; too short → killed mid-epoch. Use
   short jobs + auto-requeue + resume (§9).
10. **Saving checkpoints through rank 0 only.** I/O serializes and stalls all ranks; use distributed
    checkpointing (§9).

---

## References

- Slurm docs — https://slurm.schedmd.com/ (`sbatch`, `srun`, `salloc`, `sacct`, GRES)
- PyTorch Distributed / Elastic — https://pytorch.org/docs/stable/distributed.html , `torchrun`
- NCCL env vars — https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- Enroot / Pyxis — https://github.com/NVIDIA/enroot , https://github.com/NVIDIA/pyxis
- Apptainer — https://apptainer.org/
- DeepOps — https://github.com/NVIDIA/deepops ; Base Command Manager (DGX)
- Lustre — https://www.lustre.org/ ; BeeGFS — https://www.beegfs.io/
- WebDataset — https://github.com/webdataset/webdataset ; NVIDIA DALI — https://github.com/NVIDIA/DALI
- Kubeflow Training Operator — https://www.kubeflow.org/ ; Volcano — https://volcano.sh/
- DCGM / dcgm-exporter — https://github.com/NVIDIA/DCGM
- Related in this repo: [NAVSIM hands-on](../autonomous_driving/driving_benchmarks/navsim_hands_on.md)
  (single-node conda example), [mmEngine guide](../autonomous_driving/mmengine/mmengine_guide.md)
  (DDP/AMP via Runner config).
