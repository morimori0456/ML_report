#!/bin/bash
# Multi-node training inside an NGC container via Enroot + Pyxis (no Docker daemon).
# The --container-* flags are provided by the Pyxis SPANK plugin for srun.
# Submit with:  sbatch container_pyxis.sbatch
#SBATCH --job-name=train_ctr
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

IMAGE=nvcr.io/nvidia/pytorch:24.05-py3        # matched CUDA/cuDNN/NCCL from NGC
MOUNTS=/lustre/datasets:/data,/lustre/$USER/work:/work

# srun runs one container per node; torchrun inside spawns the GPU workers.
srun \
  --container-image="$IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/work \
  bash -lc '
    torchrun \
      --nnodes="$SLURM_NNODES" \
      --nproc_per_node=8 \
      --rdzv_id="$SLURM_JOB_ID" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
      train.py --config /work/configs/big.yaml --data /data
  '
