#!/bin/bash
# Multi-node DDP/FSDP training: srun launches one torchrun per node,
# each torchrun spawns one worker per local GPU (c10d rendezvous = elastic-capable).
# Submit with:  sbatch multi_node.sbatch
#SBATCH --job-name=train_mn
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # ONE task (torchrun) per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

# --- rendezvous host = first node in the allocation ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

# --- NCCL / InfiniBand: verify it picks NET/IB (not NET/Socket) in the logs ---
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0       # control-plane NIC (adjust to your cluster)
export NCCL_IB_HCA=mlx5              # InfiniBand HCAs (adjust / remove if RoCE)
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 8))

GPUS_PER_NODE=8
srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --max-restarts=3 \
  train.py --config configs/big.yaml --resume auto
