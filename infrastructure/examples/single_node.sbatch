#!/bin/bash
# Single-node, multi-GPU DDP training via torchrun.
# Submit with:  sbatch single_node.sbatch
#SBATCH --job-name=train1n
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1          # one torchrun; it spawns the per-GPU workers
#SBATCH --cpus-per-task=96           # ~12 CPU cores per GPU for the dataloaders
#SBATCH --mem=0                      # all RAM on the node
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@90             # warn the job 90s before preemption/timeout
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

# torchrun --standalone handles rendezvous on a single node (no MASTER_ADDR needed)
torchrun \
  --standalone \
  --nproc_per_node=8 \
  train.py --config configs/base.yaml --output-dir "$NAVSIM_EXP_ROOT/$SLURM_JOB_ID"
