#!/bin/bash
# Hyperparameter sweep as a Slurm job array: 16 runs, at most 4 concurrent.
# Each array task picks one config line by its SLURM_ARRAY_TASK_ID.
# Submit with:  sbatch sweep_array.sbatch
#SBATCH --job-name=sweep
#SBATCH --partition=gpu
#SBATCH --array=0-15%4               # 16 tasks, 4 running at a time
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1            # 1 GPU per sweep point
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out   # %A=array job id, %a=array task id

set -euo pipefail
mkdir -p logs

# Build the grid (lr x weight_decay); pick this task's combo.
LRS=(1e-4 3e-4 1e-3 3e-3)
WDS=(0.0 0.01 0.05 0.1)
LR=${LRS[$(( SLURM_ARRAY_TASK_ID / 4 ))]}
WD=${WDS[$(( SLURM_ARRAY_TASK_ID % 4 ))]}

echo "task=$SLURM_ARRAY_TASK_ID  lr=$LR  wd=$WD"
torchrun --standalone --nproc_per_node=1 \
  train.py --lr "$LR" --weight-decay "$WD" \
           --run-name "sweep_lr${LR}_wd${WD}" \
           --output-dir "$NAVSIM_EXP_ROOT/sweep/$SLURM_ARRAY_TASK_ID"
