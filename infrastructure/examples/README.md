# Slurm job templates

Ready-to-adapt `sbatch` scripts referenced by
[../ml_training_infrastructure.md](../ml_training_infrastructure.md). They assume a `train.py`
that reads `RANK` / `LOCAL_RANK` / `WORLD_SIZE` from the environment and calls
`torch.distributed.init_process_group("nccl")`.

| File | What it shows |
|---|---|
| [single_node.sbatch](single_node.sbatch) | 1 node × 8 GPUs, `torchrun --standalone` DDP |
| [multi_node.sbatch](multi_node.sbatch) | 4 nodes, `srun` × `torchrun` c10d rendezvous, NCCL/IB env |
| [container_pyxis.sbatch](container_pyxis.sbatch) | same, inside an NGC container via Enroot+Pyxis |
| [sweep_array.sbatch](sweep_array.sbatch) | hyperparameter sweep as a job array (`--array=0-15%4`) |

Adjust `--partition`, `--gpus-per-node`, `--cpus-per-task`, `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA`,
and the container image/mounts to your cluster. These are templates, not runnable as-is (they need
your `train.py` and a Slurm cluster).
