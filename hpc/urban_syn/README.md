# URBAN-SYN Container Environment

Docker provides the same Python environment for local runs and HPC. On HPC, we run the Docker image through **Apptainer** and use **Slurm** to request GPUs. Model code and datasets stay in the repository, not in the image.

| File | Purpose |
|---|---|
| `Dockerfile`, `compose.yaml` | Build and run the environment |
| `submit.sh` | Choose an experiment and submit Slurm jobs |
| `run_urban_syn.sbatch` | Run Python inside Apptainer and collect job logs |
| `resolve_submission.py`, `study_checkpoints.py` | Resolve study settings and pretrained checkpoints |
| [Slurm guide](README_SLURM.md) | Submitting experiments and troubleshooting |
| [MLflow guide](README_MLFLOW.md) | Tracking runs and opening results locally |

The container installs packages under `/opt/venv` and mounts the repository at `/workspace`. The VS Code Dev Container is a separate development environment.

## 1. Build the image locally

From the repository root on your **host computer** (not inside VS Code's Dev Container):

```bash
docker build -f hpc/urban_syn/Dockerfile -t urban-syn-joint:cu118 .
```

Or use `docker compose -f hpc/urban_syn/compose.yaml build`. The image installs the dependencies from `uv.lock`; rebuild it when those dependencies change. Routine Python/YAML edits don't require rebuilding because the repository is mounted at runtime.

To check GPU access locally:

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm joint \
  python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Use the image locally

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm joint \
  python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml --validate-only
```

Remove `--validate-only` to execute training and (if enabled) generation. The separate link-pretraining entry point is `python -m urban_syn.link_pretrain`. See the [joint pipeline](../../src/urban_syn/experiment/README.md) and [pretraining guide](../../src/urban_syn/link_pretrain/README.md).

## 3. Create an HPC Apptainer image

On your local computer, export the built image and upload it to HPC:

```bash
mkdir -p container_exports
docker save -o container_exports/urban-syn-hpc.tar urban-syn-joint:cu118
scp container_exports/urban-syn-hpc.tar USER@HPC_HOST:~/containers/
```

Create `~/containers/` on HPC first if necessary. On HPC, request a **CPU compute node** (not the login node) for conversion:

```bash
srun --partition=defq --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=8G --time=01:00:00 --pty bash

apptainer build "$HOME/containers/urban-syn.sif" \
  "docker-archive:$HOME/containers/urban-syn-hpc.tar"
```

Check the result:

```bash
apptainer exec "$HOME/containers/urban-syn.sif" \
  /opt/venv/bin/python -c "import torch; print(torch.__version__)"
```

For a CUDA check, obtain a Slurm GPU allocation first and run `apptainer exec --nv ...`; don't test GPU workloads on a login node.

## 4. Run on HPC

From the repository root on HPC:

```bash
bash hpc/urban_syn/submit.sh \
  --study configs/experiments/hpc_smoke_test/study.yaml \
  --experiment joint_frozen --stage all
```

The launcher mounts the checkout and runs the same Python code used locally. Prepared road sequences, the road GeoPackage, and any needed generation inputs must exist at the YAML paths. The PyG cache can be created automatically when missing.

For **resource overrides, dependent pretraining jobs, output files, and monitoring**, continue with [README_SLURM.md](README_SLURM.md).
