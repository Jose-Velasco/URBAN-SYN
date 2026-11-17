# URBAN-SYN Container Environment

This directory contains the **Docker environment used for local execution and HPC deployment**. It supports the URBAN-SYN model as it evolves, not just the current GNN + Diff-RNTraj implementation.

The model itself lives in `src/urban_syn/`; Docker and Apptainer only provide its Python and GPU environment. The VS Code `.devcontainer/` remains separate and is not changed by these files.

For model options and pipeline stages, see the [experiment README](../../src/urban_syn/experiment/README.md). For batch jobs, see [README_SLURM.md](README_SLURM.md). For metrics and viewing HPC runs locally, see [README_MLFLOW.md](README_MLFLOW.md).

## What is in this directory?

| File | Purpose |
|---|---|
| `Dockerfile` | Creates a Python 3.11 environment with dependencies from the project's `uv.lock` |
| `compose.yaml` | Runs the image locally with GPU access and the repository mounted at `/workspace` |
| `submit.sh` | Submits a single experiment stage to Slurm on HPC |
| `run_urban_syn.sbatch` | Starts the Apptainer image on a GPU node, runs the pipeline, and records resource logs |
| `README_SLURM.md` | HPC-specific setup, job submission, tracking, and troubleshooting |
| `README_MLFLOW.md` | Tracking choices, saved metrics, download/snapshot instructions, and browser UI |
| `../../.dockerignore` | Excludes datasets, outputs, and local environments from the image build context |

The image uses **`/opt/venv`** for installed packages and **`/workspace`** for the mounted repository. This keeps Python dependencies out of the mounted local `.venv` and avoids installing packages each time a job starts.

The Docker image does **not** contain the NYC dataset, GeoPackage, generated graph cache, checkpoints, or experiment outputs. These remain on the host and are accessible through the repository mount.

## 1. Build on your local machine

Run the following commands from the **URBAN-SYN repository root** on the host, not from inside the VS Code Dev Container:

```bash
docker build \
  -f hpc/urban_syn/Dockerfile \
  -t urban-syn-joint:cu118 .
```

Or use Compose:

```bash
docker compose -f hpc/urban_syn/compose.yaml build
```

The current project uses CUDA 11.8 PyTorch wheels selected by `pyproject.toml` and `uv.lock`. The Docker build installs locked dependencies once. If dependencies change, update the lockfile and rebuild the image. Changes only to Python source or YAML generally do not require rebuilding, because the repository is mounted at runtime.

## 2. Run locally with Docker Compose

Check that the image can see your GPU:

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm joint \
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Inspect the selected pipeline stages:

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm joint \
  python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml --dry-run
```

Check configured inputs:

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm joint \
  python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml --validate-only
```

Execute the pipeline:

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm joint \
  python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml
```

You can replace `all` with `prepare-graph`, `train`, `generate`, or another supported target. All experiment settings remain in the YAML, including MLflow storage and checkpoint choice. We invoke the image's `python` directly, rather than running `uv sync` or `uv run` inside the container.

**Note for Windows and Dev Containers:** Docker bind mounts are resolved by the Docker daemon. Run these Compose commands from the host repository directory if a nested container would point the mount at the wrong location.

## 3. Export the image and upload it to HPC

We use the same approach as our previous baselines: **save the Docker image as an uncompressed TAR locally, upload it, then convert it to a SIF on a CPU compute node**. The TAR holds the image, not a snapshot of a running container.

On your local computer:

```bash
mkdir -p ./container_exports

docker save -o ./container_exports/urban-syn-hpc.tar \
  urban-syn-joint:cu118
```

Upload it to your HPC account, replacing the hostname and user as necessary. Create `~/containers/` on HPC beforehand if it does not already exist:

```bash
scp ./container_exports/urban-syn-hpc.tar \
  USER@HPC_HOST:~/containers/
```

On the HPC login node, request a CPU allocation for image conversion. Do not perform the build on the login node:

```bash
srun \
  --partition=defq \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=01:00:00 \
  --pty bash
```

Inside the allocated CPU node:

```bash
apptainer build \
  "$HOME/containers/urban-syn.sif" \
  "docker-archive:$HOME/containers/urban-syn-hpc.tar"
```

Check Python and PyTorch after the build:

```bash
apptainer exec "$HOME/containers/urban-syn.sif" \
  /opt/venv/bin/python -c "import torch; print(torch.__version__)"
```

The image can now be used for GPU jobs by passing `--nv` at execution time. For an interactive GPU check, first allocate a GPU node through Slurm, then use:

```bash
apptainer exec --nv "$HOME/containers/urban-syn.sif" \
  /opt/venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

Exit the temporary allocation when finished. Keep the TAR until the SIF is verified; afterward it can be removed to save space. Rebuilding the SIF is only needed when the image contents, such as installed dependencies, change.

## 4. Run the project on HPC

The Slurm runner bind-mounts the HPC repository checkout into `/workspace`, exposes the GPU with Apptainer `--nv`, and runs the same `urban_syn.experiment` commands used locally.

```bash
bash hpc/urban_syn/submit.sh all \
  configs/experiments/nyc_joint.yaml
```

Before the job starts, the HPC checkout must contain the necessary **input artifacts**, including the prepared road-sequence Parquet, source road GeoPackage, generation length distribution, and training FMM matches if the graph's transition-count feature needs them. The processed `road_graph.pt` and `preprocessor.joblib` do **not** have to be uploaded: the `prepare-graph` stage builds them when missing.

For the train/test split and data conversion procedure, see the [experiment README](../../src/urban_syn/experiment/README.md). For resource overrides, generation-only runs, and job logs, see [README_SLURM.md](README_SLURM.md). To inspect experiments on your desktop, see [README_MLFLOW.md](README_MLFLOW.md).

## Design principles

- **One environment for the model:** future GNN or diffusion changes reuse this image as long as the dependencies remain compatible.
- **One experiment runner:** Docker and Slurm invoke the same Python pipeline; there is no second trainer.
- **Reproducible packages:** locked dependencies live inside the image, not in the mounted project `.venv`.
- **Persistent outputs:** checkpoints, generated trajectories, and MLflow records are written outside the immutable SIF.
- **Separate development tools:** VS Code's Dev Container continues to work independently.
