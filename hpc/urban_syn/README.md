# Joint GNN + Diff-RNTraj — Docker environment

This directory defines the **execution environment** for the existing `urban_syn.experiment` pipeline. It does not duplicate model code or create a second training pipeline. VS Code's `.devcontainer/` remains independent and unchanged.

## Layout

- `Dockerfile`: Python 3.11 and third-party packages from the repository's `uv.lock` (including its PyTorch CUDA 11.8 wheels).
- `compose.yaml`: local GPU execution, with the URBAN-SYN repository bind-mounted at `/workspace`.
- `../../.dockerignore`: excludes local data, checkpoints, secrets, and caches from the Docker build context.

The container stores Python dependencies in **`/opt/venv`**, outside the source mount. Python imports the mounted code from **`/workspace/src`**. The Docker image does **not** contain your datasets, processed graphs, checkpoints, or working source tree.

> **Requirements:** Docker with GPU support, Linux/amd64 container support, and a committed/current `uv.lock` matching `pyproject.toml`. The host NVIDIA driver must support the CUDA version in the locked PyTorch wheels. The lockfile is supplied by your repository; it is not included in this Stage 4 file bundle.

## 1. Build locally

Run these commands **from the URBAN-SYN repository root**:

```bash
docker build \
  -f hpc/urban_syn/Dockerfile \
  -t urban-syn-joint:cu118 .
```

Or build through Compose:

```bash
docker compose -f hpc/urban_syn/compose.yaml build
```

The build installs dependencies once using `uv sync --locked --no-dev --no-install-project`. A stale `uv.lock` causes the build to fail rather than silently resolving new versions. When you update dependencies, update the lockfile locally and rebuild the image.

## 2. Verify the container locally

Run the GPU and library check:

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm joint \
  python -c "import torch, torch_geometric; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU ready:', torch.cuda.is_available()); assert torch.cuda.is_available()"
```

Verify pipeline inputs (requires the configured local files):

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm joint \
  python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml --validate-only
```

Run your selected experiment:

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm joint \
  python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml
```

Other supported targets are `validate`, `train`, `generate`, and `export-eval`. Experiment settings, MLflow storage, and checkpoint behavior are controlled by the **existing experiment YAML**, not Compose. Container commands intentionally call `python` directly rather than `uv run`, to avoid syncing your mounted development `.venv` or mutating the image's locked environment.

On Linux, the Compose service defaults to UID/GID `1000:1000` to avoid root-owned host output files. If needed, set `LOCAL_UID` and `LOCAL_GID` to your host user/group IDs. Compose GPU access requires the NVIDIA Container Toolkit (or equivalent Docker GPU support).

## 3. Local image → HPC → Apptainer

Once local checks work, export the built image to a **directory outside the repository**:

```bash
mkdir -p ../container_exports
docker save urban-syn-joint:cu118 | gzip -1 > ../container_exports/urban-syn-joint-cu118.tar.gz
```

Upload that archive to HPC with your usual file-transfer tool. On HPC, where Apptainer is available, convert the uploaded archive:

```bash
apptainer build urban-syn-joint-cu118.sif \
  docker-archive:urban-syn-joint-cu118.tar.gz
```

Apptainer accepts gzip-compressed Docker archives. Ensure the conversion directory and `APPTAINER_TMPDIR` have sufficient free space; this image includes large PyTorch wheels. If you use another location for the archive, use its absolute path after `docker-archive:`.

On a **GPU compute node**, a basic check will look like:

```bash
apptainer exec --nv --cleanenv \
  --bind /path/to/URBAN-SYN:/workspace \
  urban-syn-joint-cu118.sif \
  python -c "import torch; print('GPU ready:', torch.cuda.is_available())"
```

Use your actual mounted repository path. The final GPU execution, resource requests, persistent storage paths, and monitoring will be handled in the **next Stage 4 step (Slurm runner)**. A successful local Docker check does not yet establish HPC driver compatibility.

## Design notes

- **No source/data duplication:** the same YAML and Python stages work in local Docker and HPC Apptainer.
- **No runtime package installs:** Slurm jobs run the image's installed `/opt/venv/bin/python`.
- **No new trainer:** container and job scripts invoke `python -m urban_syn.experiment <target> --config <yaml>`.
- **CUDA:** the project specifies PyTorch `cu118` wheels. The base is a small Python image rather than a second CUDA runtime image; NVIDIA's host driver is exposed through Docker GPU support / Apptainer `--nv`.
- **Persistence:** experiment outputs and MLflow data are written to paths in the bind-mounted repository according to the YAML.
