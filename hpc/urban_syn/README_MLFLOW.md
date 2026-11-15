# MLflow Tracking and Viewing HPC Experiments

This guide explains how URBAN-SYN records experiments, how to inspect training results without MLflow, and how to open **HPC MLflow runs on a local computer**. It is written for teammates using the joint GNN + Diff-RNTraj experiment pipeline. Keep this file alongside the [container guide](README.md) and [Slurm guide](README_SLURM.md) under `hpc/urban_syn/`.

MLflow is **optional**. The pipeline always saves essential run files separately. MLflow adds an organized interface for comparing runs, inspecting parameters and metrics, and browsing selected artifacts.

## 1. Where experiment results are saved

All paths below are relative to the URBAN-SYN repository root unless otherwise noted. In our Docker and Apptainer containers, that root is `/workspace`.

```text
outputs/experiments/
├── <experiment_name>/
│   └── <run_id>/
│       ├── resolved_config.json       # Effective settings for the run
│       ├── data_split.json            # Train/validation split information
│       ├── checkpoints/
│       │   ├── best.pt
│       │   ├── last.pt
│       │   └── epoch_*.pt             # When enabled by the YAML
│       ├── metrics/
│       │   └── history.jsonl          # Training metrics, one JSON record per epoch
│       ├── logs/
│       │   └── train.log
│       └── generated/
│           ├── <generated_file>.parquet
│           └── generation_manifest.json
└── mlflow/
    ├── tracking.db                    # Shared SQLite tracking database
    └── artifacts/                    # Files logged as MLflow artifacts
```

**Without MLflow**, you can still read `metrics/history.jsonl`, inspect `logs/train.log`, use checkpoints, and access generated Parquet files. The exact files present depend on which pipeline stages and checkpoint settings were enabled. Disable MLflow in the experiment YAML with:

```yaml
tracking:
  enabled: false
```

Slurm also writes job and resource logs under `hpc/urban_syn/logs/`, including `urban-syn-<job_id>.out`, `urban-syn-<job_id>.err`, GPU monitoring CSVs, and resource summaries. These are separate from MLflow.

## 2. Configure MLflow in the experiment YAML

```yaml
tracking:
  enabled: true
  provider: mlflow
  experiment_name: gnn_diff_rntraj_nyc
  run_name: null                 # Optional readable name
  tracking_uri: null             # Use the pipeline's local SQLite backend
  storage_scope: shared          # shared or job
  logging:
    metrics_every_steps: 1
    gnn_diagnostics_every_steps: 1
    gnn_embedding_probe_size: 128
    log_system_metrics: false
    log_resolved_config: true
    log_git_info: true
  artifacts:
    log_best_checkpoint: true
    log_last_checkpoint: false
    log_generated_trajectories: true
    log_training_logs: true
```

This illustrates the tracking settings. Keep any other tracking fields and tags from your experiment YAML as needed.

| Setting | Meaning |
|---|---|
| `enabled` | Turn MLflow tracking on or off. Local run files are still written when off. |
| `experiment_name` | Groups related MLflow runs in the UI. |
| `run_name` | Optional display name. Reusing a name does not overwrite an independent run. |
| `tracking_uri` | Explicit MLflow backend location, if needed. When set, it overrides automatic storage paths. |
| `storage_scope: shared` | One SQLite database for sequential HPC runs, convenient for comparing them in one UI. |
| `storage_scope: job` | A separate SQLite database for each Slurm job, useful for avoiding concurrent SQLite writes. |
| `logging` | Controls metric frequency, GNN diagnostics, and configuration/Git metadata. |
| `artifacts` | Selects which checkpoints, generated files, and logs are additionally copied into MLflow. |

With `storage_scope: shared`, the database is usually:

```text
outputs/experiments/mlflow/tracking.db
```

With `storage_scope: job`, Slurm jobs use directories such as:

```text
outputs/experiments/mlflow/jobs/<slurm_job_id>/tracking.db
```

Array jobs may include the array task ID in the directory name. Local runs without a Slurm job ID use the non-job-specific location. `MLFLOW_TRACKING_URI`, when set and not overridden by YAML, can also select a different backend.

**Recommended use:** Shared SQLite works well for sequential jobs and direct comparisons. Avoid concurrent writers to one SQLite file on networked HPC storage; use `job` scope when running multiple jobs at once. Separate databases do not automatically appear together in one MLflow UI. A centralized tracking service or a separate import workflow would be needed to combine them.

You **do not need an MLflow server running on HPC to record experiments**. The training process writes directly to its configured backend.

## 3. Download an HPC MLflow snapshot

The commands below run on **your local computer**, not on the HPC login node. Use a terminal in your local URBAN-SYN repository root.

First, on HPC, identify the repository's absolute location using `pwd`. Then substitute that path and your SSH username into the command below. The same `scp` command works in **macOS Terminal** and **Windows 11 Git Bash**:

```bash
mkdir -p ./hpc_mlflow_snapshot

scp -r \
  YOUR_USERNAME@coe-hpc3.sjsu.edu:/absolute/path/to/URBAN-SYN/outputs/experiments/mlflow \
  ./hpc_mlflow_snapshot/
```

The local snapshot should look like:

```text
hpc_mlflow_snapshot/
└── mlflow/
    ├── tracking.db
    └── artifacts/
```

**Copy the entire `mlflow/` directory**, not only `tracking.db`. The database contains run information and metrics; the `artifacts/` directory contains logged files referenced by the database.

For a `job`-scoped experiment, copy the appropriate `outputs/experiments/mlflow/jobs/<job_id>/` directory instead and point the UI to that particular database. This guide's commands below use the shared layout.

Prefer copying after HPC jobs have completed. Copying a live SQLite database while it is being written can produce an inconsistent snapshot. Keep downloaded data in `hpc_mlflow_snapshot/` instead of overwriting your local `outputs/experiments/mlflow/` files.

### When new HPC experiments finish

If your runs use `storage_scope: shared`, new runs are added to the **same HPC database**. You do not need a new local snapshot directory for each experiment. Stop the local MLflow UI, refresh the downloaded `mlflow/` directory, and restart the UI.

The HPC database is the source of truth for these runs. Replacing a downloaded snapshot with a newer complete snapshot from the same HPC backend does not erase the older runs from that backend. However, it does **not merge** the HPC database with your local experiments. Keep separate local and HPC databases unless you deliberately set up an import or centralized tracking workflow.

## 4. Open the HPC snapshot in MLflow UI

For HPC runs, we recommend using the project's Docker Compose image for viewing. The experiment was created inside the container at `/workspace`, and MLflow's artifact locations may contain absolute `file:///workspace/...` paths. Mounting the snapshot at the same path makes those links work.

Make sure Docker Desktop is running. The commands below assume the Compose service is named `joint`, as in `hpc/urban_syn/compose.yaml`.

### macOS (Terminal or zsh)

From the local URBAN-SYN repository root:

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm \
  -p 5001:5000 \
  -v "$(pwd)/hpc_mlflow_snapshot/mlflow:/workspace/outputs/experiments/mlflow" \
  joint \
  mlflow ui \
  --backend-store-uri sqlite:////workspace/outputs/experiments/mlflow/tracking.db \
  --host 0.0.0.0 \
  --port 5000
```

### Windows 11 (Git Bash / MSYS)

Git Bash can rewrite Linux-style Docker paths into Windows paths. Disable that conversion for this command, and give Docker an absolute Windows host path:

```bash
MSYS_NO_PATHCONV=1 docker compose -f hpc/urban_syn/compose.yaml run --rm \
  -p 5001:5000 \
  -v "$(pwd -W)/hpc_mlflow_snapshot/mlflow:/workspace/outputs/experiments/mlflow" \
  joint \
  mlflow ui \
  --backend-store-uri sqlite:////workspace/outputs/experiments/mlflow/tracking.db \
  --host 0.0.0.0 \
  --port 5000
```

Both commands expose the HPC MLflow UI at **http://localhost:5001**. Port `5001` is used to avoid clashing with any existing local MLflow UI on port `5000`. Stop the UI with `Ctrl+C` when finished.

**Note:** The Docker Compose command uses the local image and project checkout. It does not run training, connect to the HPC GPU, or require the original HPC Slurm job to stay alive.

## 5. Verify the snapshot if the wrong runs appear

You should see the HPC experiment name from your YAML, such as `gnn_diff_rntraj_nyc`. If you instead see only your earlier local experiments, confirm that Docker opened the downloaded database.

**Windows 11 Git Bash:**

```bash
MSYS_NO_PATHCONV=1 docker compose -f hpc/urban_syn/compose.yaml run --rm \
  -v "$(pwd -W)/hpc_mlflow_snapshot/mlflow:/workspace/outputs/experiments/mlflow" \
  joint \
  python -c "import sqlite3; db='/workspace/outputs/experiments/mlflow/tracking.db'; print(sqlite3.connect(db).execute('SELECT name FROM experiments').fetchall())"
```

**macOS:**

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm \
  -v "$(pwd)/hpc_mlflow_snapshot/mlflow:/workspace/outputs/experiments/mlflow" \
  joint \
  python -c "import sqlite3; db='/workspace/outputs/experiments/mlflow/tracking.db'; print(sqlite3.connect(db).execute('SELECT name FROM experiments').fetchall())"
```

If the names are wrong, verify the snapshot's `tracking.db` is the file copied from HPC, and confirm the mount destination is exactly `/workspace/outputs/experiments/mlflow`. On Windows Git Bash, omit neither `MSYS_NO_PATHCONV=1` nor `$(pwd -W)`.

If experiment names and metrics appear but **artifacts fail to open**, check that you copied the entire artifacts directory and preserved the container path above. Some older runs may also reference separate paths outside the MLflow artifact directory; download and mount those original run files if necessary.

## 6. What to look at in MLflow

Inside an experiment, open a run to inspect:

- **Parameters:** The resolved model, training, generation, and runtime settings used for the run.
- **Metrics:** Loss components, gradient-bearing objective, GNN and denoiser gradient norms, embedding diagnostics, and other enabled metrics.
- **Tags:** Context such as the experiment configuration, Git revision, and Slurm job ID when available.
- **Artifacts:** Selected files such as resolved configuration, saved checkpoints, logs, and generated trajectories, depending on the YAML artifact toggles.

For comparing successive research runs, keep their experiment names consistent in YAML and give runs descriptive names or tags. Each independent training execution creates a distinct run. Use `training.checkpoint.resume_from` only when you intend to continue an existing checkpoint rather than start a new experiment.

For training and graph preparation, see the [experiment README](../../src/urban_syn/experiment/README.md). For Slurm jobs, see [README_SLURM.md](README_SLURM.md).
