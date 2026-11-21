# MLflow: View and Compare Experiments

MLflow stores experiment settings, losses, resource measurements, and selected artifacts. **You do not need to run an MLflow server on HPC while training.** The training jobs write to the configured backend; the UI is for inspecting results later.

Model files are also saved under `outputs/experiments/<experiment_name>/<run_id>/` when MLflow is disabled.

## 1. Where results are saved

A joint run typically contains:

```text
outputs/experiments/
  <experiment_name>/<run_id>/
    checkpoints/              # Joint best.pt, optional last.pt
    metrics/                  # Training history
    logs/
    generated/                # GPS Parquet, if enabled
  mlflow/
    tracking.db               # Runs, metrics, parameters, tags
    artifacts/                # Files copied into MLflow
```

Road-link pretraining uses its own run directory and exports `checkpoints/encoder.pt`. Pretraining and joint training have **separate MLflow runs**; a shared `tracking.experiment_name` groups related runs for comparison in the UI.

## 2. Configure tracking

In the experiment YAML:

```yaml
tracking:
  enabled: true
  provider: mlflow
  experiment_name: urban_syn_comparison
  tracking_uri: null
  storage_scope: shared
  logging:
    log_system_metrics: true
  artifacts:
    log_best_checkpoint: true
    log_generated_trajectories: true
```

Use the existing YAML's other logging/artifact settings as needed.

| Setting | Meaning |
|---|---|
| `experiment_name` | Groups runs in MLflow; use the same name for comparable experiments |
| `storage_scope: shared` | One SQLite database at `outputs/experiments/mlflow/tracking.db` |
| `storage_scope: job` | Separate database per Slurm job at `mlflow/jobs/<job_id>/tracking.db` |
| `logging`, `artifacts` | Control which measurements and files are logged |

Shared SQLite is convenient for sequential runs. For concurrent jobs on shared storage, consider job-scoped tracking to avoid database locking; **job-scoped databases do not automatically appear together** in one MLflow UI. An explicit `tracking_uri` overrides the automatic layout.

## 3. Download HPC runs to a local computer

Wait until the HPC jobs finish, then run this on your **local computer** from the repository root:

```bash
mkdir -p hpc_mlflow_snapshot
scp -r USER@HPC_HOST:/absolute/path/to/URBAN-SYN/outputs/experiments/mlflow \
  ./hpc_mlflow_snapshot/
```

Use `pwd` on HPC to find the actual repository path. Copy the **whole `mlflow/` directory**, not only `tracking.db`, so artifacts are available. Keep it separate from any local training outputs.

## 4. Open MLflow locally

The HPC runs were recorded with container paths such as `/workspace/outputs/experiments/mlflow`. Running the UI through the project Docker image preserves those artifact paths.

**macOS / Linux (Terminal):**

```bash
docker compose -f hpc/urban_syn/compose.yaml run --rm \
  -p 5001:5000 \
  -v "$(pwd)/hpc_mlflow_snapshot/mlflow:/workspace/outputs/experiments/mlflow" \
  joint mlflow ui \
  --backend-store-uri sqlite:////workspace/outputs/experiments/mlflow/tracking.db \
  --host 0.0.0.0 --port 5000
```

**Windows (Git Bash):**

```bash
MSYS_NO_PATHCONV=1 docker compose -f hpc/urban_syn/compose.yaml run --rm \
  -p 5001:5000 \
  -v "$(pwd -W)/hpc_mlflow_snapshot/mlflow:/workspace/outputs/experiments/mlflow" \
  joint mlflow ui \
  --backend-store-uri sqlite:////workspace/outputs/experiments/mlflow/tracking.db \
  --host 0.0.0.0 --port 5000
```

Open **http://localhost:5001**. Press `Ctrl+C` to stop the server. When new HPC jobs finish, stop the UI, refresh your downloaded `mlflow/` snapshot, and reopen it.

## 5. What to inspect

| MLflow tab | Useful for |
|---|---|
| Parameters | Confirm architecture, seed, optimizer, generation settings |
| Metrics | Compare training/validation loss, GNN diagnostics, and system metrics |
| Tags | Distinguish scratch, frozen, and fine-tuned runs; find Slurm job IDs |
| Artifacts | Inspect logs, saved configuration, checkpoints, and generated files (when logged) |

If runs are missing, confirm you're opening the **downloaded HPC** `tracking.db`, not your local database. If metrics appear but artifacts are inaccessible, check that `artifacts/` was copied and mounted at the path above. MLflow tracks model execution; synthetic-trajectory quality is assessed by the project's **separate shared evaluation pipeline**.
