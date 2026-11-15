# URBAN-SYN — Slurm + Apptainer

Run the existing `urban_syn.experiment` pipeline on SJSU CoE HPC. The Slurm scripts allocate resources, start the Apptainer image, collect system logs, and invoke the **same Python CLI and experiment YAML** used locally. They do not implement a second trainer or generator.

## Prerequisites

- A working SIF at `~/containers/urban-syn.sif` (or set `URBAN_SYN_SIF` to its absolute path).
- A checkout of URBAN-SYN on HPC containing the experiment YAML and required **non-Git** inputs: prepared road-sequence Parquet, road-network GeoPackage, PyG graph cache, and generation length distribution.
- An available A100 GPU. Verify current partitions and limits with `sinfo` and `scontrol show partition gpuqm`.

The runner mounts the project checkout at `/workspace` and uses `/opt/venv/bin/python` inside the image. It does not run `uv sync`, use the local `.venv`, or modify the model code.

## Run a stage

From the HPC checkout root:

```bash
# Train, then generate if generation.enabled: true in YAML.
bash hpc/urban_syn/submit.sh all configs/experiments/nyc_joint.yaml

# Train only.
bash hpc/urban_syn/submit.sh train configs/experiments/nyc_joint.yaml

# Generate only from an existing trained checkpoint (see below).
bash hpc/urban_syn/submit.sh generate configs/experiments/nyc_joint.yaml

# Inspect the planned stages and verify inputs without running the model.
bash hpc/urban_syn/submit.sh all configs/experiments/nyc_joint.yaml --validate-only
```

Other supported stages are `validate` and `export-eval`; `--dry-run` is also available. Each submission is one Slurm job. `all` preserves the current run's selected training checkpoint for generation automatically.

### Generation without retraining

Set `generation.checkpoint` in the selected experiment YAML to a **real existing checkpoint** on HPC, relative to the project root:

```yaml
generation:
  checkpoint: outputs/experiments/nyc_joint_gnn_v1/<run_id>/checkpoints/best.pt
  checkpoint_choice: best
  num_trajectories: 100
  sampler:
    name: ddpm
```

Replace `<run_id>` with the specific training run directory. Also configure `paths.road_network` and `paths.length_distribution`. Then run the standalone `generate` command above. **`generation.enabled` controls whether `all` includes generation; it does not disable the explicitly selected `generate` stage.** Generation writes into the selected checkpoint's run directory under `generated/`, so use a distinct `generation.output_name` if you want to preserve an earlier generation output from that same checkpoint.

## MLflow: choose the store in the experiment YAML

Use the same centralized experiment YAML for both the model and experiment tracking. No MLflow server is required **to log runs**. To choose the default local SQLite layout, set:

```yaml
tracking:
  enabled: true
  provider: mlflow
  experiment_name: gnn_diff_rntraj_nyc
  tracking_uri: null
  storage_scope: job  # job | shared
```

| `tracking.storage_scope` | SQLite database | Artifact root | Use when |
|---|---|---|---|
| `job` (default) | `outputs/experiments/mlflow/jobs/<job_id>/tracking.db` | `.../jobs/<job_id>/artifacts/` | Multiple simultaneous jobs; isolated SQLite writers |
| `shared` | `outputs/experiments/mlflow/tracking.db` | `outputs/experiments/mlflow/artifacts/` | Sequential jobs; compare runs together in a single MLflow UI |

`job` chooses a job subdirectory only when Slurm supplies `SLURM_JOB_ID`. Outside Slurm, it uses the common local `outputs/experiments/mlflow/tracking.db` directory. Each MLflow run still has a unique run ID and its own artifacts, even with `shared` storage. Both modes record Slurm job IDs as run tags.

The storage scope determines a default **only when `tracking.tracking_uri` is null**. An explicit URI in YAML takes precedence; a provided `MLFLOW_TRACKING_URI` is the next fallback. `URBAN_SYN_MLFLOW_SCOPE` is no longer used. The Slurm scripts pass `SLURM_JOB_ID` into Apptainer with `--cleanenv`, and the tracking adapter applies the YAML setting.

### Comparing runs

- **Shared database:** open a single MLflow UI backed by `outputs/experiments/mlflow/tracking.db` and select multiple runs in the same experiment for comparison. No server is required during training; the UI is started only to view results.
- **Per-job databases:** each database contains only its corresponding job's runs. You can inspect them one backend at a time, or query them individually using `mlflow.MlflowClient` to produce a combined report. To compare all runs in **one MLflow UI**, you must first consolidate them into a common tracking backend (with artifact paths preserved). Merely placing the database files in one directory does not merge them.
- **Concurrent jobs:** sharing one SQLite database on an HPC network filesystem can cause locking or reliability issues. Prefer `job` for concurrency. For central concurrent tracking at scale, use a proper MLflow backend (for example, PostgreSQL behind a tracking service), rather than shared SQLite.

**Generation tracking:** Standalone generation can attach to the same MLflow run when it can reach the original tracking URI. If the checkpoint refers to another per-job database, the adapter creates a new linked run rather than overwriting the previous one.

## Slurm resources and paths

The default allocation is deliberately sized for longer GPU experiments:

| Resource | Default |
|---|---|
| Partition | `gpuqm` |
| GPU | 1 × A100 |
| CPU | 8 cores |
| RAM | 64 GB |
| Wall time | 1 day |
| Apptainer image | `~/containers/urban-syn.sif` |

For a shorter job or to change limits, pass Slurm flags directly to the batch file:

```bash
mkdir -p hpc/urban_syn/logs
sbatch --partition=gpuqs --time=01:00:00 --mem=32G \
  --cpus-per-task=4 \
  --output="$PWD/hpc/urban_syn/logs/%x-%j.out" \
  --error="$PWD/hpc/urban_syn/logs/%x-%j.err" \
  hpc/urban_syn/run_urban_syn.sbatch all configs/experiments/nyc_joint.yaml
```

For direct `sbatch`, the experiment YAML still controls the MLflow storage scope. The `submit.sh` helper sets `--chdir`, exports the project root, and selects the Slurm log destinations automatically.

Override the checkout and image paths if needed:

```bash
export URBAN_SYN_PROJECT_ROOT=/absolute/path/to/URBAN-SYN
export URBAN_SYN_SIF=/absolute/path/to/urban-syn.sif
```

Experiment configuration paths remain repo-relative. The YAML must reside inside the checkout so the runner can translate its path to `/workspace/...`.

## Outputs and job monitoring

| Item | Persistent path under the HPC repository |
|---|---|
| Checkpoints, training metrics, generated GPS | `outputs/experiments/<experiment.name>/<run_id>/` |
| Offline MLflow | `outputs/experiments/mlflow/` (layout depends on scope) |
| Slurm stdout / stderr | `hpc/urban_syn/logs/urban-syn-<job_id>.out` / `.err` |
| GPU utilization and memory samples | `hpc/urban_syn/logs/gpu-<job_id>.csv` |
| CPU / RAM usage samples | `hpc/urban_syn/logs/resources-<job_id>.log` |
| Job metadata and exit code | `hpc/urban_syn/logs/job-<job_id>.txt` |

```bash
squeue -u "$USER"
tail -f hpc/urban_syn/logs/urban-syn-<job_id>.out
sacct -j <job_id> --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

All outputs remain outside the SIF on the mounted checkout. Checkpoint resume follows the existing YAML `training.checkpoint.resume_from` behavior; an interrupted job can resume only from a checkpoint it previously finished writing. Slurm signals do **not** trigger a new emergency checkpoint.

The optional `export-eval` stage writes an input handoff for the team's evaluator; it does not implement evaluation metrics. The runner does not yet support multi-GPU training, automatic requeueing, or dependent Slurm jobs.
