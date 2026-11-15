# URBAN-SYN on SJSU HPC: Slurm + Apptainer

This guide explains how to run the existing URBAN-SYN experiment pipeline on SJSU's GPU cluster. Slurm allocates the hardware, Apptainer supplies the software environment, and the Python pipeline handles graph preparation, training, and generation.

There is **no separate HPC training implementation**. The same experiment YAML and `urban_syn.experiment` stages work locally and on HPC.

For building and transferring the container, see [README.md](README.md). For model configuration and NYC data preparation, see the [experiment README](../../src/urban_syn/experiment/README.md). For downloading runs and viewing them in a browser, see [README_MLFLOW.md](README_MLFLOW.md).

## 1. What you need on HPC

- An Apptainer image at `$HOME/containers/urban-syn.sif`, or a custom absolute path via `URBAN_SYN_SIF`.
- A checkout of the URBAN-SYN repository with its experiment and graph YAML files.
- The prepared **training** road-sequence Parquet and the shared road-network GeoPackage.
- The training trajectory-length distribution when running generation.
- The training FMM matches CSV **if** the graph's transition-count feature is enabled and the graph cache must be built.
- A valid GPU allocation, using the partition and GPU type allowed by the cluster.

**You do not need to transfer `road_graph.pt` or `preprocessor.joblib` for a new checkout.** The `prepare-graph` stage generates them from the source GeoPackage and graph YAML if either is missing. The pipeline does not perform sampling, FMM map matching, or the train/test split. Prepare those inputs before submitting a model job.

The scripts mount the repository at `/workspace` and execute `/opt/venv/bin/python` inside the SIF. Package installation is not repeated during jobs. Both input and output paths are resolved to absolute locations during execution, while the experiment YAML can keep its convenient repository-relative paths.

## 2. Choose and submit a pipeline stage

Run these commands from the HPC URBAN-SYN repository root:

```bash
# Full workflow: validate, prepare graph, train, then generate if enabled.
bash hpc/urban_syn/submit.sh all \
  configs/experiments/nyc_joint.yaml

# Train only; prepares the graph when necessary.
bash hpc/urban_syn/submit.sh train \
  configs/experiments/nyc_joint.yaml

# Generate from a specified checkpoint, without retraining.
bash hpc/urban_syn/submit.sh generate \
  configs/experiments/nyc_joint.yaml

# Prepare or reuse the graph cache only.
bash hpc/urban_syn/submit.sh prepare-graph \
  configs/experiments/nyc_joint.yaml
```

Two useful planning modes are available:

```bash
# Display the selected stages without checking files.
bash hpc/urban_syn/submit.sh all \
  configs/experiments/nyc_joint.yaml --dry-run

# Check source inputs and settings without building the graph or training.
bash hpc/urban_syn/submit.sh all \
  configs/experiments/nyc_joint.yaml --validate-only
```

`all` runs `validate`, `prepare-graph`, and `train`. It adds `generate` when `generation.enabled: true`, and adds `export-eval` when a held-out GPS reference is configured. The optional handoff writes a manifest for the shared evaluator; it does not run evaluation metrics.

### Standalone generation

For a generation-only job, set `generation.checkpoint` to an existing, compatible training checkpoint in the YAML:

```yaml
generation:
  enabled: true
  checkpoint: outputs/experiments/nyc_joint_example/<run_id>/checkpoints/best.pt
  checkpoint_choice: best
  num_trajectories: 100
  sampler:
    name: ddpm
```

Replace `<run_id>` with the actual run directory. Configure `paths.road_network` and `paths.length_distribution` as well. The `generate` target loads that checkpoint and prepares the graph if required, but **does not train**. For `all`, leave `generation.checkpoint: null` to use the selected checkpoint from the current training run.

Generation writes into the checkpoint's run directory. Choose a different `generation.output_name` if you want multiple exports from the same checkpoint without replacing the earlier generated file.

## 3. Slurm resources

The runner's normal resource request is:

| Resource | Default request |
|---|---|
| Partition | `gpuqm` |
| GPU | 1 A100 |
| CPU | 8 cores |
| Memory | 64 GB |
| Wall time | 1 day |

You can override these **for a single submission** without editing the `.sbatch` file. For example:

```bash
SBATCH_PARTITION=gpuqs \
SBATCH_TIMELIMIT=00:30:00 \
SBATCH_GRES=gpu:a100:1 \
SBATCH_CPUS_PER_TASK=4 \
SBATCH_MEM_PER_NODE=32G \
bash hpc/urban_syn/submit.sh all \
  configs/experiments/nyc_joint.yaml
```

These `SBATCH_*` variables request Slurm resources without changing model hyperparameters. **Use the YAML for epochs, batches, optimizer, checkpoints, generation, and MLflow.** Partition, GPU, CPU, RAM, and wall time are scheduling choices. Confirm site-supported settings with `sinfo` or `scontrol`, and verify the actual allocation with `scontrol show job JOB_ID`, because the applied resources can differ from the requested overrides.

The `submit.sh` helper also sets absolute job working-directory and stdout/stderr paths. Use it for normal submissions rather than calling the batch script with `bash`.

If the SIF or project checkout is in a different place, set:

```bash
export URBAN_SYN_PROJECT_ROOT=/absolute/path/to/URBAN-SYN
export URBAN_SYN_SIF=/absolute/path/to/urban-syn.sif
```

The selected YAML must be inside the project checkout so its path can be mapped into `/workspace`.

## 4. MLflow tracking

MLflow records separate runs, with their parameters, metrics, tags, and selected artifacts. **You do not need a running MLflow server to log**. Tracking storage is controlled centrally by the experiment YAML:

```yaml
tracking:
  enabled: true
  provider: mlflow
  experiment_name: gnn_diff_rntraj_nyc
  tracking_uri: null
  storage_scope: shared  # shared or job
```

| Scope | Backend location under `outputs/experiments/` | When to use it |
|---|---|---|
| `shared` | `mlflow/tracking.db` | Sequential runs that should appear together in one MLflow UI |
| `job` | `mlflow/jobs/<job_id>/tracking.db` | Concurrent jobs where SQLite writers should be isolated |

Choose the scope **explicitly** in the YAML. A new run has its own MLflow ID in either mode; reusing the same YAML does not overwrite previous runs. With per-job databases, runs are preserved but cannot automatically be compared in one MLflow UI without consolidation. A shared SQLite backend is convenient for sequential experiments, but concurrent access on network storage can lead to locking problems.

If `tracking.tracking_uri` is set explicitly, it takes precedence over the automatic layout. The runner passes the Slurm job ID into Apptainer so it can be recorded with the run and used when `storage_scope: job`. For local snapshot viewing, including Windows 11 Git Bash path conversion, follow [README_MLFLOW.md](README_MLFLOW.md).

## 5. Where files and logs go

Model outputs are written to the **mounted repository**, not to the SIF:

```text
outputs/experiments/
├── <experiment_name>/<run_id>/
│   ├── resolved_config.json
│   ├── checkpoints/
│   ├── metrics/history.jsonl
│   ├── logs/train.log
│   └── generated/
└── mlflow/
    └── ...
```

Slurm logs are kept in `hpc/urban_syn/logs/`:

| File | What it contains |
|---|---|
| `urban-syn-<job_id>.out` | Pipeline messages and progress |
| `urban-syn-<job_id>.err` | Errors and warnings |
| `gpu-<job_id>.csv` | GPU activity, memory, temperature, and power samples |
| `resources-<job_id>.log` | CPU and memory samples from Slurm |
| `job-<job_id>.txt` | Job details, resolved paths, Git revision, and exit code |

To check a submitted job:

```bash
squeue -u "$USER"
```

` squeue ` shows waiting and running jobs, not completed jobs. For a finished job, use:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

Then inspect its messages:

```bash
tail -n 80 hpc/urban_syn/logs/urban-syn-JOB_ID.err
tail -n 80 hpc/urban_syn/logs/urban-syn-JOB_ID.out
```

Replace `JOB_ID` with the Slurm job number. You can also follow a running job with:

```bash
tail -f hpc/urban_syn/logs/urban-syn-JOB_ID.out
```

Common first checks are the configured input files, the SIF path, the CUDA allocation, and whether the graph YAML refers to a training FMM CSV available on HPC. A missing PyG cache alone is not an error if source files exist and graph preparation is enabled.

## 6. Resuming and limitations

To continue interrupted training, set `training.checkpoint.resume_from` to a compatible `last.pt` path in the YAML and submit `train` or `all` again. Checkpoints are saved according to the configured epoch cadence. Receiving a Slurm timeout or termination signal does **not** force the runner to write an extra emergency checkpoint.

The current scripts execute one pipeline target per Slurm job on one GPU. They do not provide automatic requeueing, distributed multi-GPU training, or dependent train/generate job submission. These can be added without changing the existing model runner if later experiments require them.
