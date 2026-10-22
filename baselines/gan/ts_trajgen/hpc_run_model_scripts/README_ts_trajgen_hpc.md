# TS-TrajGen — HPC Scripts

Slurm/Apptainer scripts for running the NYC TS-TrajGen pipeline on the SJSU CoE HPC cluster. Run these commands from `baselines/gan/ts_trajgen/` on the cluster (not on the login node directly for training).

## Before running

**IMPORTANT**: In HPC put these files in `/URBAN-SYN/baselines/gan/ts_trajgen/hpc` to run directly

- Confirm the paths in `hpc/run_ts_trajgen.sbatch` match your account. **Project root, Slurm working/log directories, notification email, and container path are currently hardcoded for one user.** Update them before submitting.
- The script expects `/home/$USER/containers/ts-trajgen.sif`, `pipeline_configs/nyc.yaml`, `repo/configs/ts_trajgen_nyc.yaml`, and the upstream FMM/NYC dataset files under the project tree.
- Check the YAML settings (especially smoke-test limits, debug mode, and training epochs) before a full run. Keep `GAN_EXP_ID` in the batch script consistent with `gan.exp_id` in the pipeline YAML.
- The batch script requests **1 A100 GPU, 4 CPUs, and 32 GB RAM** by default. Its default partition/time are `gpuqs` / 2 hours; adjust the `#SBATCH` settings or override them on submission when needed.

## Run

```bash
# Submit all stages as one job (default target: all)
sbatch hpc/run_ts_trajgen.sbatch all

# Recommended for longer runs: submit a dependent job chain
bash hpc/submit_full_run.sh

# Run specific stages
sbatch hpc/run_ts_trajgen.sbatch setup-gans   # preprocessing + road/region pretraining
sbatch hpc/run_ts_trajgen.sbatch train-road   # road GAN
sbatch hpc/run_ts_trajgen.sbatch train-region # region GAN
sbatch hpc/run_ts_trajgen.sbatch train        # road GAN, then region GAN
sbatch hpc/run_ts_trajgen.sbatch generate     # generate using trained GAN checkpoints
```

`submit_full_run.sh` submits **setup-gans → (road GAN and region GAN independently) → generation**. Each dependent stage runs only if its prerequisites succeed. The script overrides partitions and time limits for its jobs; review those values before submission. It does **not** archive outputs automatically.

Other supported single-stage targets: `prepare`, `pretrain-road`, `prepare-region`, `pretrain-region`.

## Monitor and archive

```bash
squeue -u "$USER"                    # queued/running jobs
sacct -j <job-id> --format=JobID,State,Elapsed,ExitCode

# After the run has finished
bash hpc/archive_run.sh --job-id <job-id> --label full_nyc

# Preserve the dataset and checkpoints in their working directories
bash hpc/archive_run.sh --job-id <job-id> --label full_nyc --copy
```

- **Logs:** `hpc/logs/` (`<job-name>-<job-id>.out/.err`, GPU and resource monitoring logs).
- **Working outputs:** `datasets/nyc/`, `repo/save/nyc/`, and `repo/temp/nyc/`.
- **Archives:** `hpc/runs/<timestamp>_job<id>_<label>/` (artifacts, configurations, logs, and run metadata).
- **Important:** By default, `archive_run.sh` **moves** the dataset and model checkpoints out of their working directories, and also moves temporary/upstream logs. `--copy` copies the dataset/checkpoints instead, **but still moves** temporary/upstream logs. Archive only after relevant jobs finish, especially when jobs share intermediate artifacts. The archiver records Slurm logs for the supplied job ID only; for a multi-job chain, keep track of all job IDs/logs.

## Main data and output locations

For the NYC run used in this project:

```text
Input source data:
<URBAN-SYN>/data/nyc_output_tabular/output/traj_cleaned.parquet
<URBAN-SYN>/fmm_scripts/data/road_network/nyc.gpkg
<URBAN-SYN>/fmm_scripts/outputs/nyc_fmm_match.csv

Prepared TS-TrajGen data:
<URBAN-SYN>/baselines/gan/ts_trajgen/datasets/nyc/

Training checkpoints:
<URBAN-SYN>/baselines/gan/ts_trajgen/repo/save/nyc/

Generated trajectories:
<URBAN-SYN>/baselines/gan/ts_trajgen/outputs/

Temporary training artifacts:
<URBAN-SYN>/baselines/gan/ts_trajgen/repo/temp/nyc/

Slurm/resource/GPU logs:
<URBAN-SYN>/baselines/gan/ts_trajgen/hpc/logs/

Archived runs:
<URBAN-SYN>/baselines/gan/ts_trajgen/hpc/runs/
```

**Clean-run guard:** The `all` and `setup-gans` targets refuse to start if the NYC dataset or model-save directory is nonempty. Archive prior artifacts before launching a new clean run.
