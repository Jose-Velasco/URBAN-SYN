# LSTM-TrajGAN — HPC Scripts

Slurm scripts for preprocessing, training, generating trajectories, and archiving LSTM-TrajGAN experiments on the SJSU CoE HPC cluster.

Run the commands below from `baselines/gan/lstm_trajgan/`. Keep these scripts in its `hpc/` directory.

## Scripts

- `run_lstm_trajgan.sbatch` — Runs one pipeline stage in Apptainer (1 P100 GPU, 4 CPUs, 64 GB RAM by default).
- `submit_full_run.sh` — Submits `prepare → train → generate` with `afterok` dependencies.
- `archive_run.sh` — Saves the run's data, checkpoints, generated results, logs, configuration, and metadata.

## Run the pipeline

```bash
# Submit all stages in a single job (default target: all)
sbatch hpc/run_lstm_trajgan.sbatch all

# Recommended for a full run: separate dependent jobs
bash hpc/submit_full_run.sh

# Run specific pipeline stages
sbatch hpc/run_lstm_trajgan.sbatch prepare        # build data + encode train/test
sbatch hpc/run_lstm_trajgan.sbatch train          # train model; needs prepared inputs
sbatch hpc/run_lstm_trajgan.sbatch generate       # generate trajectories; needs checkpoint

# Run individual preprocessing stages if needed
sbatch hpc/run_lstm_trajgan.sbatch build          # build NYC input CSVs
sbatch hpc/run_lstm_trajgan.sbatch encode-train   # encode training data
sbatch hpc/run_lstm_trajgan.sbatch encode-test    # encode test data
```

`submit_full_run.sh` requests **35 minutes** for `prepare` (`gpuqs`), **24 hours** for `train` (`gpuqm`), and **4 hours** for `generate` (`gpuqs`). Downstream jobs start only if preceding jobs succeed.

## Monitor jobs

```bash
# Check your queued/running jobs
squeue -u "$USER"

# Check completion status and exit codes
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,ExitCode

# View live standard output (replace with actual log filename)
tail -f hpc/logs/<JOB_NAME>-<JOB_ID>.out

# View resource and GPU monitoring logs
tail -f hpc/logs/resources-<JOB_ID>.log
tail -f hpc/logs/gpu-<JOB_ID>.csv
```

## Archive a run

```bash
# Archive one job; automatically name the run with timestamp, job ID, and label
bash hpc/archive_run.sh --job-id <JOB_ID> --label nyc_smoke

# Archive the full dependent job chain (include all three job IDs)
bash hpc/archive_run.sh \
  --run-id nyc_full_run_01 \
  --job-id <PREPARE_JOB_ID> \
  --job-id <TRAIN_JOB_ID> \
  --job-id <GENERATE_JOB_ID>

# Keep the working data, outputs, and logs instead of moving them
bash hpc/archive_run.sh --job-id <JOB_ID> --label nyc_full --copy

# Show archiver options
bash hpc/archive_run.sh --help
```

Archives go to `hpc/runs/<RUN_ID>/`. **By default, archiving moves artifacts out of the working directories** and recreates empty directories. Use `--copy` to keep working copies; archive only after the relevant jobs finish.

## Main data and output locations

For the run used in this project:

```text
Input source data:
<URBAN-SYN>/data/

Prepared LSTM-TrajGAN data:
<URBAN-SYN>/baselines/gan/lstm_trajgan/data/nyc_full/

Training checkpoints:
<URBAN-SYN>/baselines/gan/lstm_trajgan/outputs/lstm_trajgan_nyc_full/training_params/

Generated trajectories:
<URBAN-SYN>/baselines/gan/lstm_trajgan/outputs/lstm_trajgan_nyc_full/results/

Slurm/resource/GPU logs:
<URBAN-SYN>/baselines/gan/lstm_trajgan/hpc/logs/
```


## Important paths and setup

| Purpose | Location (relative to `lstm_trajgan/`) |
|---|---|
| Pipeline configuration | `pipeline_configs/nyc.yaml` |
| Prepared input data | `data/nyc_full/` |
| Model checkpoints | `outputs/lstm_trajgan_nyc_full/training_params/` |
| Generated trajectories | `outputs/lstm_trajgan_nyc_full/results/` |
| Slurm and resource logs | `hpc/logs/` |
| Archived runs | `hpc/runs/` |

**Before running on another account:** edit the absolute project paths, `#SBATCH --output`, `--error`, `--chdir`, email address, and Apptainer image path in `run_lstm_trajgan.sbatch`. The current image location is `/home/016184753/containers/lstm-trajgan.sif`. The runner also expects the source dataset at `<URBAN-SYN>/data/`.

The scripts assume their original `baselines/gan/lstm_trajgan/hpc/` location and are **not directly relocatable** without adjusting paths.
