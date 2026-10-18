# LSTM-TrajGAN HPC Run Scripts

This directory contains the Slurm/HPC scripts used to run and archive the LSTM-TrajGAN baseline on the SJSU College of Engineering HPC cluster.



## Files

**IMPORTANT**: In HPC put these files in `/URBAN-SYN/baselines/gan/lstm_trajgan/hpc` to run directly

- `run_lstm_trajgan.sbatch` — Runs one pipeline target (`prepare`, `train`, `generate`, etc.) inside the LSTM-TrajGAN Apptainer container. It requests one NVIDIA P100 GPU, 4 CPUs, and 64 GB RAM, and records Slurm, CPU/RAM, and GPU monitoring logs.
- `submit_full_run.sh` — Submits the full run as three dependent Slurm jobs:

  ```text
  Prepare --> Train --> Generate
  ```

  The training job starts only if preparation succeeds, and generation starts only if training succeeds.
- `archive_run.sh` — Archives run artifacts, prepared data, model outputs/checkpoints, Slurm/resource logs, configuration snapshots, and run metadata into a run-specific directory.

## Full-run submission

The full pipeline was submitted with:

```bash
./submit_full_run.sh
```

The script submits:

```text
prepare   -> gpuqs, 35 min
train     -> gpuqm, 24 hr
generate  -> gpuqs, 4 hr
```

The underlying `.sbatch` script runs the project wrapper as:

```bash
python /workspace/run_pipeline.py <target>     --config /workspace/pipeline_configs/nyc.yaml
```

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

## Archiving a completed run

Pass all Slurm job IDs from the dependency chain:

```bash
./archive_run.sh   --run-id <RUN_ID>   --job-id <PREPARE_JOB_ID>   --job-id <TRAIN_JOB_ID>   --job-id <GENERATE_JOB_ID>
```

By default, the archiver moves the working data/output directories into the archive and recreates clean working directories. Use `--copy` to preserve the working copies.

## Note about this directory

These files are kept here as a record of the scripts used for the HPC experiments. Some paths inside `run_lstm_trajgan.sbatch` still explicitly reference the original `lstm_trajgan/hpc/` directory for logs and execution paths. If these copies are used directly from `hpc_run_model_scripts/`, review those paths first rather than assuming the directory is fully relocatable.
