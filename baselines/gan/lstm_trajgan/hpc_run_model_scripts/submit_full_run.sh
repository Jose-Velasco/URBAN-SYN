#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_lstm_trajgan.sbatch"

echo "Submitting LSTM-TrajGAN full pipeline..."

PREPARE_JOB=$(sbatch --parsable \
    --job-name=lstm-prepare \
    --partition=gpuqs \
    --time=00:35:00 \
    "${SBATCH_SCRIPT}" prepare)

echo "Submitted prepare: ${PREPARE_JOB}"

TRAIN_JOB=$(sbatch --parsable \
    --job-name=lstm-train \
    --partition=gpuqm \
    --time=1-00:00:00 \
    --dependency=afterok:${PREPARE_JOB} \
    "${SBATCH_SCRIPT}" train)

echo "Submitted train: ${TRAIN_JOB}"

GENERATE_JOB=$(sbatch --parsable \
    --job-name=lstm-generate \
    --partition=gpuqs \
    --time=04:00:00 \
    --dependency=afterok:${TRAIN_JOB} \
    "${SBATCH_SCRIPT}" generate)

echo "Submitted generate: ${GENERATE_JOB}"

echo
echo "============================================================"
echo "LSTM-TrajGAN Slurm dependency chain"
echo "============================================================"
echo "Prepare:  ${PREPARE_JOB}"
echo "Train:    ${TRAIN_JOB}"
echo "Generate: ${GENERATE_JOB}"
echo
echo "Dependency graph:"
echo
echo "Prepare --> Train --> Generate"
echo "============================================================"

