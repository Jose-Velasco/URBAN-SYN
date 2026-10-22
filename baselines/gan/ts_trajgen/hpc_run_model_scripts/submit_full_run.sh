#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_ts_trajgen.sbatch"

echo "Submitting TS-TrajGen full pipeline..."

# ---------------------------------------------------------------------
# 1. Preprocessing + road/region pretraining
# ---------------------------------------------------------------------
SETUP_JOB=$(sbatch --parsable \
    --job-name=ts-setup-gans \
    --partition=gpuqm \
    --time=1-00:00:00 \
    "${SBATCH_SCRIPT}" setup-gans)

echo "Submitted setup-gans: ${SETUP_JOB}"


# ---------------------------------------------------------------------
# 2. Road GAN
# Starts only if setup-gans succeeds.
# ---------------------------------------------------------------------
ROAD_JOB=$(sbatch --parsable \
    --job-name=ts-road-gan \
    --partition=gpuql \
    --time=3-00:00:00 \
    --dependency=afterok:${SETUP_JOB} \
    "${SBATCH_SCRIPT}" train-road)

echo "Submitted road GAN: ${ROAD_JOB}"


# ---------------------------------------------------------------------
# 3. Region GAN
# Starts only if setup-gans succeeds.
# Independent of road GAN, so Slurm may run both GAN jobs in parallel.
# ---------------------------------------------------------------------
REGION_JOB=$(sbatch --parsable \
    --job-name=ts-region-gan \
    --partition=gpuqm \
    --time=1-00:00:00 \
    --dependency=afterok:${SETUP_JOB} \
    "${SBATCH_SCRIPT}" train-region)

echo "Submitted region GAN: ${REGION_JOB}"


# ---------------------------------------------------------------------
# 4. Generation
# Starts only after BOTH GAN jobs succeed.
# ---------------------------------------------------------------------
GENERATE_JOB=$(sbatch --parsable \
    --job-name=ts-generate \
    --partition=gpuql \
    --time=3-00:00:00 \
    --dependency=afterok:${ROAD_JOB}:${REGION_JOB} \
    "${SBATCH_SCRIPT}" generate)

echo "Submitted generation: ${GENERATE_JOB}"


echo
echo "============================================================"
echo "TS-TrajGen Slurm dependency chain"
echo "============================================================"
echo "Setup GANs: ${SETUP_JOB}"
echo "Road GAN:   ${ROAD_JOB}"
echo "Region GAN: ${REGION_JOB}"
echo "Generate:   ${GENERATE_JOB}"
echo
echo "Dependency graph:"
echo
echo "                    +--> Road GAN -----+"
echo "Setup GANs ---------+                  +--> Generate"
echo "                    +--> Region GAN ---+"
echo "============================================================"
