#!/bin/bash
#SBATCH --job-name=fixed-metadata
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=standalone-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# PYTHON_BIN must refer to an environment with torch, pandas/pyarrow, PyYAML,
# and the repository dependencies. Slurm controls CUDA_VISIBLE_DEVICES.
"${PYTHON_BIN:-python}" -m urban_syn.diffusion.conditional.standalone "${1:?check|train|generate|evaluate}" --config "${2:?config.yaml}" --device "${DEVICE:-cuda:0}" "${@:3}"
