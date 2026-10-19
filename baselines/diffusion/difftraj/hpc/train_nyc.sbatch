#!/bin/bash
#SBATCH --job-name=difftraj_nyc_h100
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Train DiffTraj on NYC inside the Apptainer image, then generate trajectories:
#   1. main.py      train on the TRAIN trips (repo/traj.npy, repo/head.npy)
#   2. generate.py  one trajectory per TEST trip (repo/head_test.npy), in order
# The data files come from prepare_data.py (run it before uploading; see the README).
#
# Submit from the difftraj folder on the HPC, with difftraj.sif next to this hpc/ folder:
#   cd ~/URBAN-SYN/baselines/diffusion/difftraj && mkdir -p logs && sbatch hpc/train_nyc.sbatch
# Training settings (epochs, batch size, run name) come from repo/utils/config.py.
# Outputs: repo/DiffTraj/<run>/models/<timestamp>/unet_*.pt   (checkpoints every 50 epochs)
#          repo/DiffTraj/<run>/Files/<timestamp>/synthetic_trajectories.npy + synthetic_index.npy
#
# Generation only, with an existing model (newest checkpoint unless MODEL is set, relative to repo/):
#   sbatch --export=ALL,GENERATE_ONLY=1 --time=02:00:00 hpc/train_nyc.sbatch
#   sbatch --export=ALL,GENERATE_ONLY=1,MODEL="DiffTraj/<run>/models/<timestamp>/unet_2000.pt" hpc/train_nyc.sbatch
#
# Asks for an H100. If none is free, use another GPU type without editing this file:
#   sbatch --gres=gpu:a100:1 --job-name=difftraj_nyc_a100 hpc/train_nyc.sbatch
#   sbatch --gres=gpu:p100:1 --job-name=difftraj_nyc_p100 hpc/train_nyc.sbatch

set -eo pipefail
# Paths below (difftraj.sif, logs/, repo/) are relative to the folder the job was submitted from.
cd "${SLURM_SUBMIT_DIR:-$HOME/URBAN-SYN/baselines/diffusion/difftraj}"
[ -f difftraj.sif ] || { echo "difftraj.sif not found in $PWD - submit from the difftraj folder"; exit 1; }
# Apptainer (/usr/bin/apptainer on the CoE HPC); Singularity as a fallback.
RUN=$(command -v apptainer || command -v singularity)

nvidia-smi
echo "Node $(hostname), assigned GPU index: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk 'NR==1' || true)

# --nv exposes the GPU; repo/ (code + data) is mounted, results are written back into it.
container() {
    "$RUN" exec --nv --env PYTHONUNBUFFERED=1 --env MPLBACKEND=Agg \
        --bind "$PWD/repo:/workspace/repo" --pwd /workspace/repo difftraj.sif "$@"
}
now() { date '+%Y-%m-%d %H:%M:%S'; }
hms() { printf '%dh %02dm %02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

for f in head_test.npy traj_stats.npy; do
    [ -f "repo/$f" ] || { echo "repo/$f not found - run prepare_data.py and upload its files"; exit 1; }
done
if [ "${GENERATE_ONLY:-0}" != 1 ]; then
    for f in traj.npy head.npy; do
        [ -f "repo/$f" ] || { echo "repo/$f not found - run prepare_data.py and upload its files"; exit 1; }
    done
fi

# Quick GPU test before anything else, so a broken GPU stops the job in seconds.
if ! container python -c "import torch; torch.zeros(1).cuda()" >/dev/null 2>&1; then
    echo "[$(now)] GPU ${CUDA_VISIBLE_DEVICES:-?} on $(hostname) is not usable."
    echo "Resubmit avoiding this node, e.g.:  sbatch --exclude=$(hostname) --gres=gpu:a100:1 hpc/train_nyc.sbatch"
    exit 1
fi

START=$SECONDS
if [ "${GENERATE_ONLY:-0}" = 1 ]; then
    echo "[$(now)] GENERATE_ONLY=1: skipping training"
else
    container python -c "
import numpy as np
print(f'train trips: {len(np.load(\"head.npy\"))}   test trips: {len(np.load(\"head_test.npy\"))}')
" || true
    echo "[$(now)] training"
    container python main.py
fi
TRAIN_S=$((SECONDS - START))
echo "[$(now)] training done in $(hms $TRAIN_S)"

echo "[$(now)] generating from head_test.npy"
if [ -n "${MODEL:-}" ]; then
    container python generate.py --model "$MODEL" --head head_test.npy --stats traj_stats.npy
else
    container python generate.py --head head_test.npy --stats traj_stats.npy   # newest checkpoint = the one just trained
fi
GEN_S=$((SECONDS - START - TRAIN_S))
echo "[$(now)] generation done in $(hms $GEN_S) - output in $(cat repo/DiffTraj/last_generation.txt 2>/dev/null)"

TOTAL_S=$((SECONDS - START))
echo "[$(now)] total (training + generation): $(hms $TOTAL_S)"

# One line per run, so the times can be compared later.
TIMES=repo/DiffTraj/run_times.txt
mkdir -p repo/DiffTraj
[ -f "$TIMES" ] || echo "finished,job_id,gpu,generate_only,training,generation,total" > "$TIMES"
echo "$(now),${SLURM_JOB_ID:-none},$GPU,${GENERATE_ONLY:-0},$(hms $TRAIN_S),$(hms $GEN_S),$(hms $TOTAL_S)" >> "$TIMES"
