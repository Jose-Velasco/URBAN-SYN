#!/bin/bash
#SBATCH --job-name=diffrntraj_nyc_h100
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Train Diff-RNTraj on NYC inside the Apptainer image, then generate trajectories with the trained model 
# Submit from the diff_rntraj folder on the HPC, e.g.
#   cd ~/URBAN-SYN/baselines/diffusion/diff_rntraj && mkdir -p logs && sbatch hpc/train_nyc.sbatch

# Extra arguments go to both multi_main.py and generate_data.py (they take the same
# options), e.g.   sbatch hpc/train_nyc.sbatch --epochs 5

# Outputs: repo/results/NYC/ (model, losses, run_times.txt) and repo/generate_data/NYC/ (trajectories).
# Generation only, with the model already in repo/results/NYC/:
#   sbatch --export=ALL,GENERATE_ONLY=1 hpc/train_nyc.sbatch

set -eo pipefail
# Paths below (repo/, diffrntraj.sif, logs/) are relative to the folder the job was submitted from.
cd "${SLURM_SUBMIT_DIR:-$HOME/URBAN-SYN/baselines/diffusion/diff_rntraj}"
[ -f diffrntraj.sif ] || { echo "diffrntraj.sif not found in $PWD - submit from the diff_rntraj folder"; exit 1; }

# Apptainer (/usr/bin/apptainer on the CoE HPC); Singularity as a fallback.
RUN=$(command -v apptainer || command -v singularity)
now() { date '+%Y-%m-%d %H:%M:%S'; }

[ -d repo/data/NYC/gen_train ] || { echo "repo/data/NYC/gen_train is missing - rerun the convert step locally and upload repo/data/NYC"; exit 1; }

nvidia-smi

echo "Node $(hostname), assigned GPU index: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# awk reads all lines (head -1 would kill nvidia-smi with SIGPIPE on multi-GPU nodes,
# and pipefail would then stop the job); || true so this line can never fail the job.
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk 'NR==1' || true)

# --nv exposes the GPU; the repo (code + data/NYC) is mounted, results are written back into it.
container() {
    # PYTHONUNBUFFERED=1: Python output shows up in the .out log right away, not in delayed chunks.
    "$RUN" exec --nv --env PYTHONUNBUFFERED=1 --bind "$PWD/repo:/workspace/repo" --pwd /workspace/repo diffrntraj.sif "$@"
}

# Quick GPU test before anything else: a broken GPU (e.g. cs002 GPU 0, "CUDA-capable
# device(s) is/are busy or unavailable") then stops the job in seconds with a clear hint.
if ! container python -c "import torch; torch.zeros(1).cuda()" >/dev/null 2>&1; then
    echo "[$(now)] GPU ${CUDA_VISIBLE_DEVICES:-?} on $(hostname) is not usable."
    echo "Resubmit avoiding this node, e.g.:  sbatch --exclude=$(hostname) --gres=gpu:a100:1 hpc/train_nyc.sbatch"
    exit 1
fi
hms() { printf '%dh %02dm %02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

START=$SECONDS
if [ "${GENERATE_ONLY:-0}" = 1 ]; then
    [ -f repo/results/NYC/val-best-model.pt ] || { echo "GENERATE_ONLY=1 but repo/results/NYC/val-best-model.pt is missing"; exit 1; }
    echo "[$(now)] GENERATE_ONLY=1: skipping training, using repo/results/NYC/val-best-model.pt"
else
    echo "[$(now)] training"
    container python multi_main.py --dataset NYC "$@"
fi
TRAIN_S=$((SECONDS - START))
echo "[$(now)] training done in $(hms $TRAIN_S)"

# generate_data.py appends to generate_data/NYC/<length>.txt, so clear old output first.
echo "[$(now)] generating trajectories"
rm -rf repo/generate_data/NYC
container python generate_data.py --dataset NYC "$@"
GEN_S=$((SECONDS - START - TRAIN_S))
echo "[$(now)] generation done in $(hms $GEN_S)"

TOTAL_S=$((SECONDS - START))
echo "[$(now)] total (training + generation): $(hms $TOTAL_S)"

# One line per run, kept next to the model so the times can be compared later.
TIMES=repo/results/NYC/run_times.txt
[ -f "$TIMES" ] || echo "finished,job_id,gpu,options,training,generation,total" > "$TIMES"
echo "$(now),${SLURM_JOB_ID:-none},$GPU,\"$*\",$(hms $TRAIN_S),$(hms $GEN_S),$(hms $TOTAL_S)" >> "$TIMES"
