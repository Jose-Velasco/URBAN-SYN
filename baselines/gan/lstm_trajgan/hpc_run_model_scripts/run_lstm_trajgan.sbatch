#!/bin/bash
#SBATCH --job-name=lstm-trajgan
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

#SBATCH --mail-user=jose.velascojuarez@sjsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output=/home/016184753/URBAN-SYN/baselines/gan/lstm_trajgan/hpc/logs/%x-%j.out
#SBATCH --error=/home/016184753/URBAN-SYN/baselines/gan/lstm_trajgan/hpc/logs/%x-%j.err
#SBATCH --chdir=/home/016184753/URBAN-SYN/baselines/gan/lstm_trajgan

set -euo pipefail

# =============================================================================
# Experiment configuration
# =============================================================================

DATASET_NAME="nyc"
PIPELINE_GROUP="${1:-all}"

# =============================================================================
# Paths
# =============================================================================

PROJECT="/home/016184753/URBAN-SYN"
BASE="${PROJECT}/baselines/gan/lstm_trajgan"
HPC_DIR="${BASE}/hpc"

SIF="/home/016184753/containers/lstm-trajgan.sif"
PIPELINE_CONFIG="${BASE}/pipeline_configs/${DATASET_NAME}.yaml"

DATA_DIR="${BASE}/data/nyc_full"
OUTPUT_DIR="${BASE}/outputs/lstm_trajgan_nyc_full"
CHECKPOINT_DIR="${OUTPUT_DIR}/training_params"
RESULTS_DIR="${OUTPUT_DIR}/results"

SOURCE_DATA_DIR="${PROJECT}/data"

RESOURCE_LOG="${HPC_DIR}/logs/resources-${SLURM_JOB_ID}.log"
GPU_LOG="${HPC_DIR}/logs/gpu-${SLURM_JOB_ID}.csv"

# =============================================================================
# Pipeline group validation
# =============================================================================

case "${PIPELINE_GROUP}" in
    build|encode-train|encode-test|prepare|train|generate|all)
        ;;
    *)
        echo "ERROR: Invalid pipeline group: ${PIPELINE_GROUP}"
        echo "Valid groups: build encode-train encode-test prepare train generate all"
        exit 1
        ;;
esac

# =============================================================================
# Basic safety checks
# =============================================================================

require_file() {
    local file="$1"
    if [[ ! -f "${file}" ]]; then
        echo "ERROR: Required file does not exist:"
        echo "  ${file}"
        exit 1
    fi
}

require_dir() {
    local dir="$1"
    if [[ ! -d "${dir}" ]]; then
        echo "ERROR: Required directory does not exist:"
        echo "  ${dir}"
        exit 1
    fi
}

resolve_checkpoint_epoch() {
    apptainer exec \
        --bind "${BASE}:/workspace" \
        "${SIF}" \
        python -c '
import yaml

with open("/workspace/pipeline_configs/nyc.yaml", "r") as f:
    config = yaml.safe_load(f)

checkpoint_epoch = config.get("generate", {}).get("checkpoint_epoch")

if checkpoint_epoch is None:
    checkpoint_epoch = config["train"]["epochs"]

print(int(checkpoint_epoch))
'
}

require_file "${SIF}"
require_file "${PIPELINE_CONFIG}"
if [[ "${PIPELINE_GROUP}" == "build" ||
      "${PIPELINE_GROUP}" == "prepare" ||
      "${PIPELINE_GROUP}" == "all" ]]; then
    require_dir "${SOURCE_DATA_DIR}"
fi

mkdir -p "${HPC_DIR}/logs"

if [[ "${PIPELINE_GROUP}" == "train" ]]; then
    require_file "${DATA_DIR}/${DATASET_NAME}_train_latlon.csv"
    require_file "${DATA_DIR}/${DATASET_NAME}_test_latlon.csv"
    require_file "${DATA_DIR}/final_train.npy"
fi

if [[ "${PIPELINE_GROUP}" == "generate" ]]; then
    require_file "${DATA_DIR}/${DATASET_NAME}_train_latlon.csv"
    require_file "${DATA_DIR}/${DATASET_NAME}_test_latlon.csv"
    require_file "${DATA_DIR}/${DATASET_NAME}_dev_test_encoded_final.csv"
    require_file "${DATA_DIR}/final_test.npy"

    CHECKPOINT_EPOCH="$(resolve_checkpoint_epoch)"
    require_file "${CHECKPOINT_DIR}/G_model_${CHECKPOINT_EPOCH}.h5"
fi

# =============================================================================
# Job information
# =============================================================================

echo
echo "================================================================================"
echo "LSTM-TrajGAN"
echo "================================================================================"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Job name:             ${SLURM_JOB_NAME}"
echo "Host:                 $(hostname)"
echo "Start:                $(date --iso-8601=seconds)"
echo "Partition:            ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs per task:        ${SLURM_CPUS_PER_TASK:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Working directory:    $(pwd)"
echo
echo "Dataset:              ${DATASET_NAME}"
echo "Pipeline group:       ${PIPELINE_GROUP}"
echo "Pipeline config:      ${PIPELINE_CONFIG}"
echo "Container:            ${SIF}"
echo
echo "Data directory:       ${DATA_DIR}"
echo "Output directory:     ${OUTPUT_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo
echo "Memory per node:      ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "GPUs:                 ${SLURM_GPUS:-unknown}"
echo "GRES:                 ${SLURM_JOB_GPUS:-unknown}"
echo "================================================================================"

echo
echo "=== GPU AT START ==="
nvidia-smi

echo
echo "=== PIPELINE CONFIG SETTINGS ==="
grep -nE \
    'max_trajectories|epochs|batch_size|save_params_rate|max_length|checkpoint_epoch|category_mode|constant_category' \
    "${PIPELINE_CONFIG}" \
    || true

# =============================================================================
# Resource monitoring
# =============================================================================

echo \
"timestamp,index,name,gpu_util_percent,memory_used_MiB,memory_total_MiB,temperature_C,power_draw_W" \
    > "${GPU_LOG}"

monitor_resources() {
    while true; do
        {
            echo "===== $(date --iso-8601=seconds) ====="
            sstat \
                -j "${SLURM_JOB_ID}.batch" \
                --format=JobID,AveCPU,AveRSS,MaxRSS,AveVMSize,MaxVMSize \
                -n \
                || true
            echo
        } >> "${RESOURCE_LOG}"

        sleep 120
    done
}

monitor_gpu() {
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits \
            >> "${GPU_LOG}" \
            || true

        sleep 120
    done
}

monitor_resources &
RESOURCE_MONITOR_PID=$!

monitor_gpu &
GPU_MONITOR_PID=$!

cleanup() {
    echo
    echo "Stopping resource monitors..."

    kill "${RESOURCE_MONITOR_PID}" 2>/dev/null || true
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true

    wait "${RESOURCE_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
}

trap cleanup EXIT

# =============================================================================
# Run pipeline
# =============================================================================

echo
echo "================================================================================"
echo "STARTING LSTM-TRAJGAN PIPELINE"
echo "================================================================================"
echo "Dataset:        ${DATASET_NAME}"
echo "Pipeline group: ${PIPELINE_GROUP}"
echo "================================================================================"
echo

apptainer exec --nv \
    --bind "${BASE}:/workspace" \
    --bind "${SOURCE_DATA_DIR}:/workspace/upstream/data:ro" \
    "${SIF}" \
    python /workspace/run_pipeline.py "${PIPELINE_GROUP}" \
        --config "/workspace/pipeline_configs/${DATASET_NAME}.yaml"

# =============================================================================
# Successful completion
# =============================================================================

echo
echo "================================================================================"
echo "PIPELINE GROUP '${PIPELINE_GROUP}' COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo "End: $(date --iso-8601=seconds)"
echo
echo "Artifacts:"
echo "  Data:        ${DATA_DIR}"
echo "  Checkpoints: ${CHECKPOINT_DIR}"
echo "  Results:     ${RESULTS_DIR}"
echo
echo "Monitoring:"
echo "  ${RESOURCE_LOG}"
echo "  ${GPU_LOG}"
echo
echo "Archive this run when the experiment is complete:"
echo "  ./hpc/archive_run.sh --job-id ${SLURM_JOB_ID} --label <label>"

