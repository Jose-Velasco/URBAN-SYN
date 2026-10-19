#!/bin/bash
#SBATCH --job-name=ts-trajgen
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Email notifications.
#SBATCH --mail-user=jose.velascojuarez@sjsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Keep Slurm output/error separate and identifiable by job name + job ID.
#SBATCH --output=/home/016184753/URBAN-SYN/baselines/gan/ts_trajgen/hpc/logs/%x-%j.out
#SBATCH --error=/home/016184753/URBAN-SYN/baselines/gan/ts_trajgen/hpc/logs/%x-%j.err

# Run from the TS-TrajGen wrapper directory.
#SBATCH --chdir=/home/016184753/URBAN-SYN/baselines/gan/ts_trajgen


set -euo pipefail


# =============================================================================
# Experiment configuration
# =============================================================================

# Dataset being processed.
DATASET_NAME="nyc"

# Pipeline group passed to run_pipeline.py.
#
# Usage:
#   sbatch hpc/run_ts_trajgen.sbatch all
#   sbatch hpc/run_ts_trajgen.sbatch setup-gans
#   sbatch hpc/run_ts_trajgen.sbatch train-road
#   sbatch hpc/run_ts_trajgen.sbatch train-region
#   sbatch hpc/run_ts_trajgen.sbatch train
#   sbatch hpc/run_ts_trajgen.sbatch generate
#
# Defaults to "all" when no argument is supplied.
PIPELINE_GROUP="${1:-all}"


# =============================================================================
# Paths
# =============================================================================

PROJECT="/home/016184753/URBAN-SYN"
BASE="${PROJECT}/baselines/gan/ts_trajgen"
HPC_DIR="${BASE}/hpc"

SIF="/home/016184753/containers/ts-trajgen.sif"

PIPELINE_CONFIG="${BASE}/pipeline_configs/${DATASET_NAME}.yaml"
MODEL_CONFIG="${BASE}/repo/configs/ts_trajgen_${DATASET_NAME}.yaml"

DATASET_DIR="${BASE}/datasets/${DATASET_NAME}"
SAVE_DIR="${BASE}/repo/save/${DATASET_NAME}"
GAN_SAVE_DIR="${SAVE_DIR}/gan"
REGION_GAN_SAVE_DIR="${SAVE_DIR}/region_gan"
TEMP_DIR="${BASE}/repo/temp/${DATASET_NAME}"

# Keep this synchronized with gan.exp_id in the pipeline YAML.
GAN_EXP_ID=1

RESOURCE_LOG="${HPC_DIR}/logs/resources-${SLURM_JOB_ID}.log"
GPU_LOG="${HPC_DIR}/logs/gpu-${SLURM_JOB_ID}.csv"


# =============================================================================
# Pipeline group validation
# =============================================================================

case "${PIPELINE_GROUP}" in
    all|prepare|pretrain-road|prepare-region|pretrain-region|setup-gans|train-road|train-region|train|generate)
        ;;
    *)
        echo "ERROR: Invalid pipeline group: ${PIPELINE_GROUP}"
        echo
        echo "Valid groups:"
        echo "  all"
        echo "  prepare"
        echo "  pretrain-road"
        echo "  prepare-region"
        echo "  pretrain-region"
        echo "  train"
        echo "  generate"
        echo
        echo "Example:"
        echo "  sbatch run_ts_trajgen.sbatch train"
        exit 1
        ;;
esac


# =============================================================================
# Basic safety checks
# =============================================================================

check_empty_directory() {
    local dir="$1"

    if [[ -d "${dir}" ]] && \
       [[ -n "$(find "${dir}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then

        echo "ERROR: Run artifact directory is not empty:"
        echo "  ${dir}"
        echo
        echo "Archive or remove the previous run before starting a clean run."
        exit 1
    fi
}


require_file() {
    local file="$1"

    if [[ ! -f "${file}" ]]; then
        echo "ERROR: Required file does not exist:"
        echo "  ${file}"
        exit 1
    fi
}


# Always verify the core configuration/container inputs.
require_file "${SIF}"
require_file "${PIPELINE_CONFIG}"
require_file "${MODEL_CONFIG}"

mkdir -p "${HPC_DIR}/logs"


# Full runs and GAN setup runs are expected to start from clean run directories.
# setup-gans executes all preprocessing and road/region pretraining stages, but
# stops before adversarial GAN training.
if [[ "${PIPELINE_GROUP}" == "all" || "${PIPELINE_GROUP}" == "setup-gans" ]]; then
    echo "Checking clean-run requirements for pipeline group '${PIPELINE_GROUP}'..."

    check_empty_directory "${DATASET_DIR}"
    check_empty_directory "${SAVE_DIR}"
fi


# Road GAN training requires the completed road-level pretraining models.
if [[ "${PIPELINE_GROUP}" == "train-road" ]]; then
    echo "Checking pretrained model requirements for pipeline group 'train-road'..."

    require_file "${SAVE_DIR}/function_g_fc.pt"
    require_file "${SAVE_DIR}/gat_fc.pt"

    echo "All road GAN training requirements found."
fi


# Region GAN training is independent of the trained road GAN, but requires the
# completed region-level pretraining models.
if [[ "${PIPELINE_GROUP}" == "train-region" ]]; then
    echo "Checking pretrained model requirements for pipeline group 'train-region'..."

    require_file "${SAVE_DIR}/region_function_g_fc.pt"
    require_file "${SAVE_DIR}/region_gat_fc.pt"

    echo "All region GAN training requirements found."
fi


# The convenience target 'train' runs road GAN training followed by region GAN
# training, so all four pretraining checkpoints must already exist.
if [[ "${PIPELINE_GROUP}" == "train" ]]; then
    echo "Checking pretrained model requirements for pipeline group 'train'..."

    require_file "${SAVE_DIR}/function_g_fc.pt"
    require_file "${SAVE_DIR}/gat_fc.pt"
    require_file "${SAVE_DIR}/region_function_g_fc.pt"
    require_file "${SAVE_DIR}/region_gat_fc.pt"

    echo "All GAN training requirements found."
fi


# GAN-based generation requires both completed adversarial generators.
if [[ "${PIPELINE_GROUP}" == "generate" ]]; then
    echo "Checking GAN requirements for pipeline group 'generate'..."

    require_file "${GAN_SAVE_DIR}/adversarial_3_generator_${GAN_EXP_ID}.pt"
    require_file "${REGION_GAN_SAVE_DIR}/adversarial_region_generator.pt"

    echo "All GAN generation requirements found."
fi


# =============================================================================
# Job information
# =============================================================================

echo
echo "================================================================================"
echo "TS-TrajGen"
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
echo "Model config:         ${MODEL_CONFIG}"
echo "Container:            ${SIF}"
echo
echo "CPUs per task:        ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory per node:      ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "GPUs:                 ${SLURM_GPUS:-unknown}"
echo "GRES:                 ${SLURM_JOB_GPUS:-unknown}"
echo "================================================================================"


echo
echo "=== GPU AT START ==="
nvidia-smi


# =============================================================================
# Record relevant configuration
# =============================================================================

echo
echo "=== PIPELINE CONFIG SETTINGS ==="

grep -nE \
    'max_train_trajectories|max_test_trajectories|debug|generation' \
    "${PIPELINE_CONFIG}" \
    || true


echo
echo "=== MODEL CONFIG TRAINING SETTINGS ==="

# Record the most relevant training parameters in the Slurm log so that the
# effective experiment settings can be recovered without opening the YAML.
grep -niE \
    'epoch|max_epoch|epochs|batch_size|dis_sample_num|gen_sample_num|rollout_times' \
    "${MODEL_CONFIG}" \
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
echo "STARTING TS-TRAJGEN PIPELINE"
echo "================================================================================"
echo "Dataset:        ${DATASET_NAME}"
echo "Pipeline group: ${PIPELINE_GROUP}"
echo "================================================================================"
echo


apptainer exec --nv \
    --bind "${BASE}:/workspace" \
    --bind "${PROJECT}/fmm_scripts/data:/workspace/upstream/fmm/data:ro" \
    --bind "${PROJECT}/fmm_scripts/outputs:/workspace/upstream/fmm/outputs:ro" \
    --bind "${PROJECT}/data/${DATASET_NAME}_output_tabular/output:/workspace/upstream/data/${DATASET_NAME}_output_tabular/output:ro" \
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
echo "  Dataset: ${DATASET_DIR}"
echo "  Models:  ${SAVE_DIR}"
echo
echo "Monitoring:"
echo "  ${RESOURCE_LOG}"
echo "  ${GPU_LOG}"
echo
echo "Archive this run when the experiment is complete:"
echo
echo "  ./hpc/archive_run.sh --job-id ${SLURM_JOB_ID} --label <label>"
