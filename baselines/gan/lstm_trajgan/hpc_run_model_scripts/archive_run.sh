#!/usr/bin/env bash
# Archive LSTM-TrajGAN run-specific artifacts into a single run directory.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  archive_run.sh [OPTIONS]

Options:
  --job-id ID       Slurm job ID associated with the run.
                    May be provided multiple times for a dependency chain.
                    If omitted, SLURM_JOB_ID is used when available.
  --label LABEL     Short descriptive label appended to the run ID.
  --run-id ID       Explicit run ID. Overrides automatic naming.
  --copy             Copy artifacts/logs instead of moving them.
  --help             Show this help message.

Examples:
  ./hpc/archive_run.sh \
      --job-id 12345 \
      --label nyc_full

  ./hpc/archive_run.sh \
      --run-id 20261012_lstm_nyc_smoke \
      --job-id 12345 \
      --job-id 12346 \
      --job-id 12347

  ./hpc/archive_run.sh \
      --run-id 20261012_lstm_nyc_full \
      --job-id 12345 \
      --job-id 12346 \
      --job-id 12347 \
      --copy
USAGE
}

# =============================================================================
# Arguments
# =============================================================================

JOB_IDS=()
LABEL=""
RUN_ID=""
MOVE_ARTIFACTS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-id)
            JOB_IDS+=("$2")
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --copy)
            MOVE_ARTIFACTS=false
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# If no job IDs were supplied explicitly, use the current Slurm job ID
# when the script is being run from inside a Slurm allocation.
if [[ ${#JOB_IDS[@]} -eq 0 && -n "${SLURM_JOB_ID:-}" ]]; then
    JOB_IDS+=("${SLURM_JOB_ID}")
fi

# =============================================================================
# Paths
# =============================================================================

HPC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "${HPC_DIR}/.." && pwd)"
PROJECT="$(cd "${BASE}/../../.." && pwd)"

DATASET_NAME="nyc"

DATA_DIR="${BASE}/data/nyc_full"
OUTPUT_DIR="${BASE}/outputs/lstm_trajgan_nyc_full"
PARAMS_DIR="${BASE}/params"

PIPELINE_CONFIG="${BASE}/pipeline_configs/${DATASET_NAME}.yaml"

SLURM_LOG_DIR="${HPC_DIR}/logs"
RUNS_DIR="${HPC_DIR}/runs"

SIF="/home/${USER}/containers/lstm-trajgan.sif"

# =============================================================================
# Run ID
# =============================================================================

timestamp="$(date +%Y%m%d_%H%M%S)"

if [[ -z "${RUN_ID}" ]]; then
    RUN_ID="${timestamp}"

    if [[ ${#JOB_IDS[@]} -eq 1 ]]; then
        RUN_ID+="_job${JOB_IDS[0]}"
    elif [[ ${#JOB_IDS[@]} -gt 1 ]]; then
        job_suffix="$(IFS=-; echo "${JOB_IDS[*]}")"
        RUN_ID+="_jobs${job_suffix}"
    fi

    if [[ -n "${LABEL}" ]]; then
        safe_label="$(
            printf '%s' "${LABEL}" \
                | tr ' ' '_' \
                | tr -cd '[:alnum:]_.-'
        )"

        RUN_ID+="_${safe_label}"
    fi
fi

RUN_DIR="${RUNS_DIR}/${RUN_ID}"

if [[ -e "${RUN_DIR}" ]]; then
    echo "ERROR: Run directory already exists: ${RUN_DIR}" >&2
    exit 1
fi

# Comma-separated representation used for metadata and sacct.
JOB_IDS_CSV=""

if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    JOB_IDS_CSV="$(IFS=,; echo "${JOB_IDS[*]}")"
fi

# =============================================================================
# Summary
# =============================================================================

echo "================================================================================"
echo "LSTM-TrajGAN Run Archiver"
echo "================================================================================"
echo "Run ID:         ${RUN_ID}"
echo "Run dir:        ${RUN_DIR}"
echo "Project:        ${PROJECT}"
echo "LSTM-TrajGAN:   ${BASE}"
echo "Slurm job IDs:  ${JOB_IDS_CSV:-not supplied}"
echo "Artifact mode:  $([[ "${MOVE_ARTIFACTS}" == true ]] && echo move || echo copy)"
echo "================================================================================"

mkdir -p \
    "${RUN_DIR}/data" \
    "${RUN_DIR}/outputs" \
    "${RUN_DIR}/params" \
    "${RUN_DIR}/logs/slurm" \
    "${RUN_DIR}/configs" \
    "${RUN_DIR}/metadata"

# =============================================================================
# Archive helpers
# =============================================================================

archive_dir() {
    local source="$1"
    local destination="$2"
    local mode="$3"

    if [[ ! -e "${source}" ]]; then
        echo "SKIP: ${source} does not exist."
        return
    fi

    mkdir -p "$(dirname "${destination}")"

    if [[ "${mode}" == "move" ]]; then
        echo "MOVE: ${source}"
        echo "   -> ${destination}"
        mv "${source}" "${destination}"
    else
        echo "COPY: ${source}"
        echo "   -> ${destination}"
        cp -a "${source}" "${destination}"
    fi
}

archive_log() {
    local source="$1"
    local destination_dir="$2"

    [[ -f "${source}" ]] || return

    if [[ "${MOVE_ARTIFACTS}" == true ]]; then
        echo "MOVE LOG: ${source}"
        mv "${source}" "${destination_dir}/"
    else
        echo "COPY LOG: ${source}"
        cp -a "${source}" "${destination_dir}/"
    fi
}

# =============================================================================
# Data and output artifacts
# =============================================================================

if [[ "${MOVE_ARTIFACTS}" == true ]]; then
    archive_dir \
        "${DATA_DIR}" \
        "${RUN_DIR}/data/nyc_full" \
        move

    archive_dir \
        "${OUTPUT_DIR}" \
        "${RUN_DIR}/outputs/lstm_trajgan_nyc_full" \
        move
else
    archive_dir \
        "${DATA_DIR}" \
        "${RUN_DIR}/data/nyc_full" \
        copy

    archive_dir \
        "${OUTPUT_DIR}" \
        "${RUN_DIR}/outputs/lstm_trajgan_nyc_full" \
        copy
fi

# The model JSON architecture files are small and useful for reproducibility.
if [[ "${MOVE_ARTIFACTS}" == true ]]; then
    archive_dir \
        "${PARAMS_DIR}" \
        "${RUN_DIR}/params" \
        move
else
    archive_dir \
        "${PARAMS_DIR}" \
        "${RUN_DIR}/params" \
        copy
fi

# =============================================================================
# Configuration/code snapshots
# =============================================================================

[[ -f "${PIPELINE_CONFIG}" ]] && \
    cp -a \
        "${PIPELINE_CONFIG}" \
        "${RUN_DIR}/configs/pipeline_${DATASET_NAME}.yaml"

[[ -f "${HPC_DIR}/run_lstm_trajgan.sbatch" ]] && \
    cp -a \
        "${HPC_DIR}/run_lstm_trajgan.sbatch" \
        "${RUN_DIR}/configs/"

[[ -f "${HPC_DIR}/submit_full_run.sh" ]] && \
    cp -a \
        "${HPC_DIR}/submit_full_run.sh" \
        "${RUN_DIR}/configs/"

[[ -f "${HPC_DIR}/archive_run.sh" ]] && \
    cp -a \
        "${HPC_DIR}/archive_run.sh" \
        "${RUN_DIR}/configs/"

[[ -f "${BASE}/run_pipeline.py" ]] && \
    cp -a \
        "${BASE}/run_pipeline.py" \
        "${RUN_DIR}/configs/"

[[ -f "${BASE}/build_lstm_trajgan_inputs.py" ]] && \
    cp -a \
        "${BASE}/build_lstm_trajgan_inputs.py" \
        "${RUN_DIR}/configs/"

# =============================================================================
# Slurm/resource/GPU logs
# =============================================================================

if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    shopt -s nullglob

    declare -A seen_logs=()

    for job_id in "${JOB_IDS[@]}"; do
        job_logs=(
            "${SLURM_LOG_DIR}"/*-"${job_id}".out
            "${SLURM_LOG_DIR}"/*-"${job_id}".err
            "${SLURM_LOG_DIR}"/slurm-"${job_id}".out
            "${SLURM_LOG_DIR}"/slurm-"${job_id}".err
            "${SLURM_LOG_DIR}"/resources-"${job_id}".log
            "${SLURM_LOG_DIR}"/gpu-"${job_id}".csv
        )

        for log_file in "${job_logs[@]}"; do
            [[ -f "${log_file}" ]] || continue

            # A glob pattern could theoretically find the same file more than
            # once, so avoid trying to archive a duplicate.
            if [[ -n "${seen_logs["${log_file}"]+x}" ]]; then
                continue
            fi

            seen_logs["${log_file}"]=1

            archive_log \
                "${log_file}" \
                "${RUN_DIR}/logs/slurm"
        done
    done

    shopt -u nullglob
else
    echo "WARNING: No Slurm job IDs supplied; job-specific logs were not archived."
fi

# =============================================================================
# Run metadata
# =============================================================================

{
    echo "run_id=${RUN_ID}"
    echo "dataset=${DATASET_NAME}"
    echo "archived_at=$(date --iso-8601=seconds)"
    echo "archive_host=$(hostname)"
    echo "slurm_job_ids=${JOB_IDS_CSV}"
    echo "project_root=${PROJECT}"
    echo "lstm_trajgan_root=${BASE}"
    echo "container=${SIF}"
    echo

    echo "=== GIT ==="

    if git -C "${PROJECT}" \
        rev-parse --is-inside-work-tree \
        >/dev/null 2>&1
    then
        echo "commit=$(git -C "${PROJECT}" rev-parse HEAD)"

        echo \
            "branch=$(git -C "${PROJECT}" \
                symbolic-ref --quiet --short HEAD \
                2>/dev/null || echo detached)"

        echo
        echo "--- git status --short ---"
        git -C "${PROJECT}" status --short || true

        echo
        echo "--- git diff --stat ---"
        git -C "${PROJECT}" diff --stat || true
    else
        echo "Git metadata unavailable."
    fi

    echo
    echo "=== APPTAINER ==="

    if command -v apptainer >/dev/null 2>&1; then
        apptainer --version || true

        if [[ -f "${SIF}" ]]; then
            echo
            echo "--- apptainer inspect ---"
            apptainer inspect "${SIF}" || true
        fi
    else
        echo "apptainer command unavailable."
    fi

    if [[ ${#JOB_IDS[@]} -gt 0 ]] && command -v sacct >/dev/null 2>&1; then
        echo
        echo "=== SLURM ACCOUNTING ==="

        sacct \
            -j "${JOB_IDS_CSV}" \
            --format=JobID,JobName,Partition,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU,ExitCode \
            || true
    fi
} > "${RUN_DIR}/metadata/run_info.txt"

if git -C "${PROJECT}" rev-parse HEAD >/dev/null 2>&1; then
    git -C "${PROJECT}" rev-parse HEAD \
        > "${RUN_DIR}/metadata/git_commit.txt"
fi

# =============================================================================
# Recreate clean working directories
# =============================================================================

mkdir -p \
    "${DATA_DIR}" \
    "${OUTPUT_DIR}" \
    "${PARAMS_DIR}" \
    "${SLURM_LOG_DIR}"

echo
echo "================================================================================"
echo "Archive complete"
echo "================================================================================"
echo "${RUN_DIR}"
echo
echo "Working directories recreated for the next run."

if [[ "${MOVE_ARTIFACTS}" == true ]]; then
    echo "Archived artifacts and matching job logs were removed from the working directories."
else
    echo "Artifacts and logs were copied; working copies remain in place."
fi
