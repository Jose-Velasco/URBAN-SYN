#!/usr/bin/env bash
# Archive TS-TrajGen run-specific artifacts into a single run directory.
# Intended location: baselines/gan/ts_trajgen/hpc/archive_run.sh
# Example: ./hpc/archive_run.sh --job-id 87755 --label drive_smoke

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  archive_run.sh [OPTIONS]

Options:
  --job-id ID       Slurm job ID associated with the run.
                    If omitted, SLURM_JOB_ID is used when available.
  --label LABEL     Short descriptive label appended to the run ID.
  --run-id ID       Explicit run ID. Overrides automatic naming.
  --copy             Copy large artifacts instead of moving them.
  --help             Show this help message.

Examples:
  ./hpc/archive_run.sh --job-id 87755 --label drive_smoke
  ./hpc/archive_run.sh --run-id 20261010_full_drive_p100 --job-id 87760

Default run ID:
  YYYYMMDD_HHMMSS[_job<id>][_label]
USAGE
}

JOB_ID="${SLURM_JOB_ID:-}"
LABEL=""
RUN_ID=""
MOVE_ARTIFACTS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-id)
            JOB_ID="$2"
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

HPC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "${HPC_DIR}/.." && pwd)"
PROJECT="$(cd "${BASE}/../../.." && pwd)"
DATASET_NAME="nyc"

DATASET_DIR="${BASE}/datasets/${DATASET_NAME}"
DATASET_LOG_DIR="${BASE}/datasets/logs"
SAVE_DIR="${BASE}/repo/save/${DATASET_NAME}"
TEMP_DIR="${BASE}/repo/temp/${DATASET_NAME}"
UPSTREAM_LOG_DIR="${BASE}/repo/log"
PIPELINE_CONFIG="${BASE}/pipeline_configs/${DATASET_NAME}.yaml"
MODEL_CONFIG="${BASE}/repo/configs/ts_trajgen_${DATASET_NAME}.yaml"
SLURM_LOG_DIR="${HPC_DIR}/logs"
RUNS_DIR="${HPC_DIR}/runs"
SIF="/home/${USER}/containers/ts-trajgen.sif"

timestamp="$(date +%Y%m%d_%H%M%S)"

if [[ -z "${RUN_ID}" ]]; then
    RUN_ID="${timestamp}"
    [[ -n "${JOB_ID}" ]] && RUN_ID+="_job${JOB_ID}"
    if [[ -n "${LABEL}" ]]; then
        safe_label="$(printf '%s' "${LABEL}" | tr ' ' '_' | tr -cd '[:alnum:]_.-')"
        RUN_ID+="_${safe_label}"
    fi
fi

RUN_DIR="${RUNS_DIR}/${RUN_ID}"

if [[ -e "${RUN_DIR}" ]]; then
    echo "ERROR: Run directory already exists: ${RUN_DIR}" >&2
    exit 1
fi

echo "================================================================================"
echo "TS-TrajGen Run Archiver"
echo "================================================================================"
echo "Run ID:       ${RUN_ID}"
echo "Run dir:      ${RUN_DIR}"
echo "Project:      ${PROJECT}"
echo "TS-TrajGen:   ${BASE}"
echo "Slurm job ID: ${JOB_ID:-not supplied}"
echo "Large files:  $([[ "${MOVE_ARTIFACTS}" == true ]] && echo move || echo copy)"
echo "================================================================================"

mkdir -p \
    "${RUN_DIR}/datasets" \
    "${RUN_DIR}/save" \
    "${RUN_DIR}/temp" \
    "${RUN_DIR}/logs/slurm" \
    "${RUN_DIR}/logs/dataset" \
    "${RUN_DIR}/logs/upstream" \
    "${RUN_DIR}/configs" \
    "${RUN_DIR}/metadata"

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

# Core run artifacts.
if [[ "${MOVE_ARTIFACTS}" == true ]]; then
    archive_dir "${DATASET_DIR}" "${RUN_DIR}/datasets/${DATASET_NAME}" move
    archive_dir "${SAVE_DIR}" "${RUN_DIR}/save/${DATASET_NAME}" move
else
    archive_dir "${DATASET_DIR}" "${RUN_DIR}/datasets/${DATASET_NAME}" copy
    archive_dir "${SAVE_DIR}" "${RUN_DIR}/save/${DATASET_NAME}" copy
fi

# Troubleshooting artifacts. These are moved so runs do not mix together.
archive_dir "${DATASET_LOG_DIR}" "${RUN_DIR}/logs/dataset" move
archive_dir "${TEMP_DIR}" "${RUN_DIR}/temp/${DATASET_NAME}" move
archive_dir "${UPSTREAM_LOG_DIR}" "${RUN_DIR}/logs/upstream" move

# Config snapshots.
[[ -f "${PIPELINE_CONFIG}" ]] && cp -a "${PIPELINE_CONFIG}" "${RUN_DIR}/configs/pipeline_${DATASET_NAME}.yaml"
[[ -f "${MODEL_CONFIG}" ]] && cp -a "${MODEL_CONFIG}" "${RUN_DIR}/configs/model_${DATASET_NAME}.yaml"
[[ -f "${HPC_DIR}/run_ts_trajgen.sbatch" ]] && cp -a "${HPC_DIR}/run_ts_trajgen.sbatch" "${RUN_DIR}/configs/"

# Job-specific Slurm / monitoring logs.
if [[ -n "${JOB_ID}" ]]; then
    shopt -s nullglob
    job_logs=(
        "${SLURM_LOG_DIR}"/*-"${JOB_ID}".out
        "${SLURM_LOG_DIR}"/*-"${JOB_ID}".err
        "${SLURM_LOG_DIR}"/slurm-"${JOB_ID}".out
        "${SLURM_LOG_DIR}"/slurm-"${JOB_ID}".err
        "${SLURM_LOG_DIR}"/resources-"${JOB_ID}".log
        "${SLURM_LOG_DIR}"/gpu-"${JOB_ID}".csv
    )

    declare -A seen_logs=()
    for log_file in "${job_logs[@]}"; do
        [[ -f "${log_file}" ]] || continue
        [[ -n "${seen_logs["${log_file}"]+x}" ]] && continue
        seen_logs["${log_file}"]=1
        cp -a "${log_file}" "${RUN_DIR}/logs/slurm/"
    done
    shopt -u nullglob
else
    echo "WARNING: No Slurm job ID supplied; job-specific Slurm/resource logs were not copied."
fi

# Reproducibility metadata.
{
    echo "run_id=${RUN_ID}"
    echo "dataset=${DATASET_NAME}"
    echo "archived_at=$(date --iso-8601=seconds)"
    echo "archive_host=$(hostname)"
    echo "slurm_job_id=${JOB_ID:-}"
    echo "project_root=${PROJECT}"
    echo "ts_trajgen_root=${BASE}"
    echo "container=${SIF}"
    echo

    echo "=== GIT ==="
    if git -C "${PROJECT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "commit=$(git -C "${PROJECT}" rev-parse HEAD)"
        echo "branch=$(git -C "${PROJECT}" symbolic-ref --quiet --short HEAD 2>/dev/null || echo detached)"
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

    if [[ -n "${JOB_ID}" ]] && command -v sacct >/dev/null 2>&1; then
        echo
        echo "=== SLURM ACCOUNTING ==="
        sacct \
            -j "${JOB_ID}" \
            --format=JobID,JobName,Partition,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU,ExitCode \
            || true
    fi
} > "${RUN_DIR}/metadata/run_info.txt"

if git -C "${PROJECT}" rev-parse HEAD >/dev/null 2>&1; then
    git -C "${PROJECT}" rev-parse HEAD > "${RUN_DIR}/metadata/git_commit.txt"
fi

# Recreate normal upstream-compatible working directories for the next run.
mkdir -p \
    "${DATASET_DIR}" \
    "${DATASET_LOG_DIR}" \
    "${SAVE_DIR}" \
    "${TEMP_DIR}" \
    "${UPSTREAM_LOG_DIR}" \
    "${SLURM_LOG_DIR}"

echo
echo "================================================================================"
echo "Archive complete"
echo "================================================================================"
echo "${RUN_DIR}"
echo "Working directories recreated for the next run."
