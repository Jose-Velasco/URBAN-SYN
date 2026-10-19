#!/usr/bin/env bash
# Archive TS-TrajGen artifacts and logs from one or more Slurm jobs into one run.
# Install at: baselines/gan/ts_trajgen/hpc/archive_run.sh

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./hpc/archive_run.sh [OPTIONS]

Options:
  --job-id ID       Slurm job ID; repeat for each job in a dependency chain.
                    If omitted, uses SLURM_JOB_ID when available.
  --label LABEL     Short label appended to the automatic run ID.
  --run-id ID       Explicit archive directory name (must not already exist).
  --copy            Copy ALL artifacts and matched logs instead of moving them.
  --help            Show this help message.

Examples:
  ./hpc/archive_run.sh --job-id 12345 --label smoke
  ./hpc/archive_run.sh --run-id nyc_full_1 \
      --job-id 12345 --job-id 12346 --job-id 12347 --job-id 12348
  ./hpc/archive_run.sh --run-id nyc_full_1 \
      --job-id 12345 --job-id 12346 --job-id 12347 --job-id 12348 --copy

Default run ID:
  YYYYMMDD_HHMMSS[_job<id>|_jobs<id>-<id>...][_label]

Run this AFTER all selected jobs finish. Only their job-specific Slurm,
GPU and resource logs are selected; other log files are left untouched.
USAGE
}

# Preserve the job IDs from an entire Slurm dependency chain.
JOB_IDS=()
LABEL=""
RUN_ID=""
MOVE_ARTIFACTS=true

add_job_id() {
    local job_id="$1"
    local existing

    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid Slurm job ID: ${job_id}" >&2
        exit 2
    fi

    # Avoid processing a job's logs twice if its ID was repeated.
    for existing in "${JOB_IDS[@]}"; do
        [[ "${existing}" == "${job_id}" ]] && return
    done
    JOB_IDS+=("${job_id}")
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-id)
            [[ $# -ge 2 ]] || { echo "ERROR: --job-id needs a value" >&2; exit 2; }
            add_job_id "$2"
            shift 2
            ;;
        --label)
            [[ $# -ge 2 ]] || { echo "ERROR: --label needs a value" >&2; exit 2; }
            LABEL="$2"
            shift 2
            ;;
        --run-id)
            [[ $# -ge 2 ]] || { echo "ERROR: --run-id needs a value" >&2; exit 2; }
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

# Only fall back to the current Slurm job when no IDs were supplied explicitly.
if [[ ${#JOB_IDS[@]} -eq 0 && -n "${SLURM_JOB_ID:-}" ]]; then
    add_job_id "${SLURM_JOB_ID}"
fi

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
    if [[ ${#JOB_IDS[@]} -eq 1 ]]; then
        RUN_ID+="_job${JOB_IDS[0]}"
    elif [[ ${#JOB_IDS[@]} -gt 1 ]]; then
        job_suffix="$(IFS=-; echo "${JOB_IDS[*]}")"
        RUN_ID+="_jobs${job_suffix}"
    fi
    if [[ -n "${LABEL}" ]]; then
        safe_label="$(printf '%s' "${LABEL}" | tr ' ' '_' | tr -cd '[:alnum:]_.-')"
        [[ -n "${safe_label}" ]] && RUN_ID+="_${safe_label}"
    fi
fi

# The archive directory must stay under hpc/runs/.
if [[ ! "${RUN_ID}" =~ ^[[:alnum:]_.-]+$ || "${RUN_ID}" == "." || "${RUN_ID}" == ".." ]]; then
    echo "ERROR: Invalid run ID: ${RUN_ID}" >&2
    exit 2
fi

RUN_DIR="${RUNS_DIR}/${RUN_ID}"
if [[ -e "${RUN_DIR}" ]]; then
    echo "ERROR: Run directory already exists: ${RUN_DIR}" >&2
    exit 1
fi

JOB_IDS_CSV=""
if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    JOB_IDS_CSV="$(IFS=,; echo "${JOB_IDS[*]}")"
fi

printf '%s\n' \
    '================================================================================' \
    'TS-TrajGen Run Archiver' \
    '================================================================================' \
    "Run ID:         ${RUN_ID}" \
    "Run dir:        ${RUN_DIR}" \
    "Slurm job IDs:  ${JOB_IDS_CSV:-not supplied}" \
    "Artifact mode:  $([[ "${MOVE_ARTIFACTS}" == true ]] && echo move || echo copy)" \
    '================================================================================'

mkdir -p \
    "${RUN_DIR}/datasets" \
    "${RUN_DIR}/save" \
    "${RUN_DIR}/temp" \
    "${RUN_DIR}/logs/slurm" \
    "${RUN_DIR}/logs/dataset" \
    "${RUN_DIR}/logs/upstream" \
    "${RUN_DIR}/configs" \
    "${RUN_DIR}/metadata"

# Preserve the existing TS-TrajGen archive directory layout.
archive_dir() {
    local source="$1"
    local destination="$2"

    if [[ ! -e "${source}" ]]; then
        echo "SKIP: ${source} does not exist."
        return
    fi

    mkdir -p "$(dirname "${destination}")"
    if [[ "${MOVE_ARTIFACTS}" == true ]]; then
        echo "MOVE: ${source} -> ${destination}"
        mv "${source}" "${destination}"
    else
        echo "COPY: ${source} -> ${destination}"
        cp -a "${source}" "${destination}"
    fi
}

archive_log() {
    local source="$1"

    [[ -f "${source}" ]] || return
    if [[ "${MOVE_ARTIFACTS}" == true ]]; then
        echo "MOVE LOG: ${source}"
        mv "${source}" "${RUN_DIR}/logs/slurm/"
    else
        echo "COPY LOG: ${source}"
        cp -a "${source}" "${RUN_DIR}/logs/slurm/"
    fi
}

# Dataset, checkpoints, temp data, and non-Slurm logs follow the same move/copy mode.
archive_dir "${DATASET_DIR}" "${RUN_DIR}/datasets/${DATASET_NAME}"
archive_dir "${SAVE_DIR}" "${RUN_DIR}/save/${DATASET_NAME}"
archive_dir "${DATASET_LOG_DIR}" "${RUN_DIR}/logs/dataset"
archive_dir "${TEMP_DIR}" "${RUN_DIR}/temp/${DATASET_NAME}"
archive_dir "${UPSTREAM_LOG_DIR}" "${RUN_DIR}/logs/upstream"

# Snapshot the configuration and orchestration code used to reproduce this run.
[[ -f "${PIPELINE_CONFIG}" ]] && cp -a "${PIPELINE_CONFIG}" "${RUN_DIR}/configs/pipeline_${DATASET_NAME}.yaml"
[[ -f "${MODEL_CONFIG}" ]] && cp -a "${MODEL_CONFIG}" "${RUN_DIR}/configs/model_${DATASET_NAME}.yaml"
for script_name in run_ts_trajgen.sbatch submit_full_run.sh archive_run.sh; do
    [[ -f "${HPC_DIR}/${script_name}" ]] && cp -a "${HPC_DIR}/${script_name}" "${RUN_DIR}/configs/"
done
[[ -f "${BASE}/run_pipeline.py" ]] && cp -a "${BASE}/run_pipeline.py" "${RUN_DIR}/configs/"
[[ -f "${BASE}/build_tstrajgen_inputs.py" ]] && cp -a "${BASE}/build_tstrajgen_inputs.py" "${RUN_DIR}/configs/"

# Only archive the Slurm, CPU/resource, and GPU logs matching the supplied job IDs.
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

        matched=0
        for log_file in "${job_logs[@]}"; do
            [[ -f "${log_file}" ]] || continue
            [[ -n "${seen_logs["${log_file}"]+x}" ]] && continue
            seen_logs["${log_file}"]=1
            archive_log "${log_file}"
            matched=1
        done
        [[ "${matched}" -eq 1 ]] || echo "WARNING: No matching logs found for job ${job_id}." >&2
    done
    shopt -u nullglob
else
    echo "WARNING: No job IDs supplied; no job-specific logs were archived." >&2
fi

# Capture every selected job in the metadata and Slurm accounting output.
{
    echo "run_id=${RUN_ID}"
    echo "dataset=${DATASET_NAME}"
    echo "archived_at=$(date --iso-8601=seconds)"
    echo "archive_host=$(hostname)"
    echo "slurm_job_ids=${JOB_IDS_CSV}"
    echo "project_root=${PROJECT}"
    echo "ts_trajgen_root=${BASE}"
    echo "container=${SIF}"
    echo

    echo '=== GIT ==='
    if git -C "${PROJECT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "commit=$(git -C "${PROJECT}" rev-parse HEAD)"
        echo "branch=$(git -C "${PROJECT}" symbolic-ref --quiet --short HEAD 2>/dev/null || echo detached)"
        echo
        echo '--- git status --short ---'
        git -C "${PROJECT}" status --short || true
        echo
        echo '--- git diff --stat ---'
        git -C "${PROJECT}" diff --stat || true
    else
        echo 'Git metadata unavailable.'
    fi
    echo

    echo '=== APPTAINER ==='
    if command -v apptainer >/dev/null 2>&1; then
        apptainer --version || true
        if [[ -f "${SIF}" ]]; then
            echo
            echo '--- apptainer inspect ---'
            apptainer inspect "${SIF}" || true
        fi
    else
        echo 'apptainer command unavailable.'
    fi

    if [[ ${#JOB_IDS[@]} -gt 0 ]] && command -v sacct >/dev/null 2>&1; then
        echo
        echo '=== SLURM ACCOUNTING ==='
        sacct \
            -j "${JOB_IDS_CSV}" \
            --format=JobID,JobName,Partition,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU,ExitCode \
            || true
    fi
} > "${RUN_DIR}/metadata/run_info.txt"

if git -C "${PROJECT}" rev-parse HEAD >/dev/null 2>&1; then
    git -C "${PROJECT}" rev-parse HEAD > "${RUN_DIR}/metadata/git_commit.txt"
fi

# Restore upstream-compatible working directories for subsequent experiments.
mkdir -p \
    "${DATASET_DIR}" \
    "${DATASET_LOG_DIR}" \
    "${SAVE_DIR}" \
    "${TEMP_DIR}" \
    "${UPSTREAM_LOG_DIR}" \
    "${SLURM_LOG_DIR}"

printf '\nArchive complete: %s\n' "${RUN_DIR}"
if [[ "${MOVE_ARTIFACTS}" == true ]]; then
    echo 'Artifacts and selected logs were moved; working directories were recreated.'
else
    echo 'Artifacts and selected logs were copied; working files remain in place.'
fi
