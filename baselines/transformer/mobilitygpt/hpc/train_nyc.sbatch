#!/usr/bin/env bash
#SBATCH --job-name=mobilitygpt-nyc
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

die() {
    printf 'train_nyc.sbatch: error: %s\n' "$*" >&2
    exit 2
}

# This wrapper is intentionally submitted from the outer mobilitygpt directory.
PROJECT_DIR=$(pwd -P)
REPO_DIR=${PROJECT_DIR}/repo
IMAGE=${PROJECT_DIR}/mobilitygpt.sif
INPUT_HOST_DIR=${INPUT_DIR:-${REPO_DIR}/NYC}
CONTAINER_REPO=/workspace/mobilitygpt/repo
CONTAINER_INPUT=${CONTAINER_REPO}/NYC

[[ -d ${REPO_DIR} ]] || die "missing source directory: ${REPO_DIR} (submit from the outer mobilitygpt/ directory)"
[[ -f ${IMAGE} ]] || die "missing Apptainer image: ${IMAGE}"
[[ -d ${INPUT_HOST_DIR} ]] || die "missing NYC input directory: ${INPUT_HOST_DIR}"
[[ -d ${PROJECT_DIR}/logs ]] || die "missing logs/; create it before submission: mkdir -p logs"

for input_name in nyc.geo nyc.rel nyc_mm_train.csv nyc_mm_test.csv; do
    case ${input_name} in
        nyc.geo) aliases=(nyc.geo roadmap.geo) ;;
        nyc.rel) aliases=(nyc.rel roadmap.rel) ;;
        nyc_mm_train.csv) aliases=(nyc_mm_train.csv NYC_Taxi_trajectory_train.csv) ;;
        nyc_mm_test.csv) aliases=(nyc_mm_test.csv NYC_Taxi_trajectory_test.csv) ;;
    esac
    input_found=false
    for candidate in "${aliases[@]}"; do
        if [[ -f ${INPUT_HOST_DIR}/${candidate} ]]; then
            input_found=true
            break
        fi
    done
    [[ ${input_found} == true ]] || die "missing NYC input ${input_name} (accepted alias: ${aliases[1]}) in ${INPUT_HOST_DIR}"
done

if command -v apptainer >/dev/null 2>&1; then
    RUNTIME=$(command -v apptainer)
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME=$(command -v singularity)
else
    die 'neither apptainer nor singularity is available on the compute node'
fi

printf 'node=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID:-unset}"
printf 'assigned_gpus=%s\n' "${SLURM_JOB_GPUS:-unset}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
else
    printf 'nvidia-smi is unavailable on the host; CUDA validation will be authoritative.\n'
fi
printf 'runtime=%s\nimage=%s\nrepo=%s\ninput_host=%s\n' "$RUNTIME" "$IMAGE" "$REPO_DIR" "$INPUT_HOST_DIR"

BIND_ARGS=(--bind "${REPO_DIR}:${CONTAINER_REPO}")
# Mount raw inputs read-only at the path expected by the workflow, including
# when INPUT_DIR points to a separate shared-storage location.
BIND_ARGS+=(--bind "${INPUT_HOST_DIR}:${CONTAINER_INPUT}:ro")

"${RUNTIME}" exec --nv --pwd "${CONTAINER_REPO}" "${BIND_ARGS[@]}" "${IMAGE}" \
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available inside the container"; x=torch.ones((128,128), device="cuda"); y=x @ x; torch.cuda.synchronize(); print(f"CUDA ok: {torch.cuda.get_device_name(0)}; result={y[0,0].item():.1f}")'

# Keep all workflow arguments after the fixed input path unchanged.  In
# particular, this preserves --smoke, output/resume, network, and stage flags.
"${RUNTIME}" exec --nv --pwd "${CONTAINER_REPO}" "${BIND_ARGS[@]}" "${IMAGE}" \
    python -m nyc_workflow run --input-dir NYC "$@"
