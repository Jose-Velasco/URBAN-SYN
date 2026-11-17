#!/usr/bin/env bash
#SBATCH --job-name=urban-syn
#SBATCH --partition=gpuqm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Run the existing URBAN-SYN experiment pipeline inside Apptainer.
# The companion submit.sh places Slurm stdout/stderr in hpc/urban_syn/logs/.
# Usage: sbatch hpc/urban_syn/run_urban_syn.sbatch [stage] [config] [--validate-only|--dry-run]

set -euo pipefail

STAGE="${1:-all}"
CONFIG_ARG="${2:-configs/experiments/nyc_joint.yaml}"
MODE="${3:-}"

usage() {
    cat <<'USAGE'
Usage:
  sbatch hpc/urban_syn/run_urban_syn.sbatch [stage] [config] [mode]

Stages: all | validate | prepare-graph | train | generate | export-eval
Config: Path relative to the repository root (default: configs/experiments/nyc_joint.yaml).
Mode:   --validate-only | --dry-run (optional)

Environment overrides:
  URBAN_SYN_PROJECT_ROOT  Absolute path to URBAN-SYN checkout on HPC.
  URBAN_SYN_SIF           Absolute path to urban-syn.sif on HPC.
USAGE
}

case "$STAGE" in
    all|validate|prepare-graph|train|generate|export-eval) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unsupported stage '$STAGE'" >&2; usage >&2; exit 2 ;;
esac

case "$MODE" in
    ""|--validate-only|--dry-run) ;;
    *) echo "ERROR: Unsupported mode '$MODE'" >&2; usage >&2; exit 2 ;;
esac

if [[ $# -gt 3 ]]; then
    echo "ERROR: Expected at most three arguments." >&2
    usage >&2
    exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script with sbatch, not bash." >&2
    exit 2
fi

# By default, infer the project root from this script's location.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${URBAN_SYN_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd -P)"
SIF="${URBAN_SYN_SIF:-$HOME/containers/urban-syn.sif}"
LOG_DIR="$PROJECT_ROOT/hpc/urban_syn/logs"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Apptainer image not found: $SIF" >&2
    exit 1
fi

if [[ "$CONFIG_ARG" = /* ]]; then
    CONFIG_HOST="$CONFIG_ARG"
else
    CONFIG_HOST="$PROJECT_ROOT/$CONFIG_ARG"
fi

if [[ ! -f "$CONFIG_HOST" ]]; then
    echo "ERROR: Experiment YAML not found: $CONFIG_HOST" >&2
    exit 1
fi

# Resolve symlinks and translate the host YAML path into its container path.
CONFIG_HOST="$(realpath -e -- "$CONFIG_HOST")"
if [[ "$CONFIG_HOST" != "$PROJECT_ROOT/"* ]]; then
    echo "ERROR: Experiment YAML must be inside $PROJECT_ROOT" >&2
    exit 1
fi
CONFIG_CONTAINER="/workspace/${CONFIG_HOST#"$PROJECT_ROOT"/}"

mkdir -p "$LOG_DIR"
RESOURCE_LOG="$LOG_DIR/resources-${SLURM_JOB_ID}.log"
GPU_LOG="$LOG_DIR/gpu-${SLURM_JOB_ID}.csv"
METADATA_LOG="$LOG_DIR/job-${SLURM_JOB_ID}.txt"

{
    echo "URBAN-SYN Slurm job"
    echo "job_id=${SLURM_JOB_ID}"
    echo "job_name=${SLURM_JOB_NAME:-unknown}"
    echo "host=$(hostname)"
    echo "start=$(date --iso-8601=seconds)"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
    echo "gpu_allocation=${SLURM_JOB_GPUS:-unknown}"
    echo "memory_per_node_mb=${SLURM_MEM_PER_NODE:-unknown}"
    echo "project_root=$PROJECT_ROOT"
    echo "container_image=$SIF"
    echo "stage=$STAGE"
    echo "config_host=$CONFIG_HOST"
    echo "config_container=$CONFIG_CONTAINER"
    echo "mode=${MODE:-execute}"
    echo "mlflow_scope=tracking.storage_scope in experiment YAML"
    echo "git_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)"
} | tee "$METADATA_LOG"

echo
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# --cleanenv avoids leaking host Conda/Python settings into /opt/venv.
# Pass the Slurm job ID for MLflow run tags and YAML-driven per-job storage.
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export APPTAINERENV_PYTHONPATH="/workspace/src"

# Preserve Slurm's assigned GPU visibility inside Apptainer.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is unset; cannot verify GPU assignment." >&2
    exit 1
fi
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    export APPTAINERENV_SLURM_ARRAY_TASK_ID="$SLURM_ARRAY_TASK_ID"
fi

APPTAINER_ARGS=(
    exec
    --nv
    --cleanenv
    --bind "$PROJECT_ROOT:/workspace"
    --pwd /workspace
    "$SIF"
)

# Show Slurm's GPU assignment and physical GPU information.
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi --query-gpu=index,uuid,memory.used,memory.total \
    --format=csv

# Verify CUDA visibility and identify the physical GPU used.
apptainer "${APPTAINER_ARGS[@]}" /opt/venv/bin/python -c '
import os
import torch
import pynvml

print("SLURM_JOB_ID:", os.getenv("SLURM_JOB_ID"))
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("Visible GPU count:", torch.cuda.device_count())

if not os.getenv("SLURM_JOB_ID"):
    raise SystemExit("ERROR: Slurm job ID not propagated.")

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("ERROR: Expected exactly one usable CUDA GPU.")

print("Selected GPU:", torch.cuda.get_device_name(0))

# Initialize CUDA so the active process appears in NVML.
torch.ones(1, device="cuda:0")
torch.cuda.synchronize()

pynvml.nvmlInit()
try:
    found = False
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

        if any(process.pid == os.getpid() for process in processes):
            print("Physical GPU index:", index)
            print("Physical GPU UUID:", pynvml.nvmlDeviceGetUUID(handle))
            found = True

    if not found:
        print("WARNING: Could not identify the process GPU through NVML.")
finally:
    pynvml.nvmlShutdown()
'

# Initialize the GPU monitoring CSV exactly once.
printf '%s\n' \
    'timestamp,index,name,gpu_util_percent,memory_used_MiB,memory_total_MiB,temperature_C,power_draw_W' \
    > "$GPU_LOG"

monitor_gpu() {
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits >> "$GPU_LOG" 2>&1 || true
        sleep 30
    done
}

monitor_resources() {
    while true; do
        {
            echo "===== $(date --iso-8601=seconds) ====="
            sstat -j "${SLURM_JOB_ID}.batch" \
                --format=JobID,AveCPU,AveRSS,MaxRSS,AveVMSize,MaxVMSize \
                -n 2>&1 || true
            echo
        } >> "$RESOURCE_LOG"
        sleep 60
    done
}

GPU_MONITOR_PID=""
RESOURCE_MONITOR_PID=""
PIPELINE_PID=""

cleanup() {
    local exit_code=$?
    trap - EXIT
    for pid in "$GPU_MONITOR_PID" "$RESOURCE_MONITOR_PID"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "end=$(date --iso-8601=seconds)" | tee -a "$METADATA_LOG"
    echo "exit_code=$exit_code" | tee -a "$METADATA_LOG"
    exit "$exit_code"
}

forward_signal() {
    local signal="$1"
    echo "Received $signal; forwarding to the running pipeline." >&2
    if [[ -n "$PIPELINE_PID" ]]; then
        kill -s "$signal" "$PIPELINE_PID" 2>/dev/null || true
    fi
    # The training engine saves checkpoints on its configured epoch cadence;
    # signals do not trigger an additional checkpoint.
}

trap cleanup EXIT
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

monitor_gpu &
GPU_MONITOR_PID=$!
monitor_resources &
RESOURCE_MONITOR_PID=$!

COMMAND=(
    /opt/venv/bin/python -m urban_syn.experiment "$STAGE"
    --config "$CONFIG_CONTAINER"
    --project-root /workspace
)
if [[ -n "$MODE" ]]; then
    COMMAND+=("$MODE")
fi

printf '\nStarting experiment: stage=%s, config=%s\n' "$STAGE" "$CONFIG_CONTAINER"
apptainer "${APPTAINER_ARGS[@]}" "${COMMAND[@]}" &
PIPELINE_PID=$!

if wait "$PIPELINE_PID"; then
    echo "Pipeline completed successfully."
else
    exit_code=$?
    echo "Pipeline exited with code $exit_code" >&2
    exit "$exit_code"
fi
