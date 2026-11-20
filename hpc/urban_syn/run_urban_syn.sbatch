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
# Usage: sbatch hpc/urban_syn/run_urban_syn.sbatch [stage] [config] [mode] [--run-dir DIR] [--checkpoint FILE] [--encoder-checkpoint FILE]

set -euo pipefail

STAGE="${1:-all}"
CONFIG_ARG="${2:-configs/experiments/nyc_joint.yaml}"
MODE=""
RUN_DIR=""
CHECKPOINT=""
ENCODER_CHECKPOINT=""
shift "$(( $# >= 2 ? 2 : $# ))"
# Keep the historical optional third positional mode, with optional paths.
while (($#)); do
    case "$1" in
        --dry-run|--validate-only)
            if [[ -n "$MODE" ]]; then echo "ERROR: Choose only one mode." >&2; exit 2; fi
            MODE="$1"; shift ;;
        --run-dir|--checkpoint|--encoder-checkpoint)
            if (($# < 2)); then echo "ERROR: $1 needs a path." >&2; exit 2; fi
            case "$1" in
                --run-dir) RUN_DIR="$2" ;;
                --checkpoint) CHECKPOINT="$2" ;;
                --encoder-checkpoint) ENCODER_CHECKPOINT="$2" ;;
            esac
            shift 2 ;;
        *) echo "ERROR: Unrecognized option '$1'." >&2; exit 2 ;;
    esac
done

usage() {
    cat <<'USAGE'
Usage:
  sbatch hpc/urban_syn/run_urban_syn.sbatch [stage] [config] [mode] [--run-dir DIR] [--checkpoint FILE] [--encoder-checkpoint FILE]

Joint stages: all | validate | prepare-graph | train | generate | export-eval
Pretraining stages: all | validate | prepare | train | test
Config: Path relative to the repository root (default: configs/experiments/nyc_joint.yaml).
Mode:   --validate-only | --dry-run (optional)

Environment overrides:
  URBAN_SYN_PROJECT_ROOT  Absolute path to URBAN-SYN checkout on HPC.
  URBAN_SYN_SIF           Absolute path to urban-syn.sif on HPC.
  URBAN_SYN_PIPELINE      joint | link_pretrain (normally set by submit.sh).
USAGE
}

case "$MODE" in
    ""|--validate-only|--dry-run) ;;
    *) echo "ERROR: Unsupported mode '$MODE'" >&2; usage >&2; exit 2 ;;
esac

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
    echo "pipeline=${URBAN_SYN_PIPELINE:-detect_from_yaml}"
    echo "study=${URBAN_SYN_STUDY_NAME:-none}"
    echo "study_experiment=${URBAN_SYN_EXPERIMENT_KEY:-none}"
    echo "slurm_time_limit=${SLURM_TIMELIMIT:-unknown}"
    echo "run_dir=${RUN_DIR:-none}"
    echo "checkpoint_override=${CHECKPOINT:-none}"
    echo "encoder_checkpoint=${ENCODER_CHECKPOINT:-none}"
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
export APPTAINERENV_SLURMD_NODENAME="${SLURMD_NODENAME:-$(hostname)}"
export APPTAINERENV_URBAN_SYN_STUDY_NAME="${URBAN_SYN_STUDY_NAME:-}"
export APPTAINERENV_URBAN_SYN_EXPERIMENT_KEY="${URBAN_SYN_EXPERIMENT_KEY:-}"
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

# Resolve the pipeline from the experiment YAML when run directly through sbatch.
# The normal submit.sh entry point already resolved it before allocating resources.
PIPELINE="${URBAN_SYN_PIPELINE:-}"
if [[ -z "$PIPELINE" ]]; then
    PIPELINE="$(apptainer "${APPTAINER_ARGS[@]}" /opt/venv/bin/python \
        /workspace/hpc/urban_syn/resolve_submission.py \
        --project-root /workspace --config "$CONFIG_CONTAINER" --field pipeline)"
fi
case "$PIPELINE" in
    joint)
        PYTHON_MODULE=urban_syn.experiment
        if [[ -n "$ENCODER_CHECKPOINT" ]]; then
            # The repository is mounted at /workspace inside Apptainer.
            if [[ "$ENCODER_CHECKPOINT" == "$PROJECT_ROOT/"* ]]; then
                ENCODER_CHECKPOINT="/workspace/${ENCODER_CHECKPOINT#"$PROJECT_ROOT/"}"
            fi
            if [[ "$ENCODER_CHECKPOINT" != /workspace/* ]]; then
                echo "ERROR: --encoder-checkpoint must be inside the project checkout." >&2
                exit 2
            fi
        fi
        case "$STAGE" in all|validate|prepare-graph|train|generate|export-eval) ;; *) echo "ERROR: Invalid joint stage: $STAGE" >&2; exit 2;; esac
        if [[ -n "$RUN_DIR" || -n "$CHECKPOINT" ]]; then
            echo "ERROR: Joint generation/checkpoint paths are configured in the experiment YAML." >&2
            exit 2
        fi ;;
    link_pretrain)
        if [[ -n "$ENCODER_CHECKPOINT" ]]; then
            echo "ERROR: --encoder-checkpoint is only valid for joint diffusion." >&2
            exit 2
        fi
        PYTHON_MODULE=urban_syn.link_pretrain
        case "$STAGE" in all|validate|prepare|train|test) ;; *) echo "ERROR: Invalid pretraining stage: $STAGE" >&2; exit 2;; esac ;;
    *) echo "ERROR: Unknown experiment pipeline: $PIPELINE" >&2; exit 2 ;;
esac

echo "Pipeline: $PIPELINE | Python module: $PYTHON_MODULE"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv

# Identify the GPU used by the current CUDA process. The JSON is also used
# by shared MLflow tracking to attach GPU and Slurm tags to each run.
GPU_METADATA_HOST="$LOG_DIR/gpu-assignment-${SLURM_JOB_ID}.json"
GPU_METADATA_CONTAINER="/workspace/hpc/urban_syn/logs/gpu-assignment-${SLURM_JOB_ID}.json"
apptainer "${APPTAINER_ARGS[@]}" /opt/venv/bin/python \
    /workspace/hpc/urban_syn/gpu_assignment.py --output "$GPU_METADATA_CONTAINER"
export APPTAINERENV_URBAN_SYN_GPU_METADATA="$GPU_METADATA_CONTAINER"

# Use the assigned GPU UUID for monitoring. A UUID is stable even when CUDA
# device index 0 is mapped to a different physical device by Slurm.
GPU_MONITOR_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("gpu.uuid") or "")' "$GPU_METADATA_HOST" 2>/dev/null || true)"
if [[ -z "$GPU_MONITOR_ID" ]]; then
    echo "WARNING: GPU UUID unavailable; per-device monitoring disabled." >&2
fi

# Initialize the GPU monitoring CSV exactly once.
printf '%s\n' \
    'timestamp,index,name,gpu_util_percent,memory_used_MiB,memory_total_MiB,temperature_C,power_draw_W' \
    > "$GPU_LOG"

monitor_gpu() {
    while true; do
        if [[ -n "$GPU_MONITOR_ID" ]]; then
            nvidia-smi --id="$GPU_MONITOR_ID" \
                --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
                --format=csv,noheader,nounits >> "$GPU_LOG" 2>&1 || true
        fi
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

if [[ -n "$GPU_MONITOR_ID" ]]; then
    monitor_gpu &
    GPU_MONITOR_PID=$!
fi
monitor_resources &
RESOURCE_MONITOR_PID=$!

# Link pretraining expects the same PyG cache as joint diffusion. Reuse the
# joint pipeline's existing graph builder if the cache is missing; this occurs
# only for real executions, never during dry-run or read-only validation.
if [[ "$PIPELINE" == link_pretrain && -z "$MODE" && "$STAGE" != validate ]]; then
    apptainer "${APPTAINER_ARGS[@]}" /opt/venv/bin/python \
        /workspace/hpc/urban_syn/prepare_pretrain_graph.py \
        --config "$CONFIG_CONTAINER" --project-root /workspace
fi

COMMAND=(
    /opt/venv/bin/python -m "$PYTHON_MODULE" "$STAGE"
    --config "$CONFIG_CONTAINER"
    --project-root /workspace
)
if [[ -n "$MODE" ]]; then COMMAND+=("$MODE"); fi
if [[ -n "$RUN_DIR" ]]; then COMMAND+=(--run-dir "$RUN_DIR"); fi
if [[ -n "$CHECKPOINT" ]]; then COMMAND+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "$ENCODER_CHECKPOINT" ]]; then COMMAND+=(--encoder-checkpoint "$ENCODER_CHECKPOINT"); fi

# The pretraining runner reports its exact run folder only after success.
# We publish this source for later study experiments, without guessing dates.
RESULT_FILE_HOST=""
if [[ "$PIPELINE" == link_pretrain && -n "${URBAN_SYN_STUDY_NAME:-}" && \
      -n "${URBAN_SYN_EXPERIMENT_KEY:-}" && -z "$MODE" && \
      ( "$STAGE" == all || "$STAGE" == train ) ]]; then
    RESULT_FILE_HOST="$LOG_DIR/pretrain-result-${SLURM_JOB_ID}.json"
    rm -f -- "$RESULT_FILE_HOST"
    COMMAND+=(--result-file "/workspace/hpc/urban_syn/logs/pretrain-result-${SLURM_JOB_ID}.json")
fi

printf '\nStarting experiment: pipeline=%s, stage=%s, config=%s\n' "$PIPELINE" "$STAGE" "$CONFIG_CONTAINER"
apptainer "${APPTAINER_ARGS[@]}" "${COMMAND[@]}" &
PIPELINE_PID=$!

if wait "$PIPELINE_PID"; then
    # The study reference changes only after a successful train or all job.
    if [[ -n "$RESULT_FILE_HOST" ]]; then
        python3 "$SCRIPT_DIR/study_checkpoints.py" \
            --project-root "$PROJECT_ROOT" \
            --study-name "$URBAN_SYN_STUDY_NAME" \
            --experiment-key "$URBAN_SYN_EXPERIMENT_KEY" \
            --result-file "$RESULT_FILE_HOST" \
            --job-id "$SLURM_JOB_ID"
    fi
    echo "Pipeline completed successfully."
else
    exit_code=$?
    echo "Pipeline exited with code $exit_code" >&2
    exit "$exit_code"
fi
