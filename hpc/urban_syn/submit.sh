#!/usr/bin/env bash
# Submit one joint-diffusion or link-pretraining job using the shared Slurm runner.
# Existing usage remains: submit.sh [stage] [experiment.yaml] [--dry-run|--validate-only]
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash hpc/urban_syn/submit.sh [stage] [config.yaml] [--dry-run|--validate-only] [--run-dir DIR] [--checkpoint FILE]
  bash hpc/urban_syn/submit.sh --study STUDY.yaml --experiment NAME [--stage STAGE] [--dry-run|--validate-only] [--run-dir DIR] [--checkpoint FILE]

The study YAML provides default Slurm resources. Environment overrides take priority:
  SBATCH_PARTITION, SBATCH_GRES, SBATCH_TIMELIMIT, SBATCH_MEM_PER_NODE, SBATCH_CPUS_PER_TASK
  URBAN_SYN_MAIL_USER and URBAN_SYN_MAIL_TYPE (e.g. BEGIN,END,FAIL)

--run-dir and --checkpoint are supported for standalone link-pretraining stages.
For joint generation, select generation.checkpoint in the joint experiment YAML.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${URBAN_SYN_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd -P)"
SIF="${URBAN_SYN_SIF:-$HOME/containers/urban-syn.sif}"

STAGE=all
CONFIG="configs/experiments/nyc_joint.yaml"
STUDY=""
EXPERIMENT=""
MODE=""
RUN_DIR=""
CHECKPOINT=""

# Preserve old positional commands, including their optional third mode argument.
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    STAGE="$1"; shift
    if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
        CONFIG="$1"; shift
    fi
fi

while (($#)); do
    case "$1" in
        --study|--experiment|--stage|--config|--run-dir|--checkpoint)
            if (($# < 2)); then echo "ERROR: $1 requires a value." >&2; exit 2; fi
            case "$1" in
                --study) STUDY="$2" ;;
                --experiment) EXPERIMENT="$2" ;;
                --stage) STAGE="$2" ;;
                --config) CONFIG="$2" ;;
                --run-dir) RUN_DIR="$2" ;;
                --checkpoint) CHECKPOINT="$2" ;;
            esac
            shift 2 ;;
        --dry-run|--validate-only)
            if [[ -n "$MODE" ]]; then echo "ERROR: Select only one execution mode." >&2; exit 2; fi
            MODE="$1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$STUDY" && -z "$EXPERIMENT" ]]; then
    echo "ERROR: --experiment is required with --study." >&2; exit 2
fi
if [[ -z "$STUDY" && -n "$EXPERIMENT" ]]; then
    echo "ERROR: --experiment requires --study." >&2; exit 2
fi

PLAN_ARGS=(--project-root "$PROJECT_ROOT")
if [[ -n "$STUDY" ]]; then
    PLAN_ARGS+=(--study "$STUDY" --experiment "$EXPERIMENT")
else
    PLAN_ARGS+=(--config "$CONFIG")
fi

# The submission host may not have PyYAML. In that case, use the existing
# Apptainer image, which already contains PyYAML through the project lockfile.
PLAN_FILE="$(mktemp)"
trap 'rm -f -- "$PLAN_FILE"' EXIT
if python3 -c 'import yaml' >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/resolve_submission.py" "${PLAN_ARGS[@]}" > "$PLAN_FILE"
else
    if [[ ! -f "$SIF" ]]; then
        echo "ERROR: PyYAML is unavailable on the login host and the SIF is missing: $SIF" >&2
        exit 1
    fi
    # Translate absolute host arguments into container paths where necessary.
    for i in "${!PLAN_ARGS[@]}"; do
        if [[ "${PLAN_ARGS[i]}" == "$PROJECT_ROOT" ]]; then
            PLAN_ARGS[i]="/workspace"
        elif [[ "${PLAN_ARGS[i]}" == "$PROJECT_ROOT/"* ]]; then
            PLAN_ARGS[i]="/workspace/${PLAN_ARGS[i]#"$PROJECT_ROOT/"}"
        fi
    done
    apptainer exec --cleanenv --bind "$PROJECT_ROOT:/workspace" --pwd /workspace \
        "$SIF" /opt/venv/bin/python /workspace/hpc/urban_syn/resolve_submission.py \
        "${PLAN_ARGS[@]}" > "$PLAN_FILE"
    # The container returns /workspace paths. Use host paths for Slurm's input check.
fi
mapfile -t PLAN < "$PLAN_FILE"
if ((${#PLAN[@]} != 11)); then echo "ERROR: Invalid submission settings." >&2; exit 2; fi
CONFIG_HOST="${PLAN[0]}"
PIPELINE="${PLAN[1]}"
if [[ "$CONFIG_HOST" == /workspace/* ]]; then
    CONFIG_HOST="$PROJECT_ROOT/${CONFIG_HOST#/workspace/}"
fi

# Explicit submission variables override study YAML defaults, then script defaults.
PARTITION="${SBATCH_PARTITION:-${PLAN[2]:-gpuqm}}"
GRES="${SBATCH_GRES:-${PLAN[3]:-gpu:a100:1}}"
TIMELIMIT="${SBATCH_TIMELIMIT:-${PLAN[4]:-1-00:00:00}}"
MEMORY="${SBATCH_MEM_PER_NODE:-${PLAN[5]:-64G}}"
CPUS="${SBATCH_CPUS_PER_TASK:-${PLAN[6]:-8}}"
MAIL_USER="${URBAN_SYN_MAIL_USER:-${PLAN[7]}}"
MAIL_TYPE="${URBAN_SYN_MAIL_TYPE:-${PLAN[8]:-NONE}}"
STUDY_NAME="${PLAN[9]}"
EXPERIMENT_KEY="${PLAN[10]}"

if [[ -n "$MAIL_USER" && "$MAIL_TYPE" == NONE ]]; then
    echo "ERROR: Set URBAN_SYN_MAIL_TYPE (or study hpc.mail_type) when setting a mail user." >&2
    exit 2
fi
MAIL_ARGS=(--mail-type="$MAIL_TYPE")
if [[ -n "$MAIL_USER" ]]; then
    MAIL_ARGS+=(--mail-user="$MAIL_USER")
fi

LOG_DIR="$PROJECT_ROOT/hpc/urban_syn/logs"
mkdir -p "$LOG_DIR"

# Accept host-absolute paths inside this checkout and translate them to
# the /workspace bind path before forwarding optional pretraining paths.
if [[ "$RUN_DIR" == "$PROJECT_ROOT/"* ]]; then
    RUN_DIR="/workspace/${RUN_DIR#"$PROJECT_ROOT/"}"
fi
if [[ "$CHECKPOINT" == "$PROJECT_ROOT/"* ]]; then
    CHECKPOINT="/workspace/${CHECKPOINT#"$PROJECT_ROOT/"}"
fi

# The sbatch script receives the resolved experiment YAML; it doesn't parse
# study files or change model settings after Slurm has allocated resources.
JOB_ARGS=("$STAGE" "$CONFIG_HOST")
if [[ -n "$MODE" ]]; then JOB_ARGS+=("$MODE"); fi
if [[ -n "$RUN_DIR" ]]; then JOB_ARGS+=(--run-dir "$RUN_DIR"); fi
if [[ -n "$CHECKPOINT" ]]; then JOB_ARGS+=(--checkpoint "$CHECKPOINT"); fi

printf 'Submitting %s: stage=%s, config=%s\n' "$PIPELINE" "$STAGE" "$CONFIG_HOST"
printf 'Slurm: partition=%s, gres=%s, cpus=%s, mem=%s, time=%s, mail=%s\n' \
    "$PARTITION" "$GRES" "$CPUS" "$MEMORY" "$TIMELIMIT" "$MAIL_TYPE"

# Command-line options explicitly carry the resolved precedence. Keep all
# original SBATCH_* overrides and the email settings without forcing YAML edits.
sbatch \
    --partition="$PARTITION" --gres="$GRES" --cpus-per-task="$CPUS" \
    --mem="$MEMORY" --time="$TIMELIMIT" \
    "${MAIL_ARGS[@]}" \
    --chdir="$PROJECT_ROOT" \
    --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" \
    --export="ALL,URBAN_SYN_PROJECT_ROOT=$PROJECT_ROOT,URBAN_SYN_PIPELINE=$PIPELINE,URBAN_SYN_STUDY_NAME=$STUDY_NAME,URBAN_SYN_EXPERIMENT_KEY=$EXPERIMENT_KEY" \
    "$SCRIPT_DIR/run_urban_syn.sbatch" "${JOB_ARGS[@]}"
