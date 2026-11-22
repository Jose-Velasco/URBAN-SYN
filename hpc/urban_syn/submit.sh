#!/usr/bin/env bash
# Submit one experiment; pretrained joint "all" also submits its pretraining dependency.
# Existing usage remains: submit.sh [stage] [experiment.yaml] [--dry-run|--validate-only]
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash hpc/urban_syn/submit.sh [stage] [config.yaml] [--dry-run|--validate-only] [--run-dir DIR] [--checkpoint FILE] [--encoder-checkpoint FILE]
  bash hpc/urban_syn/submit.sh --study STUDY.yaml --experiment NAME [--stage STAGE] [--dry-run|--validate-only] [--run-dir DIR] [--checkpoint FILE] [--encoder-checkpoint FILE] [--pretrain-job-id JOB_ID]

The study YAML provides default Slurm resources. Environment overrides take priority:
  SBATCH_PARTITION, SBATCH_GRES, SBATCH_TIMELIMIT, SBATCH_MEM_PER_NODE, SBATCH_CPUS_PER_TASK,
  SBATCH_EXCLUDE (e.g. cs002 or cs002,cs003)
  URBAN_SYN_MAIL_USER and URBAN_SYN_MAIL_TYPE (e.g. BEGIN,END,FAIL)

--run-dir and --checkpoint are supported for standalone link-pretraining stages.
For joint generation, select generation.checkpoint in the joint experiment YAML.
For a joint study experiment with pretrained_from, --stage all automatically submits
pretraining followed by a dependent joint job (generation follows YAML settings).
For --stage train, a previously published encoder is reused.
--encoder-checkpoint explicitly selects an existing encoder and skips pretraining.
--pretrain-job-id attaches a dependent joint job to an already submitted pretraining
job (afterok), using that job's exact encoder instead of submitting pretraining again.
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
ENCODER_CHECKPOINT=""
PRETRAIN_JOB_ID=""

# Preserve old positional commands, including their optional third mode argument.
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    STAGE="$1"; shift
    if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
        CONFIG="$1"; shift
    fi
fi

while (($#)); do
    case "$1" in
        --study|--experiment|--stage|--config|--run-dir|--checkpoint|--encoder-checkpoint|--pretrain-job-id)
            if (($# < 2)); then echo "ERROR: $1 requires a value." >&2; exit 2; fi
            case "$1" in
                --study) STUDY="$2" ;;
                --experiment) EXPERIMENT="$2" ;;
                --stage) STAGE="$2" ;;
                --config) CONFIG="$2" ;;
                --run-dir) RUN_DIR="$2" ;;
                --checkpoint) CHECKPOINT="$2" ;;
                --encoder-checkpoint) ENCODER_CHECKPOINT="$2" ;;
                --pretrain-job-id) PRETRAIN_JOB_ID="$2" ;;
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
if [[ -n "$PRETRAIN_JOB_ID" ]]; then
    if [[ ! "$PRETRAIN_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --pretrain-job-id requires a numeric Slurm job ID." >&2; exit 2
    fi
    if [[ -z "$STUDY" || "$STAGE" != all || -n "$MODE" ||
          -n "$ENCODER_CHECKPOINT" || -n "$RUN_DIR" || -n "$CHECKPOINT" ]]; then
        echo "ERROR: --pretrain-job-id requires a study joint --stage all and cannot be combined with checkpoints, run paths, or preview modes." >&2; exit 2
    fi
fi

PLAN_ARGS=(--project-root "$PROJECT_ROOT")
if [[ -n "$STUDY" ]]; then
    PLAN_ARGS+=(--study "$STUDY" --experiment "$EXPERIMENT")
else
    PLAN_ARGS+=(--config "$CONFIG")
fi
# A complete joint study experiment may produce its encoder in a future Slurm
# job. Validate the source YAML now without demanding a checkpoint that does
# not exist yet. Explicit manual checkpoints likewise need no study reference.
if [[ -n "$STUDY" && ( ( "$STAGE" == all && -z "$MODE" ) || -n "$ENCODER_CHECKPOINT" || -n "$MODE" ) ]]; then
    PLAN_ARGS+=(--pending-pretrain)
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
if ((${#PLAN[@]} != 16)); then echo "ERROR: Invalid submission settings." >&2; exit 2; fi
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
EXCLUDE="${SBATCH_EXCLUDE:-${PLAN[11]}}"
# A study source is resolved once during submission, never looked up again by
# the running job. This pins every joint run to one producing pretraining run.
if [[ -z "$ENCODER_CHECKPOINT" ]]; then
    ENCODER_CHECKPOINT="${PLAN[12]}"
fi
SOURCE_RUN="${PLAN[13]}"
SOURCE_KEY="${PLAN[14]}"
SOURCE_CONFIG_HOST="${PLAN[15]}"
if [[ "$SOURCE_CONFIG_HOST" == /workspace/* ]]; then
    SOURCE_CONFIG_HOST="$PROJECT_ROOT/${SOURCE_CONFIG_HOST#/workspace/}"
fi

# A requested exclusion is passed as one quoted Slurm host-list argument.
# No --exclude option is sent when neither YAML nor the environment specifies it.
EXCLUDE_ARGS=()
if [[ -n "$EXCLUDE" ]]; then
    EXCLUDE_ARGS+=(--exclude="$EXCLUDE")
fi

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
if [[ -n "$ENCODER_CHECKPOINT" ]]; then
    if [[ "$PIPELINE" != joint ]]; then
        echo "ERROR: --encoder-checkpoint is for joint diffusion training only." >&2
        exit 2
    fi
    if [[ "$ENCODER_CHECKPOINT" == "$PROJECT_ROOT/"* ]]; then
        ENCODER_CHECKPOINT="/workspace/${ENCODER_CHECKPOINT#"$PROJECT_ROOT/"}"
    elif [[ "$ENCODER_CHECKPOINT" != /workspace/* ]]; then
        # Paths supplied by the user may be repository-relative.
        if [[ "$ENCODER_CHECKPOINT" == /* ]]; then
            echo "ERROR: Encoder checkpoint must be inside the repository bind mount." >&2
            exit 2
        fi
        ENCODER_CHECKPOINT="/workspace/${ENCODER_CHECKPOINT#./}"
    fi
    CHECKPOINT_HOST="$PROJECT_ROOT/${ENCODER_CHECKPOINT#/workspace/}"
    if [[ ! -f "$CHECKPOINT_HOST" ]]; then
        echo "ERROR: Pretrained encoder checkpoint does not exist: $CHECKPOINT_HOST" >&2
        exit 2
    fi
fi

# The sbatch script receives the resolved experiment YAML. The selected joint
# training configuration is never rewritten by this launcher.
JOB_ARGS=("$STAGE" "$CONFIG_HOST")
if [[ -n "$MODE" ]]; then JOB_ARGS+=("$MODE"); fi
if [[ -n "$RUN_DIR" ]]; then JOB_ARGS+=(--run-dir "$RUN_DIR"); fi
if [[ -n "$CHECKPOINT" ]]; then JOB_ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "$ENCODER_CHECKPOINT" ]]; then JOB_ARGS+=(--encoder-checkpoint "$ENCODER_CHECKPOINT"); fi

# Only joint 'all' automatically runs its pretraining source. Standalone
# training/testing/generation and explicit encoder overrides retain their
# previous behavior. 'all' does not submit every experiment in the study.
CHAIN_PRETRAIN=false
if [[ "$PIPELINE" == joint && "$STAGE" == all && -z "$MODE" && \
      -n "$SOURCE_KEY" && -z "$ENCODER_CHECKPOINT" ]]; then
    CHAIN_PRETRAIN=true
fi
if [[ "$CHAIN_PRETRAIN" == true && ( -n "$RUN_DIR" || -n "$CHECKPOINT" ) ]]; then
    echo "ERROR: Joint --stage all does not accept pretraining --run-dir or --checkpoint." >&2
    exit 2
fi
if [[ -n "$PRETRAIN_JOB_ID" ]]; then
    if [[ "$PIPELINE" != joint || -z "$SOURCE_KEY" || -n "$ENCODER_CHECKPOINT" ]]; then
        echo "ERROR: --pretrain-job-id requires a pretrained joint experiment with a study source." >&2; exit 2
    fi
    # A supplied upstream job replaces new pretraining, never both.
    CHAIN_PRETRAIN=false
fi

printf 'Submitting %s: stage=%s, config=%s\n' "$PIPELINE" "$STAGE" "$CONFIG_HOST"
if [[ -n "$ENCODER_CHECKPOINT" ]]; then
    printf 'Pretrained encoder: %s (source run: %s)\n' "$ENCODER_CHECKPOINT" "${SOURCE_RUN:-manual}"
fi
printf 'Slurm: partition=%s, gres=%s, cpus=%s, mem=%s, time=%s, exclude=%s, mail=%s\n' \
    "$PARTITION" "$GRES" "$CPUS" "$MEMORY" "$TIMELIMIT" "${EXCLUDE:-none}" "$MAIL_TYPE"

# Every submitted job receives the same effective resources and email options.
# These are resolved as: environment overrides > study YAML > defaults.
SLURM_ARGS=(
    --partition="$PARTITION" --gres="$GRES" --cpus-per-task="$CPUS"
    --mem="$MEMORY" --time="$TIMELIMIT"
    "${EXCLUDE_ARGS[@]}" "${MAIL_ARGS[@]}"
    --chdir="$PROJECT_ROOT"
    --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err"
)

if [[ "$CHAIN_PRETRAIN" == true ]]; then
    # --parsable returns the allocated Slurm job ID. Use its immutable encoder
    # manifest rather than the study's mutable selected_encoder.json.
    printf 'Workflow: %s pretraining -> %s joint diffusion\n' "$SOURCE_KEY" "$EXPERIMENT_KEY"
    PRETRAIN_OUTPUT="$(sbatch --parsable \
        "${SLURM_ARGS[@]}" --job-name=urban-syn-pretrain \
        --export="ALL,URBAN_SYN_PROJECT_ROOT=$PROJECT_ROOT,URBAN_SYN_PIPELINE=link_pretrain,URBAN_SYN_STUDY_NAME=$STUDY_NAME,URBAN_SYN_EXPERIMENT_KEY=$SOURCE_KEY" \
        "$SCRIPT_DIR/run_urban_syn.sbatch" all "$SOURCE_CONFIG_HOST")"
    PRETRAIN_JOB_ID="${PRETRAIN_OUTPUT%%;*}"
    if [[ ! "$PRETRAIN_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Could not read the pretraining Slurm job ID: $PRETRAIN_OUTPUT" >&2
        exit 1
    fi
    printf 'Submitted pretraining job %s (%s).\n' "$PRETRAIN_JOB_ID" "$SOURCE_KEY"
fi

if [[ -n "$PRETRAIN_JOB_ID" ]]; then
    # 'afterok' waits for successful pretraining. The batch script resolves
    # the job-specific manifest at runtime, after it has been published.
    JOB_ARGS+=(--encoder-from-job "$PRETRAIN_JOB_ID" \
               --encoder-source-experiment "$SOURCE_KEY")
    printf 'Submitting dependent joint job (afterok:%s).\n' "$PRETRAIN_JOB_ID"
    if ! sbatch "${SLURM_ARGS[@]}" --job-name=urban-syn-joint \
        --dependency="afterok:$PRETRAIN_JOB_ID" \
        --export="ALL,URBAN_SYN_PROJECT_ROOT=$PROJECT_ROOT,URBAN_SYN_PIPELINE=joint,URBAN_SYN_STUDY_NAME=$STUDY_NAME,URBAN_SYN_EXPERIMENT_KEY=$EXPERIMENT_KEY" \
        "$SCRIPT_DIR/run_urban_syn.sbatch" "${JOB_ARGS[@]}"; then
        echo "ERROR: Dependent joint submission failed (pretraining job $PRETRAIN_JOB_ID)." >&2
        exit 1
    fi
else
    # Backward-compatible standalone and separately submitted experiments.
    sbatch "${SLURM_ARGS[@]}" \
        --export="ALL,URBAN_SYN_PROJECT_ROOT=$PROJECT_ROOT,URBAN_SYN_PIPELINE=$PIPELINE,URBAN_SYN_STUDY_NAME=$STUDY_NAME,URBAN_SYN_EXPERIMENT_KEY=$EXPERIMENT_KEY" \
        "$SCRIPT_DIR/run_urban_syn.sbatch" "${JOB_ARGS[@]}"
fi
