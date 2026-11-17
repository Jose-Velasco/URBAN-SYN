#!/usr/bin/env bash
# Submit one URBAN-SYN pipeline job and route Slurm stdout/stderr to hpc/urban_syn/logs/.
# Usage: bash hpc/urban_syn/submit.sh [stage] [config] [mode]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${URBAN_SYN_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd -P)"
LOG_DIR="$PROJECT_ROOT/hpc/urban_syn/logs"
mkdir -p "$LOG_DIR"

# Email notifications are disabled by default for each submission.
# Set both variables when submitting a job to opt in to notifications.
MAIL_ARGS=(--mail-type=NONE)
if [[ -n "${URBAN_SYN_MAIL_USER:-}" && -z "${URBAN_SYN_MAIL_TYPE:-}" ]]; then
    echo "ERROR: Set URBAN_SYN_MAIL_TYPE when setting URBAN_SYN_MAIL_USER." >&2
    exit 2
fi
if [[ -n "${URBAN_SYN_MAIL_TYPE:-}" ]]; then
    MAIL_ARGS=(--mail-type="$URBAN_SYN_MAIL_TYPE")
    if [[ -n "${URBAN_SYN_MAIL_USER:-}" ]]; then
        MAIL_ARGS+=(--mail-user="$URBAN_SYN_MAIL_USER")
    fi
fi

# MLflow storage scope is configured in the experiment YAML.
# Keep deployment paths and optional email notifications as submission settings.
sbatch \
    "${MAIL_ARGS[@]}" \
    --chdir="$PROJECT_ROOT" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    --export="ALL,URBAN_SYN_PROJECT_ROOT=$PROJECT_ROOT" \
    "$SCRIPT_DIR/run_urban_syn.sbatch" "$@"
