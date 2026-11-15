#!/usr/bin/env bash
# Submit one URBAN-SYN pipeline job and route Slurm stdout/stderr to hpc/urban_syn/logs/.
# Usage: bash hpc/urban_syn/submit.sh [stage] [config] [mode]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${URBAN_SYN_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd -P)"
LOG_DIR="$PROJECT_ROOT/hpc/urban_syn/logs"
mkdir -p "$LOG_DIR"

# MLflow storage scope is configured in the experiment YAML.
# Keep only deployment paths as shell-level overrides.
sbatch \
    --chdir="$PROJECT_ROOT" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    --export="ALL,URBAN_SYN_PROJECT_ROOT=$PROJECT_ROOT" \
    "$SCRIPT_DIR/run_urban_syn.sbatch" "$@"
