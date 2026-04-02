#!/usr/bin/env bash
# Strict mode:
# -e: exit on error
# -u: error on undefined vars
# -o pipefail: catch errors in pipelines
# Fail fast: exit on errors, undefined variables, and pipeline failures
set -euo pipefail 

cd /home/dev/src

# Avoid hardcoding path
WORKSPACE_DIR=${WORKSPACE_DIR:-/home/dev/src}
cd "$WORKSPACE_DIR"

# Marks this mounted repo from host as safe so Git can be used
# Mark repo as safe (idempotent)
if ! git config --global --get-all safe.directory | grep -q "$WORKSPACE_DIR"; then
  git config --global --add safe.directory "$WORKSPACE_DIR"
fi 

sudo mkdir -p /home/vscode/.cache/uv
sudo chown -R vscode:vscode /home/vscode/.cache/uv

# Sync project deps from uv.lock (creates .venv automatically)
# Since your repo is mounted, this creates .venv in the
# workspace instead of baking it into the image, which avoids the “mounted volume hides image .venv” problem.
uv sync --frozen

echo "✅ Devcontainer ready."