#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${REMOTE_PROJECT:-thinkmax:~/fpga/litex-cpu/}"

exec rsync -az --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='.vscode/' \
    --exclude='__pycache__/' \
    "$@" \
    "$project_dir/" "$destination"
