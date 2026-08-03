#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${REMOTE_PROJECT:-thinkmax:~/fpga/fpga-cpu/}"
software_destination="${destination%/}/sw/"

exec rsync -az --delete \
    --exclude='__pycache__/' \
    "$@" \
    "$project_dir/sw/" "$software_destination"
