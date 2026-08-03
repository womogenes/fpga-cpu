#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$project_dir/.venv/bin/python"

if [[ ! -x "$python" ]]; then
    echo "LiteX environment not found: $project_dir/.venv" >&2
    echo "Run make setup first." >&2
    exit 1
fi

export PATH="$project_dir/.venv/bin:$PATH"

cd "$project_dir"
exec "$python" hdl/soc.py \
    --build \
    --output-dir "$project_dir/build/fpga_soc" \
    "$@"
