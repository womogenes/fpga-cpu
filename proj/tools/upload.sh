#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
program="${PROGRAM:-framebuffer_demo}"

if [[ ! "$program" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "Invalid program name: $program" >&2
    exit 2
fi

kernel="$project_dir/sw/$program/build/$program.bin"

if [[ ! -f "$kernel" ]]; then
    echo "Program not found: $kernel" >&2
    echo "Build it with: make -C sw/$program" >&2
    exit 1
fi

exec "$project_dir/tools/serial.sh" \
    --kernel "$kernel" \
    --kernel-adr 0x40000000 \
    "$@"
