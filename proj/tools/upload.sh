#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
litex_term="$project_dir/.venv/bin/litex_term"
program="${PROGRAM:-framebuffer_demo}"
speed="${LITEX_BAUD:-115200}"
device="${LITEX_UART:-/dev/ttyUSB0}"

if [[ ! "$program" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "Invalid program name: $program" >&2
    exit 2
fi

kernel="$project_dir/sw/$program/build/$program.bin"

if [[ ! -x "$litex_term" ]]; then
    echo "litex_term not found: $litex_term" >&2
    echo "Run make setup first." >&2
    exit 1
fi

if [[ ! -f "$kernel" ]]; then
    echo "Program not found: $kernel" >&2
    echo "Build it with: make -C sw/$program" >&2
    exit 1
fi

if command -v fuser >/dev/null 2>&1; then
    uart_holders="$(fuser "$device" 2>/dev/null || true)"
    if [[ -n "${uart_holders//[[:space:]]/}" ]]; then
        echo "UART is already in use: $device" >&2
        echo "Holding process IDs:$uart_holders" >&2
        echo "Close the other serial console, then retry the upload." >&2
        exit 1
    fi
fi

exec "$litex_term" "$device" \
    --speed "$speed" \
    --kernel "$kernel" \
    --kernel-adr 0x40000000 \
    "$@"
