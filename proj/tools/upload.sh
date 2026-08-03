#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
litex_term="$project_dir/.venv/bin/litex_term"
kernel="$project_dir/sw/framebuffer_demo/build/framebuffer_demo.bin"
device="${LITEX_UART:-/dev/ttyUSB0}"
speed="${LITEX_BAUD:-115200}"

if [[ ! -x "$litex_term" ]]; then
    echo "litex_term not found: $litex_term" >&2
    echo "Run make setup first." >&2
    exit 1
fi

if [[ ! -f "$kernel" ]]; then
    echo "Program not found: $kernel" >&2
    echo "Run make software first." >&2
    exit 1
fi

exec "$litex_term" "$device" \
    --speed "$speed" \
    --kernel "$kernel" \
    --kernel-adr 0x40000000 \
    "$@"
