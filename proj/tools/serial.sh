#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
litex_term="$project_dir/.venv/bin/litex_term"
device="${LITEX_UART:-/dev/ttyUSB0}"
speed="${LITEX_BAUD:-115200}"

if [[ ! -x "$litex_term" ]]; then
    echo "litex_term not found: $litex_term" >&2
    echo "Run make setup first." >&2
    exit 1
fi

if command -v fuser >/dev/null 2>&1; then
    uart_holders="$(fuser "$device" 2>/dev/null || true)"
    if [[ -n "${uart_holders//[[:space:]]/}" ]]; then
        echo "UART is already in use: $device" >&2
        echo "Holding process IDs:$uart_holders" >&2
        echo "Close the other serial console, then retry." >&2
        exit 1
    fi
fi

echo "Opening LiteX serial monitor on $device at $speed baud."
exec "$litex_term" "$device" --speed "$speed" "$@"
