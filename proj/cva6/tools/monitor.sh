#!/usr/bin/env bash
set -euo pipefail

serial="${SERIAL:-/dev/ttyUSB0}"
baud="${UART_BAUD:-3000000}"

if python3 -c 'import serial' >/dev/null 2>&1; then
    exec "$(dirname "$0")/serial_monitor.py" --serial "$serial" --baud "$baud"
fi

if command -v picocom >/dev/null 2>&1; then
    exec picocom --baud "$baud" "$serial"
fi

echo "Install picocom or Python pyserial to monitor $serial at $baud baud." >&2
exit 1
