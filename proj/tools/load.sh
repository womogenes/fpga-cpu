#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bitstream="$project_dir/build/fpga_soc/gateware/digilent_genesys2.bit"
frequency="${JTAG_FREQ:-1000000}"

if [[ ! -f "$bitstream" ]]; then
    echo "Bitstream not found: $bitstream" >&2
    echo "Run make build first." >&2
    exit 1
fi

exec openFPGALoader -b genesys2 --freq "$frequency" "$@" "$bitstream"
