#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bitstream="$project_dir/build/genesys2_mvp/gateware/digilent_genesys2.bit"

if [[ ! -f "$bitstream" ]]; then
    echo "Bitstream not found: $bitstream" >&2
    echo "Run make build first." >&2
    exit 1
fi

exec openFPGALoader -b genesys2 --freq 1000000 "$@" "$bitstream"
