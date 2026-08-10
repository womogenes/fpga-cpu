#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 ARTIFACT_DIR MODEL" >&2
    exit 2
fi

artifact_dir="$(cd "$1" && pwd)"
model="$(realpath "$2")"
log="$artifact_dir/handoff.log"
ready="$artifact_dir/handoff.ready"

exec > >(tee -a "$log") 2>&1
echo "$(date --iso-8601=seconds) Starting FPGA chat handoff"
rm -f "$ready"
tmux kill-session -t bonsai-console 2>/dev/null || true
pkill -f 'serial.tools.miniterm /dev/ttyUSB0' 2>/dev/null || true

python3 "$artifact_dir/program_and_upload.py" \
    --bitstream "$artifact_dir/cva6_genesys2_hdmi_coherent_ordered.bit" \
    --stage2 "$artifact_dir/stage2-loader.bin" \
    --model "$model" \
    --serial /dev/ttyUSB0 \
    --baud 3000000 \
    --jtag-frequency 1000000 \
    --boot-wait 180 \
    --no-console \
    "$artifact_dir/ram-linux.bin"

python3 "$artifact_dir/serial_start_chat.py" \
    --serial /dev/ttyUSB0 \
    --baud 3000000 \
    --timeout 600

tmux new-session -d -s bonsai-console \
    "python3 -m serial.tools.miniterm /dev/ttyUSB0 3000000 --raw"
sleep 2
tmux send-keys -t bonsai-console Enter
sleep 2
tmux capture-pane -pt bonsai-console -S -80
date --iso-8601=seconds >"$ready"
echo "$(date --iso-8601=seconds) FPGA chat handoff is ready"
