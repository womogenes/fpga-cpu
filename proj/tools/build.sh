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

uart_baud="${UART_BAUD:-3000000}"
uart_fifo_depth="${UART_FIFO_DEPTH:-512}"
uart_baud_supplied=0
uart_fifo_depth_supplied=0
previous_argument=""
for argument in "$@"; do
    if [[ "$previous_argument" == "--uart-baudrate" ]]; then
        uart_baud="$argument"
        uart_baud_supplied=1
    elif [[ "$argument" == --uart-baudrate=* ]]; then
        uart_baud="${argument#*=}"
        uart_baud_supplied=1
    elif [[ "$argument" == "--uart-baudrate" ]]; then
        uart_baud_supplied=1
    fi
    if [[ "$previous_argument" == "--uart-fifo-depth" ]]; then
        uart_fifo_depth="$argument"
        uart_fifo_depth_supplied=1
    elif [[ "$argument" == --uart-fifo-depth=* ]]; then
        uart_fifo_depth="${argument#*=}"
        uart_fifo_depth_supplied=1
    elif [[ "$argument" == "--uart-fifo-depth" ]]; then
        uart_fifo_depth_supplied=1
    fi
    previous_argument="$argument"
done

build_arguments=("$@")
if [[ "$uart_baud_supplied" == 0 ]]; then
    build_arguments+=(--uart-baudrate "$uart_baud")
fi
if [[ "$uart_fifo_depth_supplied" == 0 ]]; then
    build_arguments+=(--uart-fifo-depth "$uart_fifo_depth")
fi

cd "$project_dir"
"$python" hdl/soc.py \
    --build \
    --output-dir "$project_dir/build/fpga_soc" \
    "${build_arguments[@]}"

# Record the baud associated with the successfully generated bitstream so the
# monitor and uploader select the matching rate automatically.
printf '%s\n' "$uart_baud" > "$project_dir/build/fpga_soc/uart_baud"
