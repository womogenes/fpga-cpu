#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hardware="$root/hardware"
output="$root/build/gateware"
bitstream="$output/cva6_genesys2_hdmi.bit"
hash_file="$output/inputs.sha256"
uart_baud="${UART_BAUD:-3000000}"
cpu_frequency=100000000
uart_reference_frequency=96000000

mkdir -p "$output"
if [[ "${SKIP_PATCHES:-0}" != 1 ]]; then
    "$root/tools/apply_patches.sh"
fi

if [[ -z "${RISCV:-}" || -z "${CROSSCOMPILE:-}" ]]; then
    bootrom_cc="$(command -v riscv64-elf-gcc || command -v riscv64-unknown-elf-gcc || true)"
    if [[ -z "$bootrom_cc" ]]; then
        echo "No bare-metal RISC-V compiler was found." >&2
        echo "Install riscv64-elf-gcc or set RISCV and CROSSCOMPILE." >&2
        exit 1
    fi
    RISCV="$(cd "$(dirname "$bootrom_cc")/.." && pwd)"
    CROSSCOMPILE="$(basename "$bootrom_cc")"
    CROSSCOMPILE="${CROSSCOMPILE%gcc}"
else
    bootrom_cc="$RISCV/bin/${CROSSCOMPILE}gcc"
fi
export RISCV
export CROSSCOMPILE

# The boot ROM sources are generated files tracked by the hardware submodule.
# Regenerate them before hashing so the cache key describes the exact RTL that
# Vivado will consume instead of the state left by the preceding build.
bootrom_cflags="-DCLOCK_FREQUENCY=$cpu_frequency -DUART_CLOCK_FREQUENCY=$uart_reference_frequency -DUART_BITRATE=$uart_baud -DPLAT_XILINX -DXLEN=64 -Os -ggdb -march=rv64im_zicsr_zifencei -mabi=lp64 -Wall -mcmodel=medany -mexplicit-relocs -ffreestanding -fpermissive"
make -C "$hardware/corev_apu/fpga/src/bootrom" \
    -B \
    BOARD=genesys2 \
    PLATFORM=PLAT_XILINX \
    XLEN=64 \
    CLOCK_FREQUENCY="$cpu_frequency" \
    HALF_CLOCK_FREQUENCY="$((cpu_frequency / 2))" \
    UART_CLOCK_FREQUENCY="$uart_reference_frequency" \
    UART_BITRATE="$uart_baud" \
    CFLAGS="$bootrom_cflags" \
    bootrom_64.sv

input_hash="$({
    vivado -version | sed -n '1p'
    "$bootrom_cc" --version | sed -n '1p'
    git -C "$hardware" rev-parse HEAD
    git -C "$hardware" submodule status --recursive
    git -C "$hardware" diff --binary HEAD
    sha256sum \
        "$hardware/corev_apu/fpga/src/apb_uart/src/vhdl_orig/apb_uart.vhd" \
        "$hardware/corev_apu/fpga/src/apb_uart/src/vhdl_orig/uart_baudgen.vhd" \
        "$hardware/corev_apu/fpga/src/bootrom/src/main.c"
    printf 'UART_BAUD=%s\n' "$uart_baud"
    find "$root/rtl" "$root/constraints" "$root/scripts" -type f -print0 | sort -z | xargs -0 sha256sum
} | sha256sum | cut -d' ' -f1)"

if [[ "${HASH_ONLY:-0}" == 1 ]]; then
    printf '%s\n' "$input_hash"
    exit 0
fi

if [[ "${FORCE:-0}" != 1 && -f "$bitstream" && -f "$hash_file" && "$(<"$hash_file")" == "$input_hash" ]]; then
    echo "Reusing cached $bitstream"
    exit 0
fi

if pgrep -f '[V]ivado/bin/(vivado|loader)|[V]ivado/.*/unwrapped/.*/vivado|[p]arallel_synth_helper' >/dev/null; then
    echo "Another Vivado build is already active; refusing to start a second one." >&2
    exit 1
fi

export CVA6_HDMI_ROOT="$root"
export CVA6_UART_BAUD="$uart_baud"
export RISCV
export CROSSCOMPILE
export VIVADOFLAGS="-nojournal -mode batch -source scripts/prologue.tcl -source $root/scripts/hdmi_hook.tcl"

rm -f "$hardware/corev_apu/fpga/work-fpga/ariane_xilinx.bit"
make -C "$hardware" BOARD=genesys2 XLEN=64 fpga
cp "$hardware/corev_apu/fpga/work-fpga/ariane_xilinx.bit" "$bitstream"
cp "$hardware/corev_apu/fpga/reports/ariane.timing.rpt" "$output/timing.rpt"
cp "$hardware/corev_apu/fpga/reports/ariane.utilization.rpt" "$output/utilization.rpt"
printf '%s\n' "$input_hash" > "$hash_file"

echo "Built $bitstream"
