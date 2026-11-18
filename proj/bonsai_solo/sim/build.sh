#!/usr/bin/env bash
# Build the Verilator full-system model. Run on vivado-v0, not the laptop.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/sim/obj}"
boot_hex="${BOOT_HEX:-$root/fw/build/sim_boot.hex}"
cd "$root"
verilator --cc --exe --build -j "$(nproc)" -O3 --x-assign fast --x-initial fast \
    -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
    -Wno-UNUSEDPARAM -Wno-DECLFILENAME -Wno-TIMESCALEMOD \
    -Wno-PINCONNECTEMPTY -Wno-GENUNNAMED -Wno-UNOPTFLAT \
    -Ivendor/pulp_axi/include -Ivendor/common_cells/include -Ivendor/cvfpu \
    --top-module solo_core -GBOOT_HEX="\"$boot_hex\"" \
    --Mdir "$out" -CFLAGS -O2 -o tb_solo \
    vendor/pulp_axi/axi_pkg.sv vendor/pulp_axi/axi_intf.sv \
    vendor/cvfpu/fpnew_pkg.sv vendor/common_cells/cf_math_pkg.sv \
    vendor/common_cells/lzc.sv vendor/common_cells/spill_register_flushable.sv \
    vendor/common_cells/spill_register.sv vendor/pulp_axi/axi_cut.sv \
    vendor/cvfpu/fpnew_classifier.sv vendor/cvfpu/fpnew_rounding.sv \
    vendor/cvfpu/fpnew_fma.sv vendor/picorv32/picorv32.v \
    rtl/bonsai_fp32_pipe.sv rtl/bonsai_fp32_alu.sv rtl/bonsai_fp32_cdc.sv \
    rtl/bonsai_ffn_packer.sv rtl/bonsai_residual.sv rtl/bonsai_rms_scale.sv \
    rtl/bonsai_attention.sv rtl/bonsai_attention_quad.sv \
    rtl/lfm_axi_byte_reader.sv rtl/lfm_axi_arbiter.sv \
    rtl/lfm_matvec_accelerator.sv rtl/uart_rx.sv rtl/uart_tx.sv \
    rtl/solo_ctrl.sv rtl/solo_core.sv "$root/sim/tb_solo.cpp"
