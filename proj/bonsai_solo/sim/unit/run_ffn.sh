#!/usr/bin/env bash
# FFN packer unit regression (from the CVA6 tree), for checking the
# simulator itself: run under each Verilator version and compare.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:-/tmp/ffn_unit}"
cd "$root"
verilator --binary --timing -j 0 -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
    -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-DECLFILENAME -Wno-TIMESCALEMOD \
    -Ivendor/common_cells/include -Ivendor/cvfpu --top-module bonsai_ffn_packer_tb \
    --Mdir "$out" vendor/cvfpu/fpnew_pkg.sv vendor/common_cells/cf_math_pkg.sv \
    vendor/common_cells/lzc.sv vendor/cvfpu/fpnew_classifier.sv \
    vendor/cvfpu/fpnew_rounding.sv vendor/cvfpu/fpnew_fma.sv \
    rtl/bonsai_fp32_pipe.sv rtl/bonsai_fp32_cdc.sv rtl/bonsai_ffn_packer.sv \
    sim/unit/bonsai_ffn_packer_tb.sv > "$out.build.log" 2>&1
"$out/Vbonsai_ffn_packer_tb" | tail -5
