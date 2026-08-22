#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apply_once() {
    local repository="$1"
    local patch="$2"
    if git -C "$repository" apply --check "$patch" 2>/dev/null; then
        git -C "$repository" apply "$patch"
        echo "Applied $(basename "$patch")"
    elif git -C "$repository" apply --reverse --check "$patch" 2>/dev/null; then
        echo "Already applied $(basename "$patch")"
    elif [[ "$(basename "$patch")" == "cva6-platform.patch" ]] &&
         git -C "$repository" grep -q "i_lfm_matvec_accelerator" HEAD -- \
             corev_apu/fpga/src/ariane_xilinx.sv &&
         git -C "$repository" grep -q "smp_inval_ready_o" HEAD -- \
             core/cva6.sv; then
        # Newer local CVA6 revisions carry the platform patch as a real commit.
        # A reverse check cannot match after subsequent accelerator clock/reset
        # edits, but reapplying the historical patch would be equally wrong.
        echo "Repository already contains $(basename "$patch")"
    elif [[ "$(basename "$patch")" == "apb-uart-3mbaud.patch" ]] &&
         git -C "$repository" grep -q "iBaudtick16x3M" -- \
             src/vhdl_orig/apb_uart.vhd &&
         git -C "$repository" grep -q "if (unsigned(iCounter) = 0) then" -- \
             src/vhdl_orig/uart_baudgen.vhd; then
        # The CPU-clock follow-up changes the fractional tick constants after the
        # original 3 MBaud patch, so its exact reverse context no longer fits.
        echo "Working tree already contains $(basename "$patch")"
    else
        echo "Cannot apply or recognize $patch in $repository" >&2
        exit 1
    fi
}

apply_stack() {
    local repository="$1"
    shift
    local patches=("$@")
    local start=0

    # Later patches intentionally edit context introduced by earlier patches,
    # so an early patch is not necessarily reverse-applicable once the entire
    # stack is present. Find the newest applied layer, then apply only the
    # remaining suffix.
    for ((index=${#patches[@]} - 1; index >= 0; --index)); do
        if git -C "$repository" apply --reverse --check "${patches[index]}" 2>/dev/null; then
            start=$((index + 1))
            echo "Patch stack already contains $(basename "${patches[index]}")"
            break
        fi
    done

    for ((index=start; index < ${#patches[@]}; ++index)); do
        if ! git -C "$repository" apply --check "${patches[index]}" 2>/dev/null; then
            echo "Cannot apply ${patches[index]} in $repository" >&2
            exit 1
        fi
        git -C "$repository" apply "${patches[index]}"
        echo "Applied $(basename "${patches[index]}")"
    done
}

apply_once "$root/hardware" "$root/patches/cva6-platform.patch"
apply_once "$root/hardware/corev_apu/fpga/src/apb_uart" "$root/patches/apb-uart-3mbaud.patch"
