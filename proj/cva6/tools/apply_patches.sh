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

apply_stack "$root/hardware" \
    "$root/patches/cva6-bootrom-uart-clock.patch" \
    "$root/patches/cva6-soc-video-accelerator.patch" \
    "$root/patches/cva6-100mhz.patch" \
    "$root/patches/cva6-smp.patch" \
    "$root/patches/cva6-four-hart.patch"
apply_once "$root/hardware/core/cvfpu" "$root/patches/cvfpu-fpga-fma-pipeline.patch"
apply_once "$root/hardware/corev_apu/fpga/src/apb_uart" "$root/patches/apb-uart-3mbaud.patch"
