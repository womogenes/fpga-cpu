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

apply_once "$root/hardware" "$root/patches/cva6-bootrom-uart-clock.patch"
apply_once "$root/hardware" "$root/patches/cva6-soc-video-accelerator.patch"
apply_once "$root/hardware/corev_apu/fpga/src/apb_uart" "$root/patches/apb-uart-3mbaud.patch"
