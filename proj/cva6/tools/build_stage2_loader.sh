#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host="$root/sdk/buildroot/output/host"
cross="$host/bin/riscv64-buildroot-linux-gnu-"
output="$root/build/sdk"

if [[ ! -x "${cross}gcc" ]]; then
    echo "Buildroot cross compiler is missing; run 'make image' first." >&2
    exit 1
fi

mkdir -p "$output"
"${cross}gcc" -nostdlib -nostartfiles -static -march=rv64imac_zifencei -mabi=lp64 -Wl,--build-id=none -T "$root/loader/linker.ld" -o "$output/stage2-loader.elf" "$root/loader/stage2.S"
"${cross}objcopy" -O binary "$output/stage2-loader.elf" "$output/stage2-loader.bin"
"${cross}size" "$output/stage2-loader.elf"
sha256sum "$output/stage2-loader.bin"
