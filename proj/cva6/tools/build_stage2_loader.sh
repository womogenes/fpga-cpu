#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host="$root/sdk/buildroot/output/host"
cross="$host/bin/riscv64-buildroot-linux-gnu-"
output="$root/build/sdk"

if [[ -x "${cross}gcc" ]]; then
    gcc="${cross}gcc"
    objcopy="${cross}objcopy"
    size="${cross}size"
else
    gcc="$(command -v riscv64-elf-gcc || command -v riscv64-unknown-elf-gcc || true)"
    if [[ -z "$gcc" ]]; then
        echo "No RISC-V compiler found; run 'make image' or install a bare-metal toolchain." >&2
        exit 1
    fi
    tool_prefix="${gcc%gcc}"
    objcopy="${tool_prefix}objcopy"
    size="${tool_prefix}size"
fi

mkdir -p "$output"
"$gcc" -nostdlib -nostartfiles -static -march=rv64imac_zifencei -mabi=lp64 -Wl,--build-id=none -T "$root/loader/linker.ld" -o "$output/stage2-loader.elf" "$root/loader/stage2.S"
"$objcopy" -O binary "$output/stage2-loader.elf" "$output/stage2-loader.bin"
"$size" "$output/stage2-loader.elf"
sha256sum "$output/stage2-loader.bin"
