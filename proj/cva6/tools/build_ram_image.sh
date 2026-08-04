#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="$root/build/sdk"
host="$root/sdk/buildroot/output/host"
build="$root/sdk/buildroot/output/build"
cross="$host/bin/riscv64-buildroot-linux-gnu-"
build_path="$root/build/host-tools/bin:$host/bin:$PATH"

kernel="$(find "$build" -maxdepth 1 -type d -name 'linux-[0-9]*' -print -quit)"
opensbi="$(find "$build" -maxdepth 1 -type d -name 'opensbi-[0-9]*' -print -quit)"
rootfs="$output/rootfs.cpio"
payload_base=$((0x80200000))
fdt_addr=$((0xbfe00000))

if [[ -z "$kernel" || -z "$opensbi" || ! -f "$rootfs" ]]; then
    echo "Buildroot kernel, OpenSBI, or rootfs.cpio is missing; run 'make image' first." >&2
    exit 1
fi

echo "Embedding $(du -h "$rootfs" | cut -f1) root filesystem into Linux..."
"$kernel/scripts/config" --file "$kernel/.config" \
    --enable BLK_DEV_INITRD \
    --enable RD_ZSTD \
    --set-str INITRAMFS_SOURCE "$rootfs" \
    --set-val INITRAMFS_ROOT_UID 0 \
    --set-val INITRAMFS_ROOT_GID 0 \
    --enable INITRAMFS_COMPRESSION_NONE \
    --disable INITRAMFS_COMPRESSION_ZSTD \
    --disable INITRAMFS_COMPRESSION_BZIP2 \
    --disable INITRAMFS_COMPRESSION_LZMA \
    --disable INITRAMFS_COMPRESSION_XZ \
    --disable INITRAMFS_COMPRESSION_LZO \
    --disable INITRAMFS_COMPRESSION_LZ4 \
    --disable INITRAMFS_COMPRESSION_GZIP

PATH="$build_path" make -C "$kernel" ARCH=riscv CROSS_COMPILE="$cross" olddefconfig
PATH="$build_path" make -j"$(nproc)" -C "$kernel" ARCH=riscv CROSS_COMPILE="$cross" Image

kernel_size="$(stat -c %s "$kernel/arch/riscv/boot/Image")"
payload_end=$((payload_base + kernel_size))
if ((payload_end > fdt_addr)); then
    printf 'Linux payload ends at 0x%x and overlaps the FDT at 0x%x.\n' "$payload_end" "$fdt_addr" >&2
    exit 1
fi

"$host/bin/dtc" -q -I dts -O dtb -i "$root/config" -o "$output/cv64a6_ram.dtb" "$root/config/cv64a6_ram.dts"

ram_opensbi="$root/build/ram-opensbi"
PATH="$build_path" make -C "$opensbi" O="$ram_opensbi" clean
PATH="$build_path" make -j"$(nproc)" -C "$opensbi" \
    O="$ram_opensbi" \
    CROSS_COMPILE="$cross" \
    PLATFORM=generic \
    FW_PAYLOAD_PATH="$kernel/arch/riscv/boot/Image" \
    FW_FDT_PATH="$output/cv64a6_ram.dtb" \
    FW_PAYLOAD_FDT_ADDR="$(printf '0x%x' "$fdt_addr")"

install -m 0644 "$ram_opensbi/platform/generic/firmware/fw_payload.bin" "$output/ram-linux.bin"
install -m 0644 "$ram_opensbi/platform/generic/firmware/fw_payload.elf" "$output/ram-linux.elf"
sha256sum "$output/ram-linux.bin"
du -h "$output/ram-linux.bin"
