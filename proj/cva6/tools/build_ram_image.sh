#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="$root/build/sdk"
buildroot_output="${BUILDROOT_OUTPUT:-$root/build/buildroot}"
if [[ ! -x "$buildroot_output/host/bin/riscv64-buildroot-linux-gnu-gcc" && -x "$root/sdk/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc" ]]; then
    # Keep recovery payload generation available while a newly isolated
    # Buildroot output tree is still being populated.
    buildroot_output="$root/sdk/buildroot/output"
fi
host="$buildroot_output/host"
build="$buildroot_output/build"
cross="$host/bin/riscv64-buildroot-linux-gnu-"
build_path="$root/build/host-tools/bin:$host/bin:$PATH"

kernel="$(find "$build" -maxdepth 1 -type d -name 'linux-[0-9]*' -print -quit)"
opensbi="$(find "$build" -maxdepth 1 -type d -name 'opensbi-[0-9]*' -print -quit)"
rootfs="$output/rootfs.cpio"
ram_rootfs="$output/rootfs-ram.cpio"
payload_base=$((0x80200000))
fdt_addr=$((0xbfe00000))

if [[ -z "$kernel" || -z "$opensbi" || ! -f "$rootfs" ]]; then
    echo "Buildroot kernel, OpenSBI, or rootfs.cpio is missing; run 'make image' first." >&2
    exit 1
fi

if [[ ! -f "$output/fw_payload.bin" || ! -f "$output/fitImage.itb" ]]; then
    echo "SD boot payloads are missing; run 'make image' first." >&2
    exit 1
fi

# A RAM boot can install itself onto the board's SD card without transferring
# the sparse 1.7 GiB disk image. Append only the two small boot payloads to the
# initramfs; sd-install formats the card and copies the live root filesystem.
installer_overlay="$(mktemp -d)"
trap 'rm -rf "$installer_overlay"' EXIT
mkdir -p "$installer_overlay/usr/share/sd-installer"
install -m 0644 "$output/fw_payload.bin" "$installer_overlay/usr/share/sd-installer/fw_payload.bin"
install -m 0644 "$output/fitImage.itb" "$installer_overlay/usr/share/sd-installer/fitImage.itb"
cp "$rootfs" "$ram_rootfs"
(
    cd "$installer_overlay"
    find . -print0 | sort -z | cpio --null --quiet -o --format=newc --owner=0:0
) >>"$ram_rootfs"

echo "Embedding $(du -h "$ram_rootfs" | cut -f1) RAM root filesystem into Linux..."
"$kernel/scripts/config" --file "$kernel/.config" \
    --enable BLK_DEV_INITRD \
    --enable RD_ZSTD \
    --set-str INITRAMFS_SOURCE "$ram_rootfs" \
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

if [[ -n "${RAM_DTB_SOURCE:-}" ]]; then
    install -m 0644 "$RAM_DTB_SOURCE" "$output/cv64a6_ram.dtb"
else
    "$host/bin/dtc" -q -I dts -O dtb -i "$root/config" -o "$output/cv64a6_ram.dtb" "$root/config/cv64a6_ram.dts"
fi

ram_opensbi="$root/build/ram-opensbi"
PATH="$build_path" make -C "$opensbi" O="$ram_opensbi" clean
PATH="$build_path" make -j"$(nproc)" -C "$opensbi" \
    O="$ram_opensbi" \
    CROSS_COMPILE="$cross" \
    PLATFORM=generic \
    PLATFORM_RISCV_ISA=rv64imac_zicsr_zifencei \
    PLATFORM_RISCV_ABI=lp64 \
    FW_PAYLOAD_PATH="$kernel/arch/riscv/boot/Image" \
    FW_FDT_PATH="$output/cv64a6_ram.dtb" \
    FW_PAYLOAD_FDT_ADDR="$(printf '0x%x' "$fdt_addr")"

install -m 0644 "$ram_opensbi/platform/generic/firmware/fw_payload.bin" "$output/ram-linux.bin"
install -m 0644 "$ram_opensbi/platform/generic/firmware/fw_payload.elf" "$output/ram-linux.elf"
sha256sum "$output/ram-linux.bin"
du -h "$output/ram-linux.bin"
