#!/bin/sh
set -eu

device=${SD_DEVICE:-/dev/mmcblk0}
boot_payload_dir=/usr/share/sd-installer
root_mount=/mnt/sd-root
model_mount=/mnt/sd-models
payload_sectors=8192
boot_sectors=65536
models_start=1386496

usage() {
    cat <<'EOF'
usage: sd-install prepare [DEVICE]
       sd-install refresh [DEVICE]
       sd-install model {lfm|bonsai} [DEVICE]
       sd-install status [DEVICE]

prepare repartitions DEVICE, installs the current Linux root filesystem, and
installs OpenSBI/U-Boot plus the FIT image. It destroys all data on DEVICE.

refresh replaces the boot payloads and root filesystem on a card previously
created by prepare, while preserving its models partition.

model copies the selected GGUF from reserved DDR into the persistent models
partition. Boot the RAM image with that model staged before running it.
EOF
}

part() {
    case "$device" in
        *[0-9]) printf '%sp%s' "$device" "$1" ;;
        *) printf '%s%s' "$device" "$1" ;;
    esac
}

require_root_and_device() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "sd-install: must run as root" >&2
        exit 1
    fi
    if [ ! -b "$device" ]; then
        echo "sd-install: block device not found: $device" >&2
        exit 1
    fi
}

require_ram_install_source() {
    if [ ! -f "$boot_payload_dir/fw_payload.bin" ] || [ ! -f "$boot_payload_dir/fitImage.itb" ]; then
        echo "sd-install: this RAM image does not contain the SD boot payloads" >&2
        exit 1
    fi
    payload_bytes=$(wc -c <"$boot_payload_dir/fw_payload.bin")
    fit_bytes=$(wc -c <"$boot_payload_dir/fitImage.itb")
    if [ "$payload_bytes" -gt $((payload_sectors * 512)) ] || [ "$fit_bytes" -gt $((boot_sectors * 512)) ]; then
        echo "sd-install: boot payload is larger than its prepared partition" >&2
        exit 1
    fi
    root_source=$(findmnt -rn -o SOURCE /)
    root_fstype=$(findmnt -rn -o FSTYPE /)
    case "$root_fstype" in
        rootfs|tmpfs) ;;
        *)
            echo "sd-install: system installation must run from the RAM filesystem, not $root_fstype" >&2
            exit 1
            ;;
    esac
    case "$root_source" in
        "$device"*)
            echo "sd-install: refusing to overwrite the device containing /" >&2
            exit 1
            ;;
    esac
}

wait_for_partitions() {
    count=0
    while [ ! -b "$(part 4)" ]; do
        count=$((count + 1))
        if [ "$count" -ge 50 ]; then
            echo "sd-install: partition devices did not appear" >&2
            exit 1
        fi
        sleep 0.1
    done
}

unmount_card() {
    index=4
    while [ "$index" -ge 1 ]; do
        partition=$(part "$index")
        target=$(findmnt -rn -S "$partition" -o TARGET 2>/dev/null || true)
        if [ -n "$target" ]; then
            umount "$target"
        fi
        index=$((index - 1))
    done
}

copy_live_root() {
    mkdir -p "$root_mount"
    # Installation is followed by sync and a clean unmount, so batching
    # metadata commits avoids forcing the SPI card to flush its cache while
    # every small Python and development file is copied. Normal SD boots use
    # the conservative options from /etc/fstab.
    mount -o noatime,commit=600,barrier=0 "$(part 3)" "$root_mount"
    for source in /*; do
        name=${source#/}
        case "$name" in
            dev|proc|sys|run|tmp|mnt|lost+found) continue ;;
            root)
                mkdir -p "$root_mount/root"
                for root_source in /root/.[!.]* /root/..?* /root/*; do
                    [ -e "$root_source" ] || continue
                    [ "${root_source##*/}" = models ] && continue
                    cp -a "$root_source" "$root_mount/root/"
                done
                ;;
            *) cp -a "$source" "$root_mount/" ;;
        esac
    done
    rm -rf "$root_mount/usr/share/sd-installer"
    mkdir -p "$root_mount/dev" "$root_mount/proc" "$root_mount/sys" "$root_mount/run" "$root_mount/tmp" "$root_mount/mnt" "$root_mount/root/models"
    chmod 0755 "$root_mount/dev" "$root_mount/proc" "$root_mount/sys" "$root_mount/run" "$root_mount/mnt" "$root_mount/root/models"
    chmod 1777 "$root_mount/tmp"
    sync
    umount "$root_mount"
}

write_boot_and_root() {
    payload_partition_bytes=$(blockdev --getsize64 "$(part 1)")
    boot_partition_bytes=$(blockdev --getsize64 "$(part 2)")
    if [ "$payload_bytes" -gt "$payload_partition_bytes" ] || [ "$fit_bytes" -gt "$boot_partition_bytes" ]; then
        echo "sd-install: boot payload does not fit the existing card partitions" >&2
        exit 1
    fi
    dd if="$boot_payload_dir/fw_payload.bin" of="$(part 1)" bs=1M conv=fsync
    mkfs.fat -F 16 -n BOOT "$(part 2)"
    mke2fs -q -t ext4 -F -m 0 -E nodiscard,lazy_itable_init=1,lazy_journal_init=1 -L rootfs "$(part 3)"

    mkdir -p "$root_mount/boot"
    mount "$(part 2)" "$root_mount/boot"
    cp "$boot_payload_dir/fitImage.itb" "$root_mount/boot/fitImage.itb"
    sync
    umount "$root_mount/boot"
    rmdir "$root_mount/boot"

    copy_live_root
}

require_prepared_card() {
    wait_for_partitions
    root_label=$(blkid -s LABEL -o value "$(part 3)" 2>/dev/null || true)
    model_label=$(blkid -s LABEL -o value "$(part 4)" 2>/dev/null || true)
    if [ "$root_label" != rootfs ] || [ "$model_label" != models ]; then
        echo "sd-install: $device does not contain prepared rootfs and models partitions" >&2
        exit 1
    fi
}

prepare() {
    require_root_and_device
    require_ram_install_source
    sectors=$(blockdev --getsz "$device")
    if [ "$sectors" -lt $((models_start + 1048576)) ]; then
        echo "sd-install: card must be at least 1.2 GiB" >&2
        exit 1
    fi
    echo "Installing the live system onto $device; all existing card data will be lost."
    unmount_card
    # Keep every payload and filesystem on a 1 MiB boundary. Besides avoiding
    # read-modify-write amplification in the SD flash, this exactly matches
    # the image generated from config/genimage.cfg.
    sfdisk --wipe always "$device" <<'EOF'
label: gpt
unit: sectors
first-lba: 34

start=2048, size=8192, type=AAA01589-8CD3-5F10-8785-832FFBC41D64, name="OpenSBI with u-boot as payload"
start=10240, size=65536, type=L, name="FIT image"
start=75776, size=1310720, type=L, name="rootfs"
start=1386496, type=L, name="models"
EOF
    # sfdisk normally asks the kernel to re-read the table. Repeat that
    # explicitly and rescan mdev so this works in the minimal Buildroot image
    # without pulling in GNU Parted solely for partprobe.
    blockdev --rereadpt "$device"
    mdev -s
    wait_for_partitions

    # This partition holds only a handful of large immutable weight files.
    # largefile4 avoids allocating and lazily clearing roughly two million
    # unnecessary inodes on a 32 GB card through the relatively slow SPI bus.
    # Models are immutable after an atomic copy-and-rename. A journal adds a
    # large amount of first-mount I/O on this SPI-connected card without
    # improving that update protocol, so use journal-free ext4 here.
    mke2fs -q -t ext4 -T largefile4 -O ^has_journal -F -m 0 -E nodiscard,lazy_itable_init=1 -L models "$(part 4)"
    write_boot_and_root
    echo "SD system installed. Run 'sd-install model lfm' or 'sd-install model bonsai' before rebooting."
}

refresh() {
    require_root_and_device
    require_ram_install_source
    require_prepared_card
    echo "Refreshing the boot and root filesystems on $device; persistent models will be preserved."
    unmount_card
    write_boot_and_root
    echo "SD system refreshed; the models partition was preserved."
}

install_model() {
    require_root_and_device
    kind=$1
    case "$kind" in
        lfm)
            filename=LFM2.5-350M-Q4_0.gguf
            size=219309792
            expected=85e32858daafad55b7bcd6b97a1343ee0661188e8036f9862d14d6b563142f50
            ;;
        bonsai)
            filename=Bonsai-1.7B.gguf
            size=248302272
            expected=3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3
            ;;
        *)
            echo "sd-install: unknown model: $kind" >&2
            usage >&2
            exit 1
            ;;
    esac
    wait_for_partitions
    partition=$(part 4)
    label=$(blkid -s LABEL -o value "$partition" 2>/dev/null || true)
    if [ "$label" != models ]; then
        echo "sd-install: $partition is not the prepared models filesystem" >&2
        exit 1
    fi
    destination=$(findmnt -rn -S "$partition" -o TARGET 2>/dev/null || true)
    mounted_here=0
    if [ -z "$destination" ]; then
        mkdir -p "$model_mount"
        mount -o noatime "$partition" "$model_mount"
        destination=$model_mount
        mounted_here=1
    fi
    temporary="$destination/$filename.partial"
    rm -f "$temporary"
    model-export "$size" "$temporary"
    actual=$(sha256sum "$temporary" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        rm -f "$temporary"
        [ "$mounted_here" -eq 0 ] || umount "$model_mount"
        echo "sd-install: staged $kind model hash mismatch: $actual" >&2
        exit 1
    fi
    mv "$temporary" "$destination/$filename"
    sync
    [ "$mounted_here" -eq 0 ] || umount "$model_mount"
    echo "Installed $filename ($actual)."
}

status() {
    require_root_and_device
    sfdisk -d "$device"
    partition=$(part 4)
    if [ -b "$partition" ]; then
        target=$(findmnt -rn -S "$partition" -o TARGET 2>/dev/null || true)
        mounted_here=0
        if [ -z "$target" ]; then
            mkdir -p "$model_mount"
            mount -o ro "$partition" "$model_mount"
            target=$model_mount
            mounted_here=1
        fi
        ls -lh "$target"
        [ "$mounted_here" -eq 0 ] || umount "$model_mount"
    fi
}

command=${1:-}
case "$command" in
    prepare)
        [ "$#" -le 2 ] || { usage >&2; exit 1; }
        [ "$#" -lt 2 ] || device=$2
        prepare
        ;;
    refresh)
        [ "$#" -le 2 ] || { usage >&2; exit 1; }
        [ "$#" -lt 2 ] || device=$2
        refresh
        ;;
    model)
        [ "$#" -ge 2 ] && [ "$#" -le 3 ] || { usage >&2; exit 1; }
        [ "$#" -lt 3 ] || device=$3
        install_model "$2"
        ;;
    status)
        [ "$#" -le 2 ] || { usage >&2; exit 1; }
        [ "$#" -lt 2 ] || device=$2
        status
        ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 1 ;;
esac
