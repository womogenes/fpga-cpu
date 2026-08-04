#!/bin/sh
set -eu

config_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
its_template="$config_dir/fitImage.its.template"
its_file="$BINARIES_DIR/fitImage.its"
fit_binary="$BINARIES_DIR/fitImage.itb"

if grep -q '^BR2_RISCV_64=y' "$BR2_CONFIG"; then
    kernel_load_address=0x80200000
    kernel_entry_address=0x80200000
    image_name=CV64A6Linux
else
    echo "The local CVA6 image supports only RV64." >&2
    exit 1
fi

sed -e "s/%IMAGE_NAME%/$image_name/" \
    -e "s/%KERNEL_LOAD_ADDRESS%/$kernel_load_address/" \
    -e "s/%KERNEL_ENTRY_ADDRESS%/$kernel_entry_address/" \
    -e "s;%BINARIES_DIR%;$BINARIES_DIR;" \
    "$its_template" > "$its_file"

output/host/bin/mkimage -f "$its_file" "$fit_binary"
support/scripts/genimage.sh -c "$config_dir/genimage.cfg"
