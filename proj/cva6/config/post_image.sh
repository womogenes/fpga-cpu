#!/bin/sh
set -eu

config_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
its_template="$config_dir/fitImage.its.template"
its_file="$BINARIES_DIR/fitImage.its"
fit_binary="$BINARIES_DIR/fitImage.itb"
models_image="$BINARIES_DIR/models.ext4"
project_root=$(CDPATH= cd -- "$config_dir/../../../.." && pwd)
lfm_model=${LFM_MODEL:-"$project_root/inference/models/LFM2.5-350M-Q4_0.gguf"}
bonsai_model=${BONSAI_MODEL:-"$project_root/Bonsai-1.7B.gguf"}

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

# Keep large, persistent model weights outside both the root filesystem and
# the RAM initramfs. The dedicated ext4 partition remains writable on target,
# while the checked source models are copied directly into the generated image
# without adding them to Buildroot's target tree.
truncate -s 1G "$models_image"
"$HOST_DIR/sbin/mke2fs" -q -t ext4 -T largefile4 -O ^has_journal -F -m 0 -L models "$models_image"
if [ -f "$lfm_model" ]; then
    "$HOST_DIR/sbin/debugfs" -w -R "write $lfm_model /LFM2.5-350M-Q4_0.gguf" "$models_image"
else
    echo "LFM model not found at $lfm_model; generated models partition will omit it." >&2
fi
if [ -f "$bonsai_model" ]; then
    "$HOST_DIR/sbin/debugfs" -w -R "write $bonsai_model /Bonsai-1.7B.gguf" "$models_image"
else
    echo "Bonsai model not found at $bonsai_model; generated models partition will omit it." >&2
fi

support/scripts/genimage.sh -c "$config_dir/genimage.cfg"
