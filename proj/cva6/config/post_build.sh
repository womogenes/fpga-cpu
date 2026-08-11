#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compiler="$HOST_DIR/bin/riscv64-buildroot-linux-gnu-gcc"
software="$root/software"
programs="$TARGET_DIR/usr/bin"
examples="$TARGET_DIR/root/examples"
media="$TARGET_DIR/root/media"
target_arch=(-march=rv64imafc_zicsr_zifencei_zbc_zicntr_zcb_zihpm -mabi=lp64)
cflags=(-O3 "${target_arch[@]}" -Wall -Wextra -Werror -std=gnu11 -I"$software")
inference="${LFM_INFERENCE_ROOT:-$root/../../../inference}"
llama_root="${BONSAI_LLAMA_ROOT:-$root/../../../llama.cpp}"
lfm_model="$inference/models/LFM2.5-350M-Q4_0.gguf"
bonsai_model="${BONSAI_MODEL:-$llama_root/../Bonsai-1.7B.gguf}"
lfm_main="$inference/main"
lfm_build="$root/build/lfm-runtime"

mkdir -p "$programs" "$examples" "$media"
"$compiler" "${cflags[@]}" "$software/videoctl.c" "$software/video_hw.c" -o "$programs/videoctl"
"$compiler" "${cflags[@]}" "$software/video_bench.c" "$software/video_hw.c" -o "$programs/video-bench"
"$compiler" "${cflags[@]}" "$software/mandelbrot.c" "$software/video_hw.c" -pthread -lm -o "$programs/mandelbrot"
"$compiler" "${cflags[@]}" "$software/life.c" "$software/video_hw.c" -o "$programs/life"
"$compiler" "${cflags[@]}" "$software/badapple.c" "$software/video_hw.c" -lz -o "$programs/badapple"
"$compiler" "${cflags[@]}" "$software/snake.c" -o "$programs/snake"
"$compiler" "${cflags[@]}" "$software/model_receive.c" -o "$programs/model-receive"
"$compiler" "${cflags[@]}" "$software/model_export.c" -o "$programs/model-export"
if [[ ! -f "$lfm_model" || ! -f "$lfm_main/main.c" || ! -f "$lfm_main/generated/model_manifest.h" ]]; then
    echo "Standalone LFM sources or model are missing under $inference." >&2
    echo "Set LFM_INFERENCE_ROOT to the inference checkout before building the image." >&2
    exit 1
fi
mkdir -p "$lfm_build"
cp "$lfm_main/main.c" "$lfm_build/main.c"
patch --quiet --forward "$lfm_build/main.c" <"$root/config/lfm_accelerator.patch"
"$compiler" -O3 -ffast-math "${target_arch[@]}" -Wall -Wextra -Werror -Wno-unused-function -Wshadow -Wconversion -std=gnu11 -pthread \
    -DLFM_HOST -DLFM_ACCELERATOR -DLFM_FAST_MATH -DLFM_DIRECT_MODEL_PHYS=0xaf000000ul -DLFM_MAX_CONTEXT=256 \
    -I"$software" -I"$lfm_main/generated" "$lfm_build/main.c" -lm -o "$programs/lfm"
"$HOST_DIR/bin/riscv64-buildroot-linux-gnu-strip" "$programs/videoctl" "$programs/video-bench" "$programs/mandelbrot" "$programs/life" "$programs/badapple" "$programs/snake" "$programs/model-receive" "$programs/model-export"
"$HOST_DIR/bin/riscv64-buildroot-linux-gnu-strip" "$programs/lfm"

install -m 0644 "$software/video_hw.h" "$software/video_hw.c" "$software/video_bench.c" "$software/mandelbrot.c" "$software/life.c" "$software/badapple.c" "$software/snake.c" "$examples/"
install -m 0755 "$software/numpy_bench.py" "$programs/numpy-bench"
install -m 0755 "$software/lfm_chat.py" "$programs/lfm-chat"
install -m 0755 "$software/sd_install.sh" "$TARGET_DIR/usr/sbin/sd-install"
"$root/tools/build_bonsai_runtime.sh" "$TARGET_DIR" "$llama_root"
install -m 0755 "$software/bonsai_chat.sh" "$programs/bonsai-runtime-chat"
install -m 0755 "$software/bonsai_chat.py" "$programs/bonsai-chat"
mkdir -p "$TARGET_DIR/usr/share/lfm" "$TARGET_DIR/usr/share/bonsai"
rm -f "$TARGET_DIR/etc/init.d/S99model-export"
python3 "$root/tools/build_lfm_tokenizer.py" "$lfm_model" "$TARGET_DIR/usr/share/lfm/tokenizer.pickle"
python3 "$root/tools/build_lfm_tokenizer.py" "$bonsai_model" "$TARGET_DIR/usr/share/bonsai/tokenizer.pickle"

badapple_asset="$root/../sw/bad_apple/build/video.bin"
if [[ ! -f "$badapple_asset" ]]; then
    python3 "$root/../sw/bad_apple/video_to_bin.py" "$root/../sw/bad_apple/assets/badapple.mp4" "$badapple_asset" --width 160 --height 90 --fps 30 --threshold 128
fi
install -m 0644 "$badapple_asset" "$media/badapple.mvz"

"$root/tools/build_tinycc.sh" "$TARGET_DIR" "$STAGING_DIR" "$HOST_DIR"
ln -snf python3 "$programs/python"

# The RAM-boot image has only 1 GiB available while Linux is unpacking its
# built-in initramfs. TinyCC deliberately restores static system libraries for
# on-board C builds, so perform this targeted cleanup afterward. Keep the
# shared OpenBLAS runtime used by NumPy, but omit its redundant static archive
# and NumPy's development test suite.
find "$TARGET_DIR/usr/lib" -maxdepth 1 -name 'libopenblas*.a' -delete
find "$TARGET_DIR"/usr/lib/python*/site-packages/numpy -type d -name tests -prune -exec rm -rf {} +

# Buildroot does not uninstall target files when a package is disabled in an
# incremental configuration. Purge the old IPython stack so the next image
# matches buildroot64_defconfig without requiring a full toolchain rebuild.
rm -f "$programs/ipython" "$programs/ipython3" "$TARGET_DIR/share/man/man1/ipython.1" "$TARGET_DIR/usr/share/man/man1/ipython.1"
for site_packages in "$TARGET_DIR"/usr/lib/python*/site-packages; do
    [[ -d "$site_packages" ]] || continue
    rm -rf \
        "$site_packages/IPython" \
        "$site_packages/asttokens" "$site_packages"/asttokens-*.dist-info \
        "$site_packages/decorator.py" "$site_packages"/decorator-*.dist-info \
        "$site_packages/executing" "$site_packages"/executing-*.dist-info \
        "$site_packages/ipython-"*.dist-info \
        "$site_packages/ipython_pygments_lexers.py" "$site_packages"/ipython_pygments_lexers-*.dist-info \
        "$site_packages/jedi" "$site_packages"/jedi-*.dist-info \
        "$site_packages/matplotlib_inline" "$site_packages"/matplotlib_inline-*.dist-info \
        "$site_packages/parso" "$site_packages"/parso-*.dist-info \
        "$site_packages/pexpect" "$site_packages"/pexpect-*.dist-info \
        "$site_packages/prompt_toolkit" "$site_packages"/prompt_toolkit-*.dist-info \
        "$site_packages/ptyprocess" "$site_packages"/ptyprocess-*.dist-info \
        "$site_packages/pure_eval" "$site_packages"/pure_eval-*.dist-info \
        "$site_packages/pygments" "$site_packages"/pygments-*.dist-info \
        "$site_packages/stack_data" "$site_packages"/stack_data-*.dist-info \
        "$site_packages/traitlets" "$site_packages"/traitlets-*.dist-info \
        "$site_packages/wcwidth" "$site_packages"/wcwidth-*.dist-info
    rm -f "$site_packages/__pycache__"/ipython_pygments_lexers.*.pyc
done
if [[ -x "$TARGET_DIR/tetris" ]]; then
    mv "$TARGET_DIR/tetris" "$programs/tetris"
fi
if [[ -x "$programs/tetris" ]]; then
    ln -snf tetris "$programs/vitetris"
fi
if [[ -x "$programs/ascii_invaders" ]]; then
    ln -snf ascii_invaders "$programs/ascii-invaders"
fi
sed -i -E 's#^(root:[^:]*:[^:]*:[^:]*:[^:]*:[^:]*:)/bin/sh$#\1/bin/bash#' "$TARGET_DIR/etc/passwd"

# Mount devtmpfs before creating its child mount points. Mounting it from
# fstab would hide /dev/pts and /dev/shm immediately before mount -a needs them.
if ! grep -q '^::sysinit:/bin/mount -t devtmpfs devtmpfs /dev$' "$TARGET_DIR/etc/inittab"; then
    sed -i '\#^::sysinit:/bin/mkdir -p /dev/pts /dev/shm$#i::sysinit:/bin/mount -t devtmpfs devtmpfs /dev' "$TARGET_DIR/etc/inittab"
fi

# There is no battery-backed RTC on the current bring-up system.  Start from
# the image creation time so freshly installed files are not years in the
# future from make's point of view; NTP or an RTC can advance it later.
build_epoch="$(date +%s)"
cat >"$TARGET_DIR/etc/init.d/S01builddate" <<EOF
#!/bin/sh
date -s "@$build_epoch" >/dev/null 2>&1 || true
EOF
chmod 0755 "$TARGET_DIR/etc/init.d/S01builddate"
